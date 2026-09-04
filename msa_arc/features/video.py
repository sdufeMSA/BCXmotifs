"""Video branch preprocessing (Section 5.1.1, Appendix A.2.1).

Frames are decoded at 10 fps, faces detected and cropped, normalised to
224x224, and encoded by an ImageNet-pretrained ResNet-50 into 2048-dimensional
per-frame features.  The ResNet-50 is frozen and used purely as a fixed feature
extractor, so this runs once in Stage A and never again during training.

Releasing these tensors rather than the frames is what makes the study's data
statement workable: the recordings carry identifiable faces, the pooled
ResNet-50 features do not reconstruct them.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from msa_arc.config import VideoFeatureConfig
from msa_arc.features.base import FeatureExtractor

logger = logging.getLogger(__name__)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class VideoFeatureExtractor(FeatureExtractor):
    """Face-cropped ResNet-50 frame-sequence extractor.

    Args:
        cfg: Video preprocessing configuration.
        device: Torch device string for the ResNet-50 forward pass.
    """

    modality = "video"

    def __init__(self, cfg: VideoFeatureConfig, device: str = "cpu") -> None:
        self.cfg = cfg
        self.device = device
        self._encoder = None
        self._transform = None
        self._detector = None

    # lazily constructed heavy components

    def _build_encoder(self):
        import torch
        from torchvision import models

        weights = models.ResNet50_Weights.IMAGENET1K_V2
        resnet = models.resnet50(weights=weights)
        # Drop the classifier: the 2048-d pooled feature is what we want.
        resnet.fc = torch.nn.Identity()
        resnet.eval().to(self.device)
        for parameter in resnet.parameters():
            parameter.requires_grad_(False)
        return resnet

    def _build_transform(self):
        from torchvision import transforms

        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((self.cfg.image_size, self.cfg.image_size), antialias=True),
                transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ]
        )

    @property
    def encoder(self):
        if self._encoder is None:
            self._encoder = self._build_encoder()
        return self._encoder

    @property
    def transform(self):
        if self._transform is None:
            self._transform = self._build_transform()
        return self._transform

    # frame handling

    def read_frames(
        self, path: str, start: Optional[float], end: Optional[float]
    ) -> List[np.ndarray]:
        """Decode frames at the configured rate, optionally within a segment.

        Args:
            path: Video file path.
            start: Segment start in seconds, or ``None`` for the whole clip.
            end: Segment end in seconds.

        Returns:
            RGB frames as ``(H, W, 3)`` uint8 arrays.

        Raises:
            ImportError: If OpenCV is not installed.
            IOError: If the video cannot be opened.
        """
        try:
            import cv2
        except ImportError as error:  # pragma: no cover - environment dependent
            raise ImportError(
                "opencv-python is required for video feature extraction; "
                "install the 'features' extra"
            ) from error

        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            raise OSError(f"cannot open video: {path}")

        try:
            source_fps = capture.get(cv2.CAP_PROP_FPS) or float(self.cfg.fps)
            step = max(int(round(source_fps / self.cfg.fps)), 1)
            first = 0 if start is None else int(round(start * source_fps))
            last = None if end is None else int(round(end * source_fps))

            capture.set(cv2.CAP_PROP_POS_FRAMES, first)
            frames: List[np.ndarray] = []
            index = first
            while len(frames) < self.cfg.max_frames:
                ok, frame = capture.read()
                if not ok or (last is not None and index >= last):
                    break
                if (index - first) % step == 0:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                index += 1
            return frames
        finally:
            capture.release()

    def crop_face(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect and crop the largest face in a frame.

        Args:
            frame: RGB frame.

        Returns:
            The cropped face, or ``None`` when no face is detected. A frame
            without a face is dropped rather than passed through whole: an
            uncropped frame would feed background rather than expression into
            the branch.
        """
        detector = self._get_detector()
        if detector is None:
            return frame
        box = detector(frame)
        if box is None:
            return None
        top, right, bottom, left = box
        return frame[top:bottom, left:right]

    def _get_detector(self):
        """Build the configured face detector, or ``None`` to skip cropping."""
        if self._detector is not None:
            return self._detector
        if self.cfg.face_detector == "none":
            return None
        try:
            import face_recognition

            def detect(frame: np.ndarray):
                locations = face_recognition.face_locations(frame)
                if not locations:
                    return None
                # Largest detected face: the interviewee fills more of the frame
                # than anyone incidentally in shot.
                return max(
                    locations,
                    key=lambda b: (b[2] - b[0]) * (b[1] - b[3]),
                )

            self._detector = detect
        except ImportError:  # pragma: no cover - environment dependent
            logger.warning(
                "face detection backend unavailable; frames are encoded uncropped, "
                "which departs from the preprocessing the paper reports"
            )
            self._detector = None
        return self._detector

    def encode_frames(self, frames: List[np.ndarray]) -> np.ndarray:
        """Run cropped frames through the frozen ResNet-50.

        Args:
            frames: Cropped RGB frames.

        Returns:
            ``(n_frames, 2048)`` float32 features.
        """
        import torch

        batch = torch.stack([self.transform(frame) for frame in frames]).to(self.device)
        with torch.no_grad():
            features = self.encoder(batch)
        return features.cpu().numpy().astype(np.float32)

    def extract(self, source: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
        """Extract the frame-feature sequence for one instance.

        Args:
            source: ``{"path": ..., "start": ..., "end": ...}``, or ``None``.

        Returns:
            ``(frames, 2048)`` float32 array, or ``None`` when the instance has
            no video or no frame contained a detectable face.
        """
        if source is None:
            return None
        frames = self.read_frames(source["path"], source.get("start"), source.get("end"))
        if not frames:
            logger.warning("no frames decoded for %s; skipping", source["path"])
            return None

        cropped = [face for face in (self.crop_face(f) for f in frames) if face is not None]
        if not cropped:
            logger.warning("no face detected in any frame of %s; skipping", source["path"])
            return None
        if len(cropped) < len(frames):
            logger.debug(
                "dropped %d of %d frames without a detected face",
                len(frames) - len(cropped),
                len(frames),
            )
        return self.encode_frames(cropped)


__all__ = ["VideoFeatureExtractor"]
