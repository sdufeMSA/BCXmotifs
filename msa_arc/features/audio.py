"""Audio branch preprocessing (Section 5.1.1, Appendix A.2.1).

Noise reduction, resampling to 16 kHz, framing with a 25 ms window and 10 ms
hop, and conversion to a 40-bank Mel-spectrogram.  The result is the
``(frames, 40)`` array the audio LSTM consumes.

The extractor accepts either a pre-segmented clip or a full-session recording
plus offsets, matching the two layouts the manifest allows.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np

from msa_arc.config import AudioFeatureConfig
from msa_arc.features.base import FeatureExtractor

logger = logging.getLogger(__name__)


class AudioFeatureExtractor(FeatureExtractor):
    """Mel-spectrogram extractor for the prosodic channel.

    Args:
        cfg: Audio preprocessing configuration.
    """

    modality = "audio"

    def __init__(self, cfg: AudioFeatureConfig) -> None:
        self.cfg = cfg

    def load_waveform(self, path: str, start: Optional[float], end: Optional[float]):
        """Read a waveform, slicing a segment out of a session recording.

        Args:
            path: Audio file path.
            start: Segment start in seconds, or ``None`` for a whole clip.
            end: Segment end in seconds.

        Returns:
            A mono waveform resampled to ``cfg.sample_rate``.

        Raises:
            ImportError: If librosa is not installed.
        """
        try:
            import librosa
        except ImportError as error:  # pragma: no cover - environment dependent
            raise ImportError(
                "librosa is required for audio feature extraction; "
                "install the 'features' extra"
            ) from error

        offset = 0.0 if start is None else float(start)
        duration = None if (start is None or end is None) else float(end) - float(start)
        waveform, _ = librosa.load(
            path, sr=self.cfg.sample_rate, mono=True, offset=offset, duration=duration
        )
        return waveform

    def denoise(self, waveform: np.ndarray) -> np.ndarray:
        """Spectral-gating noise reduction.

        Falls back to the untouched waveform when ``noisereduce`` is absent, and
        says so, rather than silently changing the preprocessing.
        """
        if not self.cfg.denoise:
            return waveform
        try:
            import noisereduce
        except ImportError:  # pragma: no cover - environment dependent
            logger.warning(
                "noisereduce not installed; audio is used without noise reduction, "
                "which departs from the preprocessing the paper reports"
            )
            return waveform
        return noisereduce.reduce_noise(y=waveform, sr=self.cfg.sample_rate)

    def mel_spectrogram(self, waveform: np.ndarray) -> np.ndarray:
        """Convert a waveform to a log-Mel-spectrogram.

        Args:
            waveform: Mono waveform at ``cfg.sample_rate``.

        Returns:
            ``(frames, n_mels)`` log-power spectrogram, time-major to match the
            LSTM's ``batch_first`` layout.
        """
        import librosa

        n_fft = int(self.cfg.sample_rate * self.cfg.window_ms / 1000)
        hop_length = int(self.cfg.sample_rate * self.cfg.hop_ms / 1000)
        spectrogram = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.cfg.sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            n_mels=self.cfg.n_mels,
        )
        log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        return log_spectrogram.T.astype(np.float32)

    def extract(self, source: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
        """Extract the Mel-spectrogram for one instance.

        Args:
            source: ``{"path": ..., "start": ..., "end": ...}`` as returned by
                ``msa_arc.features.manifest.media_segment``, or ``None``.

        Returns:
            ``(frames, n_mels)`` float32 array truncated to ``cfg.max_frames``,
            or ``None`` when the instance has no audio.
        """
        if source is None:
            return None
        waveform = self.load_waveform(source["path"], source.get("start"), source.get("end"))
        if waveform.size == 0:
            logger.warning("empty audio segment for %s; skipping", source["path"])
            return None
        features = self.mel_spectrogram(self.denoise(waveform))
        if features.shape[0] > self.cfg.max_frames:
            features = features[: self.cfg.max_frames]
        return features


__all__ = ["AudioFeatureExtractor"]
