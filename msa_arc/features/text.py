"""Text branch preprocessing (Section 5.1.1).

Jieba segmentation runs before the backbone's own SentencePiece tokenisation for
two domain reasons stated in the manuscript.  Verbatim transcripts of elderly
speech contain long unpunctuated runs, and the word boundaries Jieba recovers
let us insert sentence delimiters the subword tokeniser would otherwise have no
cue for.  The particles removed are semantically empty fillers characteristic of
hesitant elderly speech, and keeping them lengthens the sequence without adding
attitude-relevant content.

Both steps operate on the character string only.  The model input is still
produced by the backbone's own subword tokeniser, so no vocabulary mismatch is
introduced.
"""

import logging
import re
from typing import Any, List, Optional, Sequence

import numpy as np

from msa_arc.config import TextFeatureConfig
from msa_arc.features.base import FeatureExtractor

logger = logging.getLogger(__name__)

#: Punctuation and symbols stripped before tokenisation. Sentence-ending marks
#: are kept: they are the delimiters the segmentation step exists to recover.
_SYMBOL_RE = re.compile(r"[^\w一-鿿。！？，、；：\s]+")
_WHITESPACE_RE = re.compile(r"\s+")

_SENTENCE_DELIMITER = "。"


def segment(text: str, use_jieba: bool = True) -> List[str]:
    """Segment Mandarin text into words.

    Args:
        text: Raw transcript.
        use_jieba: When ``False``, falls back to whitespace splitting, which
            keeps the module importable in environments without Jieba.

    Returns:
        The token list.
    """
    if not use_jieba:
        return text.split()
    try:
        import jieba
    except ImportError:
        logger.warning("jieba not installed; falling back to whitespace segmentation")
        return text.split()
    return [token for token in jieba.cut(text) if token.strip()]


def clean_transcript(
    text: str,
    particles: Sequence[str] = (),
    strip_symbols: bool = True,
    use_jieba: bool = True,
) -> str:
    """Apply the manuscript's text preprocessing to one transcript.

    Args:
        text: Raw verbatim transcript.
        particles: Discourse particles to drop.
        strip_symbols: Whether to remove non-word symbols.
        use_jieba: Whether to segment with Jieba.

    Returns:
        The cleaned string, with recovered word boundaries joined by spaces and
        a sentence delimiter appended if the transcript ends without one.
    """
    if strip_symbols:
        text = _SYMBOL_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    if not text:
        return ""

    tokens = segment(text, use_jieba=use_jieba)
    particle_set = set(particles)
    kept = [token for token in tokens if token not in particle_set]
    if not kept:
        return ""

    cleaned = " ".join(kept)
    if cleaned[-1] not in "。！？":
        cleaned += _SENTENCE_DELIMITER
    return cleaned


class TextFeatureExtractor(FeatureExtractor):
    """Clean a transcript and encode it to padded mT5 token ids.

    Args:
        cfg: Text preprocessing configuration.
        tokenizer: A tokenizer exposing HuggingFace's ``__call__`` interface.
            Injected so tests can supply a stub and run offline.
    """

    modality = "text"

    def __init__(self, cfg: TextFeatureConfig, tokenizer: Any) -> None:
        self.cfg = cfg
        self.tokenizer = tokenizer

    def clean(self, text: str) -> str:
        """Run the cleaning half of the pipeline on its own."""
        return clean_transcript(
            text,
            particles=self.cfg.discourse_particles,
            strip_symbols=self.cfg.strip_symbols,
            use_jieba=self.cfg.use_jieba,
        )

    def extract(self, source: Optional[str]) -> Optional[np.ndarray]:
        """Encode one transcript.

        Args:
            source: The verbatim transcript.

        Returns:
            ``(2, max_length)`` array stacking token ids and attention mask, or
            ``None`` when the transcript is empty after cleaning.
        """
        if source is None:
            return None
        cleaned = self.clean(str(source))
        if not cleaned:
            logger.warning("transcript is empty after cleaning; skipping instance")
            return None

        encoded = self.tokenizer(
            cleaned,
            max_length=self.cfg.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        ids = np.asarray(encoded["input_ids"][0], dtype=np.int64)
        mask = np.asarray(encoded["attention_mask"][0], dtype=np.int64)
        return np.stack([ids, mask], axis=0)


__all__ = ["TextFeatureExtractor", "clean_transcript", "segment"]
