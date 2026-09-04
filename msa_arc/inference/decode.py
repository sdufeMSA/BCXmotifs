"""Decoding the unified target string, with a fixed and reportable fallback.

Decoding proceeds in three stages, so that the parse-failure rate is a
measured quantity:

1. beam search with ``num_beams`` beams, no sampling;
2. any sequence that fails to parse is re-decoded greedily;
3. any sequence that still fails takes the argmax of the constrained rescoring,
   which is parseable by construction.

Every instance that reaches stage 2 or 3 is recorded, so a run can report
exactly how often each happened.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch
from transformers.modeling_outputs import BaseModelOutput

from msa_arc.config import DecodeConfig
from msa_arc.constants import ATTITUDE_CLASSES
from msa_arc.inference.probabilities import category_probabilities
from msa_arc.model.target import parse_target

logger = logging.getLogger(__name__)


@dataclass
class InstancePrediction:
    """One instance's model output, before MCVA reconciliation.

    Attributes:
        instance_key: Identifier from the manifest.
        participant_id: Owning participant.
        service_id: Service the instance is about.
        scenario: 1 for functional, 0 for dysfunctional.
        polarity: Decoded sentiment polarity.
        intensity: Decoded sentiment intensity on ``[-1, 1]``.
        category: Decoded surface attitude category.
        raw_text: The generated string, kept for the failure audit.
        decode_stage: ``beam``, ``greedy`` or ``fallback``.
        probabilities: Five-class distribution from constrained rescoring.
    """

    instance_key: str
    participant_id: str
    service_id: str
    scenario: int
    polarity: str
    intensity: float
    category: str
    raw_text: str
    decode_stage: str
    probabilities: List[float] = field(default_factory=list)


@dataclass
class DecodeReport:
    """Parse-failure accounting for one evaluation pass.

    Attributes:
        n_instances: How many instances were decoded.
        n_greedy_retries: How many needed the greedy re-decode.
        n_fallbacks: How many still failed and fell back to rescoring.
        failures: The raw strings that failed, for the audit log.
    """

    n_instances: int = 0
    n_greedy_retries: int = 0
    n_fallbacks: int = 0
    failures: List[Dict[str, str]] = field(default_factory=list)

    @property
    def parse_failure_rate(self) -> float:
        """Share of instances whose beam output did not parse."""
        if self.n_instances == 0:
            return 0.0
        return self.n_greedy_retries / self.n_instances

    def merge(self, other: "DecodeReport") -> None:
        """Accumulate another batch's report into this one."""
        self.n_instances += other.n_instances
        self.n_greedy_retries += other.n_greedy_retries
        self.n_fallbacks += other.n_fallbacks
        self.failures.extend(other.failures)


def _decode_strings(
    model: Any,
    tokenizer: Any,
    batch: Dict[str, Any],
    encoder_outputs: BaseModelOutput,
    num_beams: int,
    cfg: DecodeConfig,
) -> List[str]:
    """Run one generation pass and detokenise the result."""
    generated = model.generate(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        encoder_outputs=encoder_outputs,
        num_beams=num_beams,
        do_sample=False,
        max_new_tokens=cfg.max_new_tokens,
        length_penalty=cfg.length_penalty,
        early_stopping=cfg.early_stopping if num_beams > 1 else False,
    )
    return tokenizer.batch_decode(generated, skip_special_tokens=True)


@torch.no_grad()
def decode_batch(
    model: Any,
    tokenizer: Any,
    batch: Dict[str, Any],
    cfg: Optional[DecodeConfig] = None,
) -> tuple[List[InstancePrediction], DecodeReport]:
    """Decode one batch into parsed predictions with class probabilities.

    Args:
        model: A :class:`~msa_arc.model.mul_mt5.MulMT5` in eval mode.
        tokenizer: Tokenizer used for the targets.
        batch: A collated batch already moved to the model's device.
        cfg: Decoding configuration.

    Returns:
        The per-instance predictions and the parse-failure report for the batch.
    """
    cfg = cfg or DecodeConfig()
    encoder_outputs = model.encode(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        audio_features=batch.get("audio_features"),
        audio_lengths=batch.get("audio_lengths"),
        video_features=batch.get("video_features"),
        video_lengths=batch.get("video_lengths"),
    )

    texts = _decode_strings(model, tokenizer, batch, encoder_outputs, cfg.num_beams, cfg)
    parsed = [parse_target(text) for text in texts]
    stages = ["beam"] * len(parsed)
    report = DecodeReport(n_instances=len(parsed))

    retry_indices = [i for i, result in enumerate(parsed) if not result.ok]
    if retry_indices:
        report.n_greedy_retries = len(retry_indices)
        logger.debug("Re-decoding %d unparseable sequences greedily", len(retry_indices))
        greedy_texts = _decode_strings(model, tokenizer, batch, encoder_outputs, 1, cfg)
        for index in retry_indices:
            retried = parse_target(greedy_texts[index])
            if retried.ok:
                parsed[index] = retried
                stages[index] = "greedy"
            else:
                report.failures.append(
                    {
                        "instance_key": batch["instance_key"][index],
                        "beam": texts[index],
                        "greedy": greedy_texts[index],
                    }
                )
                stages[index] = "fallback"

    # Instances still unparsed take a neutral prefix so that rescoring has a
    # well-defined context; their category then comes from the rescoring argmax.
    polarities = [r.polarity if r.ok else "neutral" for r in parsed]
    intensities = [r.intensity if r.ok else 0.0 for r in parsed]

    probabilities = category_probabilities(
        model=model,
        encoder_outputs=encoder_outputs,
        attention_mask=batch["attention_mask"],
        polarities=polarities,
        intensities=intensities,
        tokenizer=tokenizer,
        temperature=cfg.probability_temperature,
    )
    argmax_categories = [ATTITUDE_CLASSES[i] for i in probabilities.argmax(dim=-1).tolist()]

    predictions: List[InstancePrediction] = []
    for index, result in enumerate(parsed):
        if result.ok:
            category = result.category
        else:
            category = argmax_categories[index]
            report.n_fallbacks += 1
        predictions.append(
            InstancePrediction(
                instance_key=batch["instance_key"][index],
                participant_id=batch["participant_id"][index],
                service_id=batch["service_id"][index],
                scenario=int(batch["scenario"][index]),
                polarity=polarities[index],
                intensity=float(intensities[index]),
                category=category,
                raw_text=texts[index],
                decode_stage=stages[index],
                probabilities=probabilities[index].tolist(),
            )
        )
    return predictions, report


@torch.no_grad()
def decode_dataset(
    model: Any,
    tokenizer: Any,
    loader: Sequence[Dict[str, Any]],
    device: Any,
    cfg: Optional[DecodeConfig] = None,
) -> tuple[List[InstancePrediction], DecodeReport]:
    """Decode every batch a loader yields.

    Args:
        model: The model, which is put into eval mode.
        tokenizer: Tokenizer used for the targets.
        loader: An iterable of collated batches.
        device: Device to move batches onto.
        cfg: Decoding configuration.

    Returns:
        All predictions in loader order, and the aggregated report.
    """
    from msa_arc.data.collate import move_to_device

    model.eval()
    predictions: List[InstancePrediction] = []
    report = DecodeReport()
    for batch in loader:
        batch_predictions, batch_report = decode_batch(
            model, tokenizer, move_to_device(batch, device), cfg
        )
        predictions.extend(batch_predictions)
        report.merge(batch_report)

    logger.info(
        "Decoded %d instances: %d greedy retries (%.3f%%), %d fallbacks",
        report.n_instances,
        report.n_greedy_retries,
        100 * report.parse_failure_rate,
        report.n_fallbacks,
    )
    return predictions, report


__all__ = ["DecodeReport", "InstancePrediction", "decode_batch", "decode_dataset"]
