"""Training loop for MSA-ARC (Section 5.1.2, Appendix A.2.1).

AdamW at 1e-5 with weight decay 0.01, a linear warm-up over the first 10% of
steps followed by cosine decay, gradient clipping at 1.0, batch size 16, at most
50 epochs with early stopping on validation macro-F1 and a patience of 5.  The
checkpoint with the best validation macro-F1 is the one evaluated on test.

Model selection deliberately uses the **surface** attitude macro-F1, before MCVA
reconciliation.  The reconciliation thresholds and confidence bands are
themselves fitted on the validation split, and letting them also drive
early stopping would make the selected checkpoint depend on a rule fitted to the
same data in the same pass.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from msa_arc.config import DecodeConfig, LossConfig, TrainConfig
from msa_arc.data.collate import move_to_device
from msa_arc.evaluation.metrics import attitude_metrics
from msa_arc.inference.decode import decode_dataset
from msa_arc.losses.contrastive import contrastive_loss
from msa_arc.losses.generation import combined_loss, generation_loss

logger = logging.getLogger(__name__)


@dataclass
class TrainingHistory:
    """Per-epoch record of a run.

    Attributes:
        epochs: One entry per epoch with the loss terms and the selection metric.
        best_epoch: Index of the epoch whose checkpoint was kept.
        best_metric: Its validation macro-F1.
        stopped_early: Whether patience ran out before ``max_epochs``.
    """

    epochs: List[Dict[str, float]] = field(default_factory=list)
    best_epoch: int = -1
    best_metric: float = float("-inf")
    stopped_early: bool = False


def build_optimizer(model: torch.nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    """AdamW over the trainable parameters only.

    Biases are excluded from weight decay, which is standard and matters here
    because the LSTM branches carry a large share of the bias parameters.

    Args:
        model: The model.
        cfg: Training configuration.

    Returns:
        The optimizer.

    Raises:
        ValueError: If the configured optimizer is not supported.
    """
    if cfg.optimizer.lower() != "adamw":
        raise ValueError(f"unsupported optimizer {cfg.optimizer!r}; expected adamw")

    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (no_decay if name.endswith("bias") else decay).append(parameter)

    logger.info(
        "Optimising %d decayed and %d undecayed tensors",
        len(decay),
        len(no_decay),
    )
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.learning_rate,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer, total_steps: int, cfg: TrainConfig
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warm-up over the first ``warmup_ratio`` of steps, then cosine decay.

    Args:
        optimizer: The optimizer to schedule.
        total_steps: Total optimisation steps across all epochs.
        cfg: Training configuration.

    Returns:
        The scheduler.
    """
    warmup_steps = max(int(total_steps * cfg.warmup_ratio), 1)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        if cfg.lr_schedule == "constant":
            return 1.0
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def training_step(
    model: torch.nn.Module,
    batch: Dict[str, Any],
    loss_cfg: LossConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Forward pass and combined loss for one batch.

    Args:
        model: The model in train mode.
        batch: A collated batch on the model's device.
        loss_cfg: Loss weights and mining strategy.

    Returns:
        The scalar loss and a dict of its components for logging.
    """
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        audio_features=batch.get("audio_features"),
        audio_lengths=batch.get("audio_lengths"),
        video_features=batch.get("video_features"),
        video_lengths=batch.get("video_lengths"),
    )
    generation = generation_loss(outputs.logits, batch["labels"])
    contrastive = contrastive_loss(
        outputs.pooled,
        batch["label_category_index"],
        margin=loss_cfg.margin,
        negative_mining=loss_cfg.negative_mining,
    )
    total = combined_loss(generation, contrastive, loss_cfg.alpha, loss_cfg.beta)
    components = {
        "loss": float(total.detach()),
        "generation": float(generation.detach()),
        "contrastive": float(contrastive.detach())
        if contrastive is not None
        else float("nan"),
    }
    return total, components


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
    train_cfg: TrainConfig,
    loss_cfg: LossConfig,
) -> Dict[str, float]:
    """Run one epoch and return its mean loss components."""
    model.train()
    totals: Dict[str, float] = {}
    n_batches = 0
    n_without_contrastive = 0

    for batch in loader:
        batch = move_to_device(batch, device)
        loss, components = training_step(model, batch, loss_cfg)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], train_cfg.grad_clip
        )
        optimizer.step()
        scheduler.step()

        if math.isnan(components["contrastive"]):
            n_without_contrastive += 1
        for key, value in components.items():
            if not math.isnan(value):
                totals[key] = totals.get(key, 0.0) + value
        n_batches += 1

    if n_without_contrastive:
        logger.debug(
            "%d of %d batches had no anchor with an in-batch positive",
            n_without_contrastive,
            n_batches,
        )
    return {key: value / max(n_batches, 1) for key, value in totals.items()}


@torch.no_grad()
def selection_metric(
    model: torch.nn.Module,
    tokenizer: Any,
    loader: DataLoader,
    device: torch.device,
    decode_cfg: DecodeConfig,
) -> float:
    """Validation macro-F1 on the surface attitude categories.

    Args:
        model: The model.
        tokenizer: Tokenizer used for the targets.
        loader: Validation loader.
        device: Device to run on.
        decode_cfg: Decoding configuration, the same one used at test time so
            that the checkpoint is selected under the conditions it is reported
            under.

    Returns:
        Macro-averaged F1.
    """
    predictions, _ = decode_dataset(model, tokenizer, loader, device, decode_cfg)
    predicted = {p.instance_key: p.category for p in predictions}

    truth: List[str] = []
    guess: List[str] = []
    for batch in loader:
        for key, label in zip(batch["instance_key"], batch["label_category"], strict=False):
            if key in predicted:
                truth.append(label)
                guess.append(predicted[key])
    return attitude_metrics(truth, guess).macro_f1


def train_model(
    model: torch.nn.Module,
    tokenizer: Any,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_cfg: TrainConfig,
    loss_cfg: Optional[LossConfig] = None,
    decode_cfg: Optional[DecodeConfig] = None,
) -> Tuple[torch.nn.Module, TrainingHistory]:
    """Fit the model, keeping the best-validation checkpoint.

    Args:
        model: An initialised model.
        tokenizer: Tokenizer used for the targets.
        train_loader: Training batches.
        val_loader: Validation batches.
        train_cfg: Optimisation settings.
        loss_cfg: Loss weights; defaults to the paper's 0.7/0.3.
        decode_cfg: Decoding settings used for the selection metric.

    Returns:
        The model with the best checkpoint loaded, and the training history.
    """
    loss_cfg = loss_cfg or LossConfig()
    decode_cfg = decode_cfg or DecodeConfig()
    device = torch.device(
        train_cfg.device if torch.cuda.is_available() or train_cfg.device == "cpu" else "cpu"
    )
    model.to(device)

    optimizer = build_optimizer(model, train_cfg)
    scheduler = build_scheduler(
        optimizer, max(len(train_loader) * train_cfg.max_epochs, 1), train_cfg
    )

    history = TrainingHistory()
    best_state: Optional[Dict[str, torch.Tensor]] = None
    epochs_without_improvement = 0

    for epoch in range(train_cfg.max_epochs):
        components = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, train_cfg, loss_cfg
        )
        metric = selection_metric(model, tokenizer, val_loader, device, decode_cfg)
        history.epochs.append({"epoch": epoch, **components, "val_macro_f1": metric})
        logger.info(
            "Epoch %d/%d | loss %.4f | val macro-F1 %.4f",
            epoch + 1,
            train_cfg.max_epochs,
            components.get("loss", float("nan")),
            metric,
        )

        if metric > history.best_metric:
            history.best_metric = metric
            history.best_epoch = epoch
            best_state = model.trainable_state_dict()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= train_cfg.early_stopping_patience:
                history.stopped_early = True
                logger.info(
                    "Early stopping at epoch %d; best was epoch %d with %.4f",
                    epoch + 1,
                    history.best_epoch + 1,
                    history.best_metric,
                )
                break

    if best_state is not None:
        model.load_trainable_state_dict(best_state)
    return model, history


__all__ = [
    "TrainingHistory",
    "build_optimizer",
    "build_scheduler",
    "selection_metric",
    "train_model",
    "train_one_epoch",
    "training_step",
]
