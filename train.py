"""5-fold stratified cross-validation training loop for MG-HGNN.

Multi-task objective
--------------------
L = λ1 * MSE(grade) + λ2 * BCE(dropout) + λ3 * CE(engagement)

with default λ1=0.40, λ2=0.40, λ3=0.20 (tunable via config / CLI).

Usage::

    # Real data
    python train.py --dataset moocc --data_dir ./data/raw

    # Quick smoke-test on synthetic data (1 fold, 5 epochs, 100 students)
    python train.py --debug
"""

from __future__ import annotations

import argparse
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import HeteroData
from tqdm import tqdm

from config import MGHGNNConfig
from data import MOOCCubeLoader, OULADLoader
from evaluate import evaluate, print_metrics
from models import MGHGNN


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ---------------------------------------------------------------------------
# Multi-label container  (grade, dropout, engagement bundled together)
# ---------------------------------------------------------------------------


class MultiTaskLabels:
    """Holds all three label tensors so callers don't pass four arguments."""

    def __init__(self, grade: Tensor, dropout: Tensor, engagement: Tensor) -> None:
        self.grade      = grade       # float  [N]
        self.dropout    = dropout     # float  [N]  values in {0,1}
        self.engagement = engagement  # long   [N]  values in {0,1,2}

    def to(self, device: torch.device) -> "MultiTaskLabels":
        return MultiTaskLabels(
            self.grade.to(device),
            self.dropout.to(device),
            self.engagement.to(device),
        )

    def __len__(self) -> int:
        return self.grade.size(0)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_dataset(cfg: MGHGNNConfig) -> Tuple[HeteroData, MultiTaskLabels]:
    """Load graph + labels for all three tasks simultaneously.

    Returns:
        data: PyG HeteroData on CPU.
        labels: :class:`MultiTaskLabels` with grade / dropout / engagement.

    Raises:
        ValueError: Unknown ``cfg.dataset``.
    """
    if cfg.dataset == "oulad":
        loader = OULADLoader(cfg.data_dir, task="grade", seq_len=cfg.behavioral_seq_len)
    elif cfg.dataset == "moocc":
        loader = MOOCCubeLoader(cfg.data_dir, task="grade", seq_len=cfg.behavioral_seq_len)
    else:
        raise ValueError(f"Unknown dataset '{cfg.dataset}'.")

    data, grade_labels = loader.load()
    n = grade_labels.size(0)

    # Re-derive dropout and engagement labels from grade using the same
    # thresholds as the individual loaders so the three tasks are coherent.
    # grade ∈ [0, 1] (normalised)
    dropout_labels    = (grade_labels < 0.4).float()                           # bottom 40%
    engagement_labels = torch.bucketize(grade_labels, torch.tensor([0.33, 0.67])).long()

    return data, MultiTaskLabels(grade_labels, dropout_labels, engagement_labels)


# ---------------------------------------------------------------------------
# Synthetic debug dataset
# ---------------------------------------------------------------------------


def _make_debug_data(n_students: int = 100, seq_len: int = 50) -> Tuple[HeteroData, MultiTaskLabels]:
    """Build a minimal synthetic HeteroData graph for pipeline verification."""
    n_courses    = 10
    n_resources  = 20
    n_instructors = 5

    data = HeteroData()

    # Node features
    data["student"].x        = torch.randn(n_students,   8)
    data["course"].x         = torch.randn(n_courses,    8)
    data["resource"].x       = torch.randn(n_resources,  8)
    data["instructor"].x     = torch.randn(n_instructors, 8)

    # Text embeddings (pre-computed BERT-size placeholder)
    data["student"].text_x   = torch.zeros(n_students,   768)
    data["course"].text_x    = torch.zeros(n_courses,    768)
    data["resource"].text_x  = torch.zeros(n_resources,  768)

    # Behavioural sequences
    data["student"].behavior = torch.randn(n_students, seq_len, 2)

    # Edges
    src = torch.randint(0, n_students, (n_students * 3,))
    dst = torch.randint(0, n_courses,  (n_students * 3,))
    data["student", "enrolled_in", "course"].edge_index    = torch.stack([src, dst])

    src2 = torch.randint(0, n_students,  (n_students * 5,))
    dst2 = torch.randint(0, n_resources, (n_students * 5,))
    data["student", "accessed", "resource"].edge_index     = torch.stack([src2, dst2])

    src3 = torch.randint(0, n_students, (n_students * 2,))
    dst3 = torch.randint(0, n_courses,  (n_students * 2,))
    data["student", "submitted_to", "course"].edge_index   = torch.stack([src3, dst3])

    src4 = torch.randint(0, n_students, (n_students * 2,))
    dst4 = torch.randint(0, n_students, (n_students * 2,))
    data["student", "collaborated_with", "student"].edge_index = torch.stack([src4, dst4])

    # Labels
    grade      = torch.rand(n_students)
    dropout    = (grade < 0.4).float()
    engagement = torch.bucketize(grade, torch.tensor([0.33, 0.67])).long()

    return data, MultiTaskLabels(grade, dropout, engagement)


# ---------------------------------------------------------------------------
# Stratified k-fold split
# ---------------------------------------------------------------------------


def _stratified_kfold_masks(
    engagement: Tensor,
    n_folds: int = 5,
    seed: int = 42,
) -> List[Tuple[Tensor, Tensor]]:
    """Return ``n_folds`` (train_mask, val_mask) pairs stratified on engagement.

    Each fold uses ~1/n_folds of the data as validation; the rest is training.
    No held-out test set: test evaluation is done per-fold on the val split.

    Args:
        engagement: Long tensor of class indices used for stratification.
        n_folds: Number of CV folds.
        seed: RNG seed for reproducibility.

    Returns:
        List of ``(train_mask, val_mask)`` boolean tensors of shape ``[N]``.
    """
    n = engagement.size(0)
    labels_np = engagement.cpu().numpy()
    rng = np.random.default_rng(seed)

    # Collect per-class indices, shuffle each
    class_indices: Dict[int, np.ndarray] = {}
    for cls in np.unique(labels_np):
        idx = np.where(labels_np == cls)[0]
        rng.shuffle(idx)
        class_indices[int(cls)] = idx

    # Split each class into n_folds chunks
    fold_val_indices: List[List[int]] = [[] for _ in range(n_folds)]
    for cls_idx in class_indices.values():
        chunks = np.array_split(cls_idx, n_folds)
        for k, chunk in enumerate(chunks):
            fold_val_indices[k].extend(chunk.tolist())

    masks = []
    for k in range(n_folds):
        val_idx = torch.tensor(fold_val_indices[k], dtype=torch.long)
        val_mask   = torch.zeros(n, dtype=torch.bool)
        train_mask = torch.zeros(n, dtype=torch.bool)
        val_mask[val_idx] = True
        train_mask[~val_mask] = True
        masks.append((train_mask, val_mask))

    return masks


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


def _build_model(cfg: MGHGNNConfig, data: HeteroData) -> MGHGNN:
    """Instantiate MGHGNN from config and infer structured dims from data."""
    structured_dims = {nt: data[nt].x.size(-1) for nt in cfg.node_types}
    return MGHGNN(
        node_types=cfg.node_types,
        edge_types=cfg.edge_types,
        structured_dims=structured_dims,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        dropout=cfg.dropout,
        num_classes=cfg.num_classes,
        behavioral_input_dim=cfg.behavioral_input_dim,
        behavioral_seq_len=cfg.behavioral_seq_len,
        text_dim=cfg.text_input_dim,
    )


# ---------------------------------------------------------------------------
# Multi-task loss
# ---------------------------------------------------------------------------


def _multitask_loss(
    preds: Dict[str, Tensor],
    labels: MultiTaskLabels,
    mask: Tensor,
    lambda_grade:      float = 0.4,
    lambda_dropout:    float = 0.4,
    lambda_engagement: float = 0.2,
) -> Tuple[Tensor, Dict[str, float]]:
    """Compute weighted multi-task loss.

    Args:
        preds: Dict from ``model.forward()``, keys ``"grade"``, ``"dropout"``,
            ``"engagement"``.
        labels: All three label tensors.
        mask: Boolean mask selecting the student nodes to include.
        lambda_*: Loss weights (must sum to 1.0 for clean bookkeeping).

    Returns:
        total_loss: Scalar tensor.
        component_losses: Dict of float scalars for logging.
    """
    mse_fn = nn.MSELoss()
    bce_fn = nn.BCEWithLogitsLoss()
    ce_fn  = nn.CrossEntropyLoss()

    l_grade = mse_fn(
        preds["grade"][mask],
        labels.grade[mask].to(preds["grade"].device),
    )
    l_dropout = bce_fn(
        preds["dropout"][mask],
        labels.dropout[mask].to(preds["dropout"].device),
    )
    l_engagement = ce_fn(
        preds["engagement"][mask],
        labels.engagement[mask].to(preds["engagement"].device),
    )

    total = (
        lambda_grade      * l_grade
        + lambda_dropout    * l_dropout
        + lambda_engagement * l_engagement
    )
    components = {
        "loss_grade":      float(l_grade.item()),
        "loss_dropout":    float(l_dropout.item()),
        "loss_engagement": float(l_engagement.item()),
    }
    return total, components


# ---------------------------------------------------------------------------
# Training epoch
# ---------------------------------------------------------------------------


def train_epoch(
    model: MGHGNN,
    data: HeteroData,
    labels: MultiTaskLabels,
    train_mask: Tensor,
    optimizer: torch.optim.Optimizer,
    cfg: MGHGNNConfig,
) -> Tuple[float, Dict[str, float]]:
    """One full-graph gradient step.

    Returns:
        total_loss: scalar float
        components: per-task loss scalars for logging
    """
    model.train()
    optimizer.zero_grad()

    preds = model(data)
    total_loss, components = _multitask_loss(
        preds, labels, train_mask,
        lambda_grade=cfg.lambda_grade,
        lambda_dropout=cfg.lambda_dropout,
        lambda_engagement=cfg.lambda_engagement,
    )

    total_loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return float(total_loss.item()), components


# ---------------------------------------------------------------------------
# Validation step
# ---------------------------------------------------------------------------


@torch.no_grad()
def validate(
    model: MGHGNN,
    data: HeteroData,
    labels: MultiTaskLabels,
    mask: Tensor,
    cfg: MGHGNNConfig,
) -> Tuple[float, Dict[str, float]]:
    """Compute val loss + per-task metrics.

    Returns:
        val_loss: scalar float (total multi-task loss on masked nodes)
        metrics: dict with rmse, auc_roc, macro_f1, accuracy, …
    """
    model.eval()
    preds = model(data)

    val_loss, _ = _multitask_loss(
        preds, labels, mask,
        lambda_grade=cfg.lambda_grade,
        lambda_dropout=cfg.lambda_dropout,
        lambda_engagement=cfg.lambda_engagement,
    )

    from evaluate import evaluate as _eval
    grade_m  = _eval(preds["grade"][mask].unsqueeze(-1),  labels.grade[mask],      "grade")
    drop_m   = _eval(preds["dropout"][mask].unsqueeze(-1), labels.dropout[mask],   "dropout")
    engage_m = _eval(preds["engagement"][mask],            labels.engagement[mask], "engagement")

    metrics = {**grade_m, **drop_m, **engage_m}
    return float(val_loss.item()), metrics


# ---------------------------------------------------------------------------
# Single-fold training
# ---------------------------------------------------------------------------


def train_fold(
    fold: int,
    model: MGHGNN,
    data: HeteroData,
    labels: MultiTaskLabels,
    train_mask: Tensor,
    val_mask: Tensor,
    cfg: MGHGNNConfig,
    device: torch.device,
) -> Dict[str, float]:
    """Train one fold and return the best-checkpoint val metrics.

    Saves best model to ``checkpoints/fold_{fold}_best.pt``.

    Args:
        fold: 1-based fold index (used only for display and checkpoint path).
        model: Fresh (untrained) model already on *device*.
        data: Full graph on *device*.
        labels: All labels on *device*.
        train_mask / val_mask: Boolean masks on CPU (moved to device inside).
        cfg: Training config.
        device: Compute device.

    Returns:
        Best validation metrics dict.
    """
    train_mask = train_mask.to(device)
    val_mask   = val_mask.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, verbose=False
    )

    ckpt_path = os.path.join(cfg.checkpoint_dir, f"fold_{fold}_best.pt")
    best_val_loss   = float("inf")
    best_val_metrics: Dict[str, float] = {}
    best_epoch      = 0
    patience_counter = 0

    pbar = tqdm(
        range(1, cfg.epochs + 1),
        desc=f"Fold {fold}/{cfg.n_folds}",
        unit="epoch",
        leave=True,
    )
    for epoch in pbar:
        train_loss, train_components = train_epoch(
            model, data, labels, train_mask, optimizer, cfg
        )
        val_loss, val_metrics = validate(model, data, labels, val_mask, cfg)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_val_metrics = val_metrics
            best_epoch       = epoch
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1

        pbar.set_postfix(
            tr_loss=f"{train_loss:.4f}",
            vl_loss=f"{val_loss:.4f}",
            rmse=f"{val_metrics.get('rmse', float('nan')):.4f}",
            auc=f"{val_metrics.get('auc_roc', float('nan')):.4f}",
            best=best_epoch,
        )

        if patience_counter >= cfg.patience:
            print(
                f"\n  [Fold {fold}] Early stop at epoch {epoch} "
                f"(no improvement for {cfg.patience} epochs)."
            )
            break

    print(f"  [Fold {fold}] Best val loss {best_val_loss:.4f} at epoch {best_epoch}.")
    print(f"  [Fold {fold}] Val metrics: "
          + "  ".join(f"{k}={v:.4f}" for k, v in best_val_metrics.items()))

    return best_val_metrics


# ---------------------------------------------------------------------------
# Cross-validation orchestrator
# ---------------------------------------------------------------------------


def cross_validate(cfg: MGHGNNConfig) -> None:
    """Run n_folds-fold stratified CV and report mean ± std of all metrics.

    Steps
    -----
    1. Load data (or generate synthetic debug data).
    2. Build fold masks (stratified on engagement label).
    3. For each fold: build a fresh model, train, collect best-val metrics.
    4. Aggregate and print mean ± std across folds.
    """
    _seed_everything(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Data ---------------------------------------------------------------
    if cfg.debug:
        print("DEBUG mode: using 100-student synthetic dataset.")
        data, labels = _make_debug_data(
            n_students=100, seq_len=cfg.behavioral_seq_len
        )
    else:
        print(f"Loading dataset '{cfg.dataset}' ...")
        data, labels = load_dataset(cfg)

    data   = data.to(device)
    labels = labels.to(device)

    n_students = data["student"].x.size(0)
    print(f"Students: {n_students}  |  Folds: {cfg.n_folds}  |  Max epochs: {cfg.epochs}")

    # ---- Fold masks ---------------------------------------------------------
    fold_masks = _stratified_kfold_masks(
        labels.engagement.cpu(), n_folds=cfg.n_folds, seed=cfg.seed
    )

    # ---- Per-fold training --------------------------------------------------
    all_metrics: Dict[str, List[float]] = defaultdict(list)

    for fold, (train_mask, val_mask) in enumerate(fold_masks, start=1):
        print(f"\n{'='*60}")
        print(f"  FOLD {fold}/{cfg.n_folds}   "
              f"train={train_mask.sum().item()}  val={val_mask.sum().item()}")
        print(f"{'='*60}")

        _seed_everything(cfg.seed + fold)   # deterministic but fold-varied
        model = _build_model(cfg, data).to(device)

        fold_metrics = train_fold(
            fold=fold,
            model=model,
            data=data,
            labels=labels,
            train_mask=train_mask,
            val_mask=val_mask,
            cfg=cfg,
            device=device,
        )
        for k, v in fold_metrics.items():
            all_metrics[k].append(v)

    # ---- Aggregate results --------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  {cfg.n_folds}-FOLD CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    for metric, values in all_metrics.items():
        arr = np.array(values)
        print(f"  {metric:<20s}  {arr.mean():.4f} ± {arr.std():.4f}")
    print(f"{'='*60}\n")

    # ---- Gate weight summary (from last fold's model) -----------------------
    print("Learned modality gate weights (α_s, α_t, α_b) — last fold:")
    try:
        gate_weights = model.inspect_gate_weights()
        for rel, ws in gate_weights.items():
            print(f"  {rel:40s}  αs={ws[0]:.3f}  αt={ws[1]:.3f}  αb={ws[2]:.3f}")
    except Exception as exc:
        print(f"  (gate weight inspection skipped: {exc})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> MGHGNNConfig:
    parser = argparse.ArgumentParser(description="Train MG-HGNN (5-fold CV, multi-task)")

    # Data
    parser.add_argument("--dataset",    default="moocc", choices=["oulad", "moocc"])
    parser.add_argument("--data_dir",   default="./data/raw")

    # Model
    parser.add_argument("--hidden_dim", type=int,   default=128)
    parser.add_argument("--num_layers", type=int,   default=2)
    parser.add_argument("--num_heads",  type=int,   default=4)
    parser.add_argument("--dropout",    type=float, default=0.3)

    # Training
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs",     type=int,   default=200)
    parser.add_argument("--patience",   type=int,   default=20)
    parser.add_argument("--n_folds",    type=int,   default=5)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--device",     default="cuda")

    # Loss weights
    parser.add_argument("--lambda_grade",      type=float, default=0.4)
    parser.add_argument("--lambda_dropout",    type=float, default=0.4)
    parser.add_argument("--lambda_engagement", type=float, default=0.2)

    # Debug
    parser.add_argument(
        "--debug", action="store_true",
        help="Run 1 fold for 5 epochs on 100 synthetic students."
    )

    args = parser.parse_args()

    cfg = MGHGNNConfig()
    cfg.dataset           = args.dataset
    cfg.data_dir          = args.data_dir
    cfg.hidden_dim        = args.hidden_dim
    cfg.num_layers        = args.num_layers
    cfg.num_heads         = args.num_heads
    cfg.dropout           = args.dropout
    cfg.lr                = args.lr
    cfg.weight_decay      = args.weight_decay
    cfg.patience          = args.patience
    cfg.seed              = args.seed
    cfg.device            = args.device

    # Multi-task weights
    cfg.lambda_grade      = args.lambda_grade
    cfg.lambda_dropout    = args.lambda_dropout
    cfg.lambda_engagement = args.lambda_engagement

    # CV settings
    cfg.n_folds = args.n_folds

    # Debug overrides
    cfg.debug = args.debug
    if args.debug:
        cfg.epochs  = 5
        cfg.n_folds = 1
        cfg.patience = 999  # never early-stop in debug

    else:
        cfg.epochs = args.epochs

    return cfg


if __name__ == "__main__":
    cross_validate(_parse_args())
