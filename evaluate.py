"""Evaluation metrics and publication-quality visualisation for MG-HGNN.

Metrics
-------
* ``"grade"``      — regression: RMSE, MAE.
* ``"dropout"``    — binary classification: AUC-ROC, F1, accuracy.
* ``"engagement"`` — multi-class classification: macro-F1, accuracy.

Visualisation
-------------
* :func:`visualize_gate_weights` — heatmap of learned α weights per relation.
* :func:`plot_training_curves`   — per-fold train/val loss curves with mean ± std.

Both figures are saved as vector PDFs with embedded fonts, suitable for direct
inclusion in an ACM / IEEE paper without quality loss.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from torch import Tensor


def evaluate(
    logits: Tensor,
    labels: Tensor,
    task: str,
) -> Dict[str, float]:
    """Compute task-appropriate metrics from model outputs and ground-truth labels.

    Args:
        logits: Raw model output tensor of shape ``[N, 1]`` for regression /
            binary classification, or ``[N, C]`` for multi-class.
        labels: Ground-truth float tensor of shape ``[N]``.
        task: One of ``"grade"``, ``"dropout"``, or ``"engagement"``.

    Returns:
        Dictionary mapping metric name → scalar value.

    Raises:
        ValueError: If *task* is not one of the three supported values.
    """
    logits = logits.detach().cpu()
    labels = labels.detach().cpu()

    if task == "grade":
        return _regression_metrics(logits, labels)
    elif task == "dropout":
        return _binary_metrics(logits, labels)
    elif task == "engagement":
        return _multiclass_metrics(logits, labels)
    else:
        raise ValueError(f"Unknown task '{task}'. Expected 'grade', 'dropout', or 'engagement'.")


# ---------------------------------------------------------------------------
# Task-specific helpers
# ---------------------------------------------------------------------------


def _regression_metrics(logits: Tensor, labels: Tensor) -> Dict[str, float]:
    """Compute RMSE and MAE for grade prediction.

    Args:
        logits: Shape ``[N, 1]`` or ``[N]``.
        labels: Shape ``[N]``.

    Returns:
        ``{"rmse": ..., "mae": ...}``
    """
    preds = logits.squeeze(-1).numpy()
    y = labels.numpy()
    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    mae = float(mean_absolute_error(y, preds))
    return {"rmse": rmse, "mae": mae}


def _binary_metrics(logits: Tensor, labels: Tensor) -> Dict[str, float]:
    """Compute AUC-ROC, F1, and accuracy for binary dropout prediction.

    Args:
        logits: Shape ``[N, 1]`` or ``[N]`` (raw logits or probabilities).
        labels: Shape ``[N]`` with values in ``{0, 1}``.

    Returns:
        ``{"auc_roc": ..., "f1": ..., "accuracy": ...}``
    """
    probs = torch.sigmoid(logits.squeeze(-1)).numpy()
    preds = (probs >= 0.5).astype(int)
    y = labels.long().numpy()

    auc = float(roc_auc_score(y, probs))
    f1 = float(f1_score(y, preds, zero_division=0))
    acc = float(accuracy_score(y, preds))
    return {"auc_roc": auc, "f1": f1, "accuracy": acc}


def _multiclass_metrics(logits: Tensor, labels: Tensor) -> Dict[str, float]:
    """Compute macro-F1 and accuracy for engagement classification.

    Args:
        logits: Shape ``[N, C]``.
        labels: Shape ``[N]`` with integer class indices.

    Returns:
        ``{"macro_f1": ..., "accuracy": ...}``
    """
    preds = logits.argmax(dim=-1).numpy()
    y = labels.long().numpy()
    macro_f1 = float(f1_score(y, preds, average="macro", zero_division=0))
    acc = float(accuracy_score(y, preds))
    return {"macro_f1": macro_f1, "accuracy": acc}


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


def print_metrics(metrics: Dict[str, float], split: str = "test") -> None:
    """Pretty-print a metrics dictionary to stdout.

    Args:
        metrics: Output of :func:`evaluate`.
        split: Label for the dataset split (e.g. ``"val"``, ``"test"``).
    """
    header = f"[{split.upper()}]"
    body = "  ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
    print(f"{header}  {body}")


# ---------------------------------------------------------------------------
# Publication rcParams — applied via plt.rc_context() in each figure function
# ---------------------------------------------------------------------------

_PAPER_RC: Dict[str, object] = {
    # Serif font stack; Times New Roman is used when installed, otherwise Times
    # (PostScript native), otherwise the bundled DejaVu Serif.
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "Times", "DejaVu Serif"],
    # STIX fonts for any in-label math (e.g. α, ±).
    "mathtext.fontset": "stix",
    # Base size; axes labels/titles override below.
    "font.size":        11,
    "axes.labelsize":   11,
    "axes.titlesize":   12,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  9,
    "axes.linewidth":   0.8,
    "xtick.direction":  "out",
    "ytick.direction":  "out",
    # 300 DPI both on screen and when saving raster fallbacks.
    "figure.dpi":       300,
    "savefig.dpi":      300,
    # Embed TrueType (Type-42) fonts in PDF/PS so publishers can edit text.
    # Type-3 bitmap fonts are rejected by most venue submission systems.
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
}


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------


def visualize_gate_weights(
    model,
    edge_types: List[str],
    save_path: str = "results/gate_weight_heatmap.pdf",
) -> str:
    """Produce a publication-quality heatmap of learned modality gate weights.

    Rows correspond to relation types; columns to modalities (structured,
    textual, behavioral).  Each cell shows the mean gate weight α returned by
    ``model.gate.get_gate_weights(edge_type)`` (bias-only / prior mode —
    no data needed).  Values within each row sum to 1.

    Color scale: white at 0.0 → deep teal at 1.0.  Cell values are annotated
    in black (light cells) or white (dark cells) at 3 decimal places.

    Args:
        model: Trained :class:`~models.MGHGNN` with a ``.gate`` attribute.
        edge_types: Ordered relation-type names to display as rows, e.g.
            ``MGHGNN.STUDENT_RELATIONS``.
        save_path: Destination path for the PDF.  Parent directories are
            created automatically.

    Returns:
        Absolute path of the saved file.

    Example::

        from models import MGHGNN
        from evaluate import visualize_gate_weights

        path = visualize_gate_weights(model, MGHGNN.STUDENT_RELATIONS)
        print(f"Saved to {path}")
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    # ---- Collect α weights (prior mode — no data required) ----------------
    model.eval()
    matrix = np.array(
        [model.gate.get_gate_weights(et) for et in edge_types],
        dtype=np.float32,
    )   # (n_relations, 3)

    n_rows, n_cols = matrix.shape
    modality_labels = ["Structured", "Textual", "Behavioral"]
    row_labels = [et.replace("_", " ").title() for et in edge_types]

    # White → deep teal (#1a6b6b) — avoids saturated primaries that reproduce
    # poorly in greyscale print.
    cmap = LinearSegmentedColormap.from_list(
        "white_teal",
        [(0.0, "#ffffff"), (1.0, "#1a6b6b")],
    )

    with plt.rc_context(_PAPER_RC):
        # Height scales with number of rows; add fixed top/bottom margin.
        fig_h = max(2.8, 0.75 * n_rows + 1.4)
        fig, ax = plt.subplots(figsize=(5.5, fig_h))

        im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0.0, vmax=1.0)

        # ---- Tick labels ----------------------------------------------------
        ax.set_xticks(range(n_cols))
        ax.set_yticks(range(n_rows))
        ax.set_xticklabels(modality_labels, fontsize=10)
        ax.set_yticklabels(row_labels, fontsize=10)

        # Move x-axis to the top (standard heatmap convention)
        ax.xaxis.set_label_position("top")
        ax.xaxis.tick_top()

        ax.set_xlabel("Modality", fontsize=11, labelpad=8)
        ax.set_ylabel("Relation type", fontsize=11, labelpad=8)
        ax.set_title(
            r"Learned modality gate weights $\alpha$",
            fontsize=12,
            pad=14,
        )

        # ---- Cell annotations ----------------------------------------------
        for i in range(n_rows):
            for j in range(n_cols):
                val = float(matrix[i, j])
                # Dark teal background → white text; light background → black text.
                fg = "white" if val > 0.52 else "black"
                ax.text(
                    j, i, f"{val:.3f}",
                    ha="center", va="center",
                    fontsize=10, color=fg, fontweight="bold",
                )

        # ---- White grid lines between cells --------------------------------
        ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.8)
        ax.tick_params(which="minor", length=0)   # hide minor tick marks

        # ---- Colorbar ------------------------------------------------------
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"Gate weight $\alpha$", fontsize=10)
        cbar.set_ticks([0.0, 0.25, 0.50, 0.75, 1.0])
        cbar.ax.tick_params(labelsize=9)

        # Thin border around the whole heatmap
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

        fig.tight_layout()

        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, format="pdf", bbox_inches="tight")
        plt.close(fig)

    abs_path = os.path.abspath(save_path)
    print(f"Gate weight heatmap saved → {abs_path}")
    return abs_path


def plot_training_curves(
    history_dict: Dict[str, Dict[str, List[float]]],
    save_path: str = "results/training_curves.pdf",
) -> str:
    """Plot per-fold training and validation loss curves.

    Displays one thin line per fold and a bold mean ± std envelope across all
    folds.  Folds that stopped early naturally have shorter curves; they are
    extended to the maximum observed epoch by repeating their final loss value
    before computing the mean/std envelope.

    Args:
        history_dict: Nested mapping ``{fold_name: {series_name: [values]}}``.
            Expected series names are ``"train_loss"`` (required) and
            ``"val_loss"`` (optional).  Example::

                {
                    "Fold 1": {"train_loss": [0.82, 0.61, ...],
                               "val_loss":   [0.90, 0.72, ...]},
                    "Fold 2": {"train_loss": [...], "val_loss": [...]},
                }

        save_path: Destination path for the PDF.

    Returns:
        Absolute path of the saved file.
    """
    import matplotlib.pyplot as plt

    fold_names = list(history_dict.keys())
    has_val    = any("val_loss" in v for v in history_dict.values())
    n_panels   = 2 if has_val else 1

    # Qualitative palette: distinct colours for up to 10 folds.
    palette = plt.cm.tab10(np.linspace(0, 0.9, max(len(fold_names), 1)))

    with plt.rc_context(_PAPER_RC):
        fig, axes = plt.subplots(
            1, n_panels,
            figsize=(4.5 * n_panels, 4.0),
            sharey=(n_panels == 2),
        )
        axes = [axes] if n_panels == 1 else list(axes)
        train_ax: object = axes[0]
        val_ax: Optional[object] = axes[1] if has_val else None

        train_curves: List[List[float]] = []
        val_curves:   List[List[float]] = []

        for fold, color in zip(fold_names, palette):
            hist = history_dict[fold]
            tr   = hist.get("train_loss", [])
            vl   = hist.get("val_loss",   [])

            if tr:
                ep_tr = np.arange(1, len(tr) + 1)
                train_ax.plot(
                    ep_tr, tr,
                    color=color, linewidth=1.2, alpha=0.65,
                    label=fold,
                )
                train_curves.append(tr)

            if vl and val_ax is not None:
                ep_vl = np.arange(1, len(vl) + 1)
                val_ax.plot(
                    ep_vl, vl,
                    color=color, linewidth=1.2, alpha=0.65,
                    label=fold,
                )
                val_curves.append(vl)

        # ---- Mean ± std envelope -------------------------------------------

        def _envelope(curves: List[List[float]]):
            """Pad to equal length, compute mean and std across folds."""
            max_len = max(len(c) for c in curves)
            padded  = np.array(
                [c + [c[-1]] * (max_len - len(c)) for c in curves],
                dtype=np.float32,
            )
            return padded.mean(0), padded.std(0), np.arange(1, max_len + 1)

        if len(train_curves) > 1:
            mu, sd, ep = _envelope(train_curves)
            train_ax.plot(ep, mu, color="black", linewidth=2.2,
                          label="Mean", zorder=5)
            train_ax.fill_between(ep, mu - sd, mu + sd,
                                  color="black", alpha=0.10, zorder=4)

        if len(val_curves) > 1 and val_ax is not None:
            mu, sd, ep = _envelope(val_curves)
            val_ax.plot(ep, mu, color="black", linewidth=2.2,
                        label="Mean", zorder=5)
            val_ax.fill_between(ep, mu - sd, mu + sd,
                                color="black", alpha=0.10, zorder=4)

        # ---- Per-panel formatting -------------------------------------------
        panel_titles = ["Training loss", "Validation loss"]
        for ax, title in zip(axes, panel_titles):
            ax.set_xlabel("Epoch", fontsize=11)
            ax.set_ylabel("Multi-task loss", fontsize=11)
            ax.set_title(title, fontsize=12)
            ax.set_xlim(left=1)
            # y lower-bound of 0 is clamped at 80 % of the minimum so curves
            # are not squashed against the x-axis.
            ax.set_ylim(bottom=0.0)
            ax.legend(
                fontsize=8, ncol=2,
                framealpha=0.75, edgecolor="0.8",
                loc="upper right",
            )
            # Remove chartjunk spines
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        n_folds = len(fold_names)
        fig.suptitle(
            f"MG-HGNN training dynamics ({n_folds}-fold CV)",
            fontsize=13, y=1.02,
        )
        fig.tight_layout()

        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, format="pdf", bbox_inches="tight")
        plt.close(fig)

    abs_path = os.path.abspath(save_path)
    print(f"Training curves saved → {abs_path}")
    return abs_path
