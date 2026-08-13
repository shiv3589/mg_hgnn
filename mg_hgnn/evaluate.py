"""
Evaluation metrics and visualisation utilities for MG-HGNN.

Public API
----------
compute_metrics(preds, labels)          -> flat metrics dict
print_results_table(results, model_name)
visualize_gate_weights(model, save_path)
plot_training_curves(history_dict, save_path)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless-safe; call before pyplot import
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)

if TYPE_CHECKING:
    # Imported only for type hints; avoids loading BERT at import time.
    from models.mg_hgnn import MG_HGNN


# ----------------------------------------------------------------
# 1. Metrics
# ----------------------------------------------------------------

def compute_metrics(preds: Dict[str, np.ndarray],
                    labels: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Compute all task metrics from prediction / label arrays.

    preds  keys: 'grade'      → float array
                 'dropout'    → float probability array [0, 1]
                 'engagement' → integer class-label array
    labels keys: same tasks, ground-truth values

    Returns a flat dict with keys like 'grade_rmse', 'dropout_auc', …
    """
    out: Dict[str, float] = {}

    # ---- Grade (regression) ----
    g_pred = np.asarray(preds["grade"],  dtype=float).ravel()
    g_true = np.asarray(labels["grade"], dtype=float).ravel()
    out["grade_rmse"] = float(np.sqrt(mean_squared_error(g_true, g_pred)))
    out["grade_mae"]  = float(mean_absolute_error(g_true, g_pred))

    # ---- Dropout (binary classification; preds are probabilities) ----
    d_prob  = np.asarray(preds["dropout"],  dtype=float).ravel()
    d_true  = np.asarray(labels["dropout"], dtype=int).ravel()
    d_pred  = (d_prob >= 0.5).astype(int)
    try:
        out["dropout_auc"] = float(roc_auc_score(d_true, d_prob))
    except ValueError:
        out["dropout_auc"] = float("nan")
    out["dropout_f1"]        = float(f1_score(d_true, d_pred, average="macro",
                                               zero_division=0))
    out["dropout_precision"] = float(precision_score(d_true, d_pred, average="macro",
                                                      zero_division=0))
    out["dropout_recall"]    = float(recall_score(d_true, d_pred, average="macro",
                                                   zero_division=0))

    # ---- Engagement (multi-class; preds are already class labels) ----
    e_pred = np.asarray(preds["engagement"],  dtype=int).ravel()
    e_true = np.asarray(labels["engagement"], dtype=int).ravel()
    out["engagement_accuracy"]    = float(accuracy_score(e_true, e_pred))
    out["engagement_f1_weighted"] = float(f1_score(e_true, e_pred,
                                                    average="weighted",
                                                    zero_division=0))

    return out


# ----------------------------------------------------------------
# 2. Results table
# ----------------------------------------------------------------

_METRIC_LABELS = {
    "grade_rmse":             ("Grade",      "RMSE"),
    "grade_mae":              ("",           "MAE"),
    "dropout_auc":            ("Dropout",    "AUC-ROC"),
    "dropout_f1":             ("",           "F1 (macro)"),
    "dropout_precision":      ("",           "Precision"),
    "dropout_recall":         ("",           "Recall"),
    "engagement_accuracy":    ("Engagement", "Accuracy"),
    "engagement_f1_weighted": ("",           "Weighted-F1"),
}


def print_results_table(results: Dict[str, float], model_name: str) -> None:
    """Pretty-print a flat metrics dict with task grouping and aligned columns."""
    W = 52
    sep = "─" * W
    print(f"\nModel: {model_name}")
    print(sep)
    print(f"  {'Task':<14}  {'Metric':<18}  {'Value':>8}")
    print(sep)
    for key, (task, metric) in _METRIC_LABELS.items():
        val = results.get(key, float("nan"))
        val_str = f"{val:.4f}" if not np.isnan(val) else "  n/a"
        print(f"  {task:<14}  {metric:<18}  {val_str:>8}")
    print(sep + "\n")


# ----------------------------------------------------------------
# 3. Gate-weight heatmap
# ----------------------------------------------------------------

def visualize_gate_weights(
    model:     "MG_HGNN",
    save_path: str = "results/gate_heatmap.pdf",
    h_s:       "torch.Tensor | None" = None,
    h_t:       "torch.Tensor | None" = None,
    h_b:       "torch.Tensor | None" = None,
) -> None:
    """
    Plot a (n_edge_types × 3) heatmap of the learned modality gate weights.

    If real encoded embeddings (h_s, h_t, h_b — student structured/text/
    behavioral embeddings, e.g. from model.struct_enc/bert_cache/model.behav_enc)
    are supplied, uses model.gate.get_gate_weights_from_data(r, h_s, h_t, h_b)
    for each edge-type relation r — the actual per-sample alpha the model
    produces, averaged over samples. This is the correct source for Figure 2 /
    Table 6.

    If h_s/h_t/h_b are omitted, falls back to the synthetic-probe
    model.gate.get_gate_weights(r). NOTE: that fallback was found (2026-08-11)
    to wash out learned per-relation differentiation to ~[0.33, 0.33, 0.33]
    regardless of training — see models/gate.py docstrings. Only use the
    fallback for a quick sanity check, never to report learned behavior.

    Saves both PDF (vector) and PNG.
    """
    edge_types = model.gate.edge_types
    modalities = ["Structured", "Text", "Behavioral"]

    if h_s is not None and h_t is not None and h_b is not None:
        weights = np.stack(
            [model.gate.get_gate_weights_from_data(r, h_s, h_t, h_b)
             for r in edge_types], axis=0
        )   # (n_rel, 3)
    else:
        weights = np.stack(
            [model.gate.get_gate_weights(r) for r in edge_types], axis=0
        )   # (n_rel, 3)

    # Readable row labels: replace underscores with spaces
    row_labels = [r.replace("_", " ") for r in edge_types]

    fig, ax = plt.subplots(figsize=(6, 3.5))

    im = ax.imshow(weights, cmap="YlGnBu", vmin=0.0, vmax=1.0,
                   aspect="auto")

    # Axis ticks
    ax.set_xticks(range(len(modalities)))
    ax.set_xticklabels(modalities, fontsize=10)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)

    # Cell annotations
    for i in range(len(edge_types)):
        for j in range(len(modalities)):
            val = weights[i, j]
            text_color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color=text_color, fontweight="bold")

    ax.set_xlabel("Modality", fontsize=11)
    ax.set_ylabel("Edge relation type", fontsize=11)
    ax.set_title("Learned modality gate weights α per edge type", fontsize=11)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Gate weight α", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()

    # Save PDF (vector) and PNG
    pdf_path = Path(save_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")

    png_path = pdf_path.with_suffix(".png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")

    plt.close(fig)
    print(f"Gate heatmap saved → {pdf_path}  and  {png_path}")


# ----------------------------------------------------------------
# 4. Training-curve plots
# ----------------------------------------------------------------

_STYLE = {
    "font.family":  "DejaVu Serif",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linestyle":     "--",
    "grid.linewidth":     0.5,
}

_TRAIN_COLOR = "#4878CF"   # muted blue
_VAL_COLOR   = "#D65F5F"   # muted red
_MEAN_COLOR  = "#222222"   # near-black for bold mean overlay


def plot_training_curves(
    history_dict: Dict[str, Dict[str, List[float]]],
    save_path: str = "results/training_curves.pdf",
) -> None:
    """
    Plot per-fold train/val loss curves with a bold mean overlay.

    Parameters
    ----------
    history_dict : dict
        Keys  'fold_0', 'fold_1', …  →  sub-dicts with
        'train_loss' and 'val_loss' lists (one value per epoch).
    save_path : str
        PDF output path. A sibling PNG is also written.
    """
    with plt.rc_context(_STYLE):
        fold_keys = sorted(history_dict.keys())
        n_folds   = len(fold_keys)

        # Layout: up to 3 columns
        ncols = min(n_folds, 3)
        nrows = int(np.ceil(n_folds / ncols))
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(4.5 * ncols, 3.2 * nrows),
            squeeze=False,
        )

        # Collect series for mean computation (pad shorter runs with last value)
        max_len = max(
            len(history_dict[k]["val_loss"]) for k in fold_keys
        )

        def _pad(lst: List[float]) -> np.ndarray:
            arr = np.array(lst, dtype=float)
            if len(arr) < max_len:
                arr = np.concatenate([arr, np.full(max_len - len(arr), arr[-1])])
            return arr

        all_train = np.stack([_pad(history_dict[k]["train_loss"]) for k in fold_keys])
        all_val   = np.stack([_pad(history_dict[k]["val_loss"])   for k in fold_keys])
        mean_train = all_train.mean(axis=0)
        mean_val   = all_val.mean(axis=0)
        epochs_axis = np.arange(1, max_len + 1)

        for idx, fold_key in enumerate(fold_keys):
            row, col = divmod(idx, ncols)
            ax = axes[row][col]

            train_loss = history_dict[fold_key]["train_loss"]
            val_loss   = history_dict[fold_key]["val_loss"]
            ep = np.arange(1, len(train_loss) + 1)

            ax.plot(ep, train_loss, color=_TRAIN_COLOR, lw=1.4,
                    alpha=0.8, label="Train loss")
            ax.plot(ep, val_loss,   color=_VAL_COLOR,   lw=1.4,
                    alpha=0.8, label="Val loss")

            # Mean overlay (same colour, bold, dashed)
            ax.plot(epochs_axis, mean_train, color=_TRAIN_COLOR,
                    lw=2.5, ls="--", alpha=0.5, label="Mean train")
            ax.plot(epochs_axis, mean_val,   color=_MEAN_COLOR,
                    lw=2.5, ls="-",  alpha=0.9, label="Mean val")

            # Best-checkpoint epoch (argmin val_loss for this fold)
            best_ep = int(np.argmin(val_loss)) + 1
            ax.axvline(best_ep, color="grey", lw=1.2, ls=":",
                       label=f"Best ckpt (ep {best_ep})")

            ax.set_title(fold_key.replace("_", " ").title(), fontsize=10)
            ax.set_xlabel("Epoch", fontsize=9)
            ax.set_ylabel("Loss",  fontsize=9)
            ax.tick_params(labelsize=8)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=5))
            if idx == 0:
                ax.legend(fontsize=7, loc="upper right", framealpha=0.7)

        # Hide unused subplot panels
        for idx in range(n_folds, nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row][col].set_visible(False)

        fig.suptitle("Training and validation loss curves", fontsize=12, y=1.01)
        fig.tight_layout()

        pdf_path = Path(save_path)
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf_path, dpi=300, bbox_inches="tight")

        png_path = pdf_path.with_suffix(".png")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")

        plt.close(fig)
        print(f"Training curves saved → {pdf_path}  and  {png_path}")
