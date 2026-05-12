import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import HeteroData
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, mean_squared_error
from tqdm import tqdm

from config import Config
from evaluate import compute_metrics, print_results_table
from models.mg_hgnn import MG_HGNN


# ----------------------------------------------------------------
# Training configuration
# ----------------------------------------------------------------

@dataclass
class TrainingConfig:
    n_folds:   int  = 5
    debug:     bool = False   # 1 fold, 5 epochs
    log_every: int  = 10      # print a summary line every N epochs (non-debug)


# ----------------------------------------------------------------
# Synthetic data builder
# ----------------------------------------------------------------

def make_synthetic_data(
    config: Config,
    n_students:  int = 200,
    n_courses:   int = 20,
    n_resources: int = 30,
) -> Tuple[HeteroData, Dict]:
    """Return a random HeteroData graph + meta dict for smoke-testing."""
    n_instructors = 10
    data = HeteroData()

    data["student"].x_struct       = torch.randn(n_students, config.structured_input_dim)
    data["student"].input_ids      = torch.randint(1, 1000, (n_students, 32))
    data["student"].attention_mask = torch.ones(n_students, 32, dtype=torch.long)
    data["student"].x_behav        = torch.randn(
        n_students, config.behavioral_seq_len, config.behavioral_input_dim
    )

    data["course"].input_ids      = torch.randint(1, 1000, (n_courses, 32))
    data["course"].attention_mask = torch.ones(n_courses, 32, dtype=torch.long)

    data["resource"].input_ids      = torch.randint(1, 1000, (n_resources, 32))
    data["resource"].attention_mask = torch.ones(n_resources, 32, dtype=torch.long)

    data["instructor"].x_struct = torch.randn(n_instructors, config.structured_input_dim)

    def _rand_ei(n_src: int, n_dst: int, n: int) -> torch.Tensor:
        return torch.stack([torch.randint(0, n_src, (n,)),
                            torch.randint(0, n_dst, (n,))])

    data["student", "enrolled_in",       "course"].edge_index  = _rand_ei(n_students, n_courses,   300)
    data["student", "collaborated_with", "student"].edge_index = _rand_ei(n_students, n_students,  150)
    data["student", "accessed",          "resource"].edge_index = _rand_ei(n_students, n_resources, 200)
    data["student", "submitted_to",      "course"].edge_index  = _rand_ei(n_students, n_courses,   250)

    meta = {
        "grade":       torch.rand(n_students) * 4.0,
        "dropout":     torch.randint(0, 2, (n_students,)).long(),
        "engagement":  torch.randint(0, 3, (n_students,)).long(),
        "student_ids": np.arange(n_students),
    }
    return data, meta


# ----------------------------------------------------------------
# Loss
# ----------------------------------------------------------------

def _multitask_loss(
    preds:  Dict[str, torch.Tensor],
    data:   HeteroData,
    idx:    torch.Tensor,
    cfg:    Config,
    mse_fn: nn.MSELoss,
    bce_fn: nn.BCELoss,
    ce_fn:  nn.CrossEntropyLoss,
) -> torch.Tensor:
    """L = λ_g·MSE(grade) + λ_d·BCE(dropout) + λ_e·CE(engagement)."""
    grade_pred   = preds["grade"][idx].squeeze()
    dropout_pred = preds["dropout"][idx].squeeze()
    eng_pred     = preds["engagement"][idx]

    grade_true   = data["student"].y_grade[idx]
    dropout_true = data["student"].y_dropout[idx]
    eng_true     = data["student"].y_engagement[idx]

    return (
        cfg.lambda_grade        * mse_fn(grade_pred,   grade_true)
        + cfg.lambda_dropout    * bce_fn(dropout_pred, dropout_true)
        + cfg.lambda_engagement * ce_fn(eng_pred,      eng_true)
    )


# ----------------------------------------------------------------
# Single-fold training
# ----------------------------------------------------------------

def _class_dist(labels: torch.Tensor) -> str:
    """Return a compact class-count string, e.g. '0:120 1:40'."""
    vals, cnts = labels.long().unique(return_counts=True)
    return "  ".join(f"{v.item()}:{c.item()}" for v, c in zip(vals, cnts))


def train_one_fold(
    model:       MG_HGNN,
    train_idx:   np.ndarray,
    val_idx:     np.ndarray,
    data:        HeteroData,
    cfg:         Config,
    train_cfg:   TrainingConfig,
    fold_k:      int = 0,
    bert_cache:  Optional[Dict] = None,
) -> Tuple[Dict[str, List[float]], Dict[str, float]]:
    """Train for one fold; return (per-epoch history, best-checkpoint metrics).

    history keys : train_loss, val_loss, val_rmse, val_auc (one value per epoch)
    best_metrics : rmse, auc, f1, best_auc_epoch
    """
    Path(cfg.checkpoint_dir).mkdir(exist_ok=True)
    ckpt_path = Path(cfg.checkpoint_dir) / f"fold_{fold_k}_best.pt"

    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    mse_fn = nn.MSELoss()
    bce_fn = nn.BCELoss()
    ce_fn  = nn.CrossEntropyLoss()

    n_epochs = 5 if train_cfg.debug else cfg.epochs
    train_t  = torch.tensor(train_idx, dtype=torch.long)
    val_t    = torch.tensor(val_idx,   dtype=torch.long)

    # ---- Class distribution log ----
    drop_labels = data["student"].y_dropout
    tr_dist = _class_dist(drop_labels[train_t])
    va_dist = _class_dist(drop_labels[val_t])
    print(f"  Dropout dist  train [{tr_dist}]  val [{va_dist}]")

    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [], "val_rmse": [], "val_auc": []
    }
    best_val_loss = float("inf")
    patience_ctr  = 0

    bar = tqdm(range(n_epochs), desc=f"Fold {fold_k}", disable=train_cfg.debug)
    epoch_times: List[float] = []

    for epoch in bar:
        t_epoch = time.time()

        # ---- Train ----
        model.train()
        optimizer.zero_grad()
        preds = model(data, bert_cache)
        train_loss = _multitask_loss(preds, data, train_t, cfg, mse_fn, bce_fn, ce_fn)
        train_loss.backward()
        optimizer.step()

        # ---- Validate ----
        model.eval()
        with torch.no_grad():
            preds    = model(data, bert_cache)
            val_loss = _multitask_loss(preds, data, val_t, cfg, mse_fn, bce_fn, ce_fn)

            g_pred = preds["grade"][val_t].squeeze().numpy()
            g_true = data["student"].y_grade[val_t].numpy()
            rmse   = float(np.sqrt(mean_squared_error(g_true, g_pred)))

            d_pred = preds["dropout"][val_t].squeeze().numpy()
            d_true = data["student"].y_dropout[val_t].numpy().astype(int)
            try:
                auc = float(roc_auc_score(d_true, d_pred))
            except ValueError:
                auc = 0.5

        tl = float(train_loss)
        vl = float(val_loss)
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["val_rmse"].append(rmse)
        history["val_auc"].append(auc)

        epoch_sec = time.time() - t_epoch
        epoch_times.append(epoch_sec)
        if epoch == 0:
            cache_tag = "cache" if bert_cache else "no-cache"
            print(f"  [Epoch 1 timing] {epoch_sec:.2f}s  ({cache_tag})")

        if not train_cfg.debug:
            bar.set_postfix(tr=f"{tl:.4f}", val=f"{vl:.4f}",
                            rmse=f"{rmse:.4f}", auc=f"{auc:.4f}")
            if (epoch + 1) % train_cfg.log_every == 0:
                tqdm.write(
                    f"  Fold {fold_k} | Epoch {epoch+1:4d}/{n_epochs}  "
                    f"train={tl:.4f}  val={vl:.4f}  rmse={rmse:.4f}  auc={auc:.4f}"
                )
        else:
            print(f"  Epoch {epoch+1:2d}/{n_epochs}  "
                  f"train={tl:.4f}  val={vl:.4f}  "
                  f"rmse={rmse:.4f}  auc={auc:.4f}")

        # ---- Early stopping ----
        if vl < best_val_loss:
            best_val_loss = vl
            patience_ctr  = 0
            torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
        else:
            patience_ctr += 1

        if patience_ctr >= cfg.patience:
            msg = f"  Early stop at epoch {epoch + 1} (patience={cfg.patience})"
            (print if train_cfg.debug else tqdm.write)(msg)
            break

    # ---- Best-AUC epoch ----
    best_auc_epoch = int(np.argmax(history["val_auc"])) + 1
    print(f"  Best val-AUC epoch: {best_auc_epoch}")

    # ---- Best-checkpoint metrics ----
    model.load_state_dict(torch.load(ckpt_path, weights_only=True)["model_state_dict"])
    model.eval()
    with torch.no_grad():
        preds = model(data, bert_cache)

        g_pred    = preds["grade"][val_t].squeeze().numpy()
        g_true    = data["student"].y_grade[val_t].numpy()
        best_rmse = float(np.sqrt(mean_squared_error(g_true, g_pred)))

        d_pred = preds["dropout"][val_t].squeeze().numpy()
        d_true = data["student"].y_dropout[val_t].numpy().astype(int)
        try:
            best_auc = float(roc_auc_score(d_true, d_pred))
        except ValueError:
            best_auc = 0.5

        e_pred  = preds["engagement"][val_t].argmax(dim=-1).numpy()
        e_true  = data["student"].y_engagement[val_t].numpy()
        best_f1 = float(f1_score(e_true, e_pred, average="macro", zero_division=0))

    avg_epoch_sec = float(np.mean(epoch_times))
    return history, {
        "rmse": best_rmse, "auc": best_auc, "f1": best_f1,
        "best_auc_epoch": best_auc_epoch,
        "avg_epoch_sec":  avg_epoch_sec,
    }


# ----------------------------------------------------------------
# Results persistence
# ----------------------------------------------------------------

def _save_results(
    all_histories: List[Dict],
    all_metrics:   List[Dict],
    out_path:      Path,
    dataset:       str,
    n_folds_total: int,
) -> None:
    """Serialize completed-so-far results to JSON (safe to call after each fold)."""
    folds_done = len(all_metrics)
    rmses = [m["rmse"] for m in all_metrics]
    aucs  = [m["auc"]  for m in all_metrics]
    f1s   = [m["f1"]   for m in all_metrics]
    payload = {
        "dataset":         dataset,
        "folds_completed": folds_done,
        "folds_total":     n_folds_total,
        "folds": [
            {k: ([float(x) for x in v] if isinstance(v, list) else float(v))
             for k, v in h.items()}
            for h in all_histories
        ],
        "best_metrics": [
            {k: float(v) for k, v in m.items()} for m in all_metrics
        ],
        "summary": {
            "rmse_mean": float(np.mean(rmses)), "rmse_std": float(np.std(rmses)),
            "auc_mean":  float(np.mean(aucs)),  "auc_std":  float(np.std(aucs)),
            "f1_mean":   float(np.mean(f1s)),   "f1_std":   float(np.std(f1s)),
        },
    }
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train MG-HGNN")
    parser.add_argument("--debug",   action="store_true",
                        help="1 fold, 5 epochs (synthetic: 100 students)")
    parser.add_argument("--dataset", choices=["synthetic", "oulad", "moocc"],
                        default="synthetic",
                        help="Data source  [synthetic | oulad | moocc]  (default: synthetic)")
    parser.add_argument("--use-cache", action="store_true",
                        help="Load pre-computed BERT embeddings instead of running "
                             "BERT each epoch (requires data/bert_cache_<dataset>.pt — "
                             "generate with: python data/cache_embeddings.py --dataset <ds>)")
    args = parser.parse_args()

    cfg       = Config()
    train_cfg = TrainingConfig(debug=args.debug)

    print(f"Dataset: {args.dataset.upper()}")
    print(f"Mode:    {'DEBUG (1 fold, 5 epochs)' if args.debug else 'FULL (5-fold CV)'}")

    # ---- Data loading ----
    if args.dataset == "oulad":
        from data.oulad_loader import OULADLoader
        loader = OULADLoader(cfg)
        data, meta = loader.load()
        loader.summary(data, meta)
    elif args.dataset == "moocc":
        from data.moocc_loader import MOOCCubeLoader
        loader = MOOCCubeLoader(cfg)
        data, meta = loader.load()
        loader.summary(data, meta)
    else:
        n_students = 100 if args.debug else 200
        print(f"Building synthetic graph ({n_students} students)...")
        data, meta = make_synthetic_data(cfg, n_students=n_students)

    # ---- BERT cache ----
    bert_cache: Optional[Dict] = None
    if args.use_cache:
        from data.cache_embeddings import cache_bert_embeddings, load_bert_cache
        cache_path = Path(f"data/bert_cache_{args.dataset}.pt")
        if cache_path.exists():
            print(f"Loading BERT cache from {cache_path}…")
            bert_cache = load_bert_cache(str(cache_path))
            print(f"  Cached node types: {list(bert_cache.keys())}")
        else:
            print(f"Cache not found at {cache_path} — building now…")
            bert_cache = cache_bert_embeddings(data, cfg, save_path=str(cache_path))

    # Attach labels onto the data object so _multitask_loss can read them
    # uniformly regardless of data source.
    data["student"].y_grade      = meta["grade"].float()
    data["student"].y_dropout    = meta["dropout"].float()   # BCE needs float
    data["student"].y_engagement = meta["engagement"].long()

    n_students     = len(meta["student_ids"])
    dropout_labels = meta["dropout"].numpy()

    n_folds = 1 if args.debug else train_cfg.n_folds

    # n_splits ≥ 2 required by StratifiedKFold; slice to n_folds afterward
    skf   = StratifiedKFold(n_splits=max(n_folds, 2), shuffle=True, random_state=42)
    folds = list(skf.split(np.arange(n_students), dropout_labels))[:n_folds]

    all_histories: List[Dict] = []
    all_metrics:   List[Dict] = []

    for fold_k, (train_idx, val_idx) in enumerate(folds):
        print(f"\n{'='*52}")
        print(f"  Fold {fold_k + 1}/{n_folds}  "
              f"(train={len(train_idx):,}, val={len(val_idx):,})")
        print(f"{'='*52}")

        model = MG_HGNN(cfg)
        history, best = train_one_fold(
            model, train_idx, val_idx, data, cfg, train_cfg,
            fold_k=fold_k, bert_cache=bert_cache,
        )
        all_histories.append(history)
        all_metrics.append(best)
        print(f"\n  Fold {fold_k + 1} best →  "
              f"RMSE={best['rmse']:.4f}  "
              f"AUC={best['auc']:.4f}  "
              f"F1={best['f1']:.4f}  "
              f"best-AUC-ep={best['best_auc_epoch']}")

        out_path = Path(cfg.results_dir) / f"training_history_{args.dataset}.json"
        _save_results(all_histories, all_metrics, out_path, args.dataset, n_folds)
        print(f"  [saved] {fold_k + 1}/{n_folds} folds → {out_path}")

    # ---- Cross-validation summary ----
    rmses = [m["rmse"] for m in all_metrics]
    aucs  = [m["auc"]  for m in all_metrics]
    f1s   = [m["f1"]   for m in all_metrics]
    W = 46
    print(f"\n{'='*W}")
    print(f"  Cross-Validation Summary  ({n_folds} fold{'s' if n_folds > 1 else ''})")
    print(f"{'='*W}")
    print(f"  RMSE:          {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")
    print(f"  AUC-ROC:       {np.mean(aucs):.4f}  ± {np.std(aucs):.4f}")
    print(f"  Engagement-F1: {np.mean(f1s):.4f}  ± {np.std(f1s):.4f}")
    print(f"{'='*W}")

    # ---- Persist final results (already written per-fold; this is the canonical copy) ----
    out_path = Path(cfg.results_dir) / f"training_history_{args.dataset}.json"
    _save_results(all_histories, all_metrics, out_path, args.dataset, n_folds)
    print(f"\n  Histories saved → {out_path}")

    # ---- Final metrics table (last fold's best checkpoint) ----
    last_fold_k  = n_folds - 1
    last_val_idx = folds[last_fold_k][1]
    last_val_t   = torch.tensor(last_val_idx, dtype=torch.long)
    last_ckpt    = Path(cfg.checkpoint_dir) / f"fold_{last_fold_k}_best.pt"

    final_model = MG_HGNN(cfg)
    final_model.load_state_dict(
        torch.load(last_ckpt, weights_only=True)["model_state_dict"]
    )
    final_model.eval()
    with torch.no_grad():
        final_preds = final_model(data, bert_cache)

    preds_np = {
        "grade":      final_preds["grade"][last_val_t].squeeze().numpy(),
        "dropout":    final_preds["dropout"][last_val_t].squeeze().numpy(),
        "engagement": final_preds["engagement"][last_val_t].argmax(dim=-1).numpy(),
    }
    labels_np = {
        "grade":      data["student"].y_grade[last_val_t].numpy(),
        "dropout":    data["student"].y_dropout[last_val_t].numpy().astype(int),
        "engagement": data["student"].y_engagement[last_val_t].numpy(),
    }
    metrics = compute_metrics(preds_np, labels_np)
    tag = "debug" if args.debug else args.dataset
    print_results_table(metrics, model_name=f"MG-HGNN ({tag})")


if __name__ == "__main__":
    main()
