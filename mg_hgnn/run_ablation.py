"""
Ablation study for MG-HGNN.
Runs 5 variants × 5 folds × 50 epochs each (~45 min on CPU).
Results saved to results/ablation_results.json.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import Adam

from config import Config
from data.oulad_loader import OULADLoader
from models.mg_hgnn import (
    MG_HGNN,
    MG_HGNN_UniformGate,
    MG_HGNN_SharedGate,
    MG_HGNN_StructOnly,
    MG_HGNN_BehavOnly,
)
from train_fast import (
    build_encode_loader,
    encode_all_batched,
    evaluate_heads,
    graph_step,
    heads_step,
    _n_students,
)

VARIANTS = {
    "MG-HGNN (full)":   MG_HGNN,
    "Uniform gate":      MG_HGNN_UniformGate,
    "Shared gate":       MG_HGNN_SharedGate,
    "Struct only":       MG_HGNN_StructOnly,
    "Behavioral only":   MG_HGNN_BehavOnly,
}

N_FOLDS  = 5
N_EPOCHS = 50   # enough to compare; full run uses 200


def _build_losses(data, train_t):
    """Return (mse_fn, bce_fn, ce_fn) with class weighting for both
    imbalanced tasks (dropout 71/29, engagement multi-class)."""
    mse_fn = nn.MSELoss()

    n_pos = int((data["student"].y_dropout[train_t] == 1).sum())
    n_neg = int((data["student"].y_dropout[train_t] == 0).sum())
    pw = torch.tensor([n_neg / n_pos])

    def bce_fn(pred, true):
        w = torch.where(true.bool(), pw.expand_as(true), torch.ones_like(true))
        return nn.functional.binary_cross_entropy(pred, true, weight=w)

    eng_lab = data["student"].y_engagement[train_t].numpy()
    eng_w   = compute_class_weight("balanced",
                                   classes=np.unique(eng_lab), y=eng_lab)
    ce_fn   = nn.CrossEntropyLoss(weight=torch.FloatTensor(eng_w))

    return mse_fn, bce_fn, ce_fn


def _graph_params(model):
    """Return all trainable params that are NOT prediction-head params."""
    head_ids = {
        id(p)
        for head in (model.grade_head, model.dropout_head, model.engagement_head)
        for p in head.parameters()
    }
    return [p for p in model.parameters()
            if p.requires_grad and id(p) not in head_ids]


def _head_params(model):
    return (
        list(model.grade_head.parameters())
        + list(model.dropout_head.parameters())
        + list(model.engagement_head.parameters())
    )


def run_variant(name, ModelClass, data, meta, bert_cache, cfg, loader):
    print(f"\n{'='*58}")
    print(f"  Variant: {name}")
    print(f"{'='*58}")
    t_start = time.time()

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    drop_labels = meta["dropout"].numpy()
    n = _n_students(data)

    fold_aucs, fold_f1s, fold_rmses = [], [], []

    for fold_k, (train_idx, val_idx) in enumerate(
            skf.split(np.zeros(len(drop_labels)), drop_labels)):

        train_t = torch.tensor(train_idx, dtype=torch.long)
        val_t   = torch.tensor(val_idx,   dtype=torch.long)

        model   = ModelClass(cfg)
        opt_g   = Adam(_graph_params(model), lr=cfg.lr,
                       weight_decay=cfg.weight_decay)
        opt_h   = Adam(_head_params(model),  lr=cfg.lr * 2)
        mse_fn, bce_fn, ce_fn = _build_losses(data, train_t)

        # Warmup: encode full graph, one graph_step to seed encoder
        h = encode_all_batched(model, loader, bert_cache, n)
        graph_step(model, loader, bert_cache, train_t, data, cfg,
                   opt_g, mse_fn, bce_fn, ce_fn, epoch=0, max_nodes=2000)

        # Head training (N_EPOCHS), track best AUC
        best_auc = 0.0
        best_f1  = 0.0
        best_rmse = float("inf")
        for epoch in range(N_EPOCHS):
            heads_step(model, h, train_t, data, cfg,
                       opt_h, mse_fn, bce_fn, ce_fn, epoch=epoch)
            vm = evaluate_heads(model, h, val_t, data, cfg,
                                mse_fn, bce_fn, ce_fn)
            if vm["val_auc"] > best_auc:
                best_auc  = vm["val_auc"]
                best_f1   = vm["val_f1"]
                best_rmse = vm["val_rmse"]

        fold_aucs.append(best_auc)
        fold_f1s.append(best_f1)
        fold_rmses.append(best_rmse)
        print(f"  Fold {fold_k+1}: AUC={best_auc:.4f}  "
              f"RMSE={best_rmse:.4f}  F1={best_f1:.4f}")

    elapsed = time.time() - t_start
    result = {
        "auc_mean":  float(np.mean(fold_aucs)),
        "auc_std":   float(np.std(fold_aucs)),
        "f1_mean":   float(np.mean(fold_f1s)),
        "rmse_mean": float(np.mean(fold_rmses)),
    }
    print(f"\n  {name}: AUC = "
          f"{result['auc_mean']:.4f} ± {result['auc_std']:.4f}  "
          f"({elapsed/60:.1f} min)")
    return result


def main():
    cfg = Config()

    print("Loading OULAD...")
    data, meta = OULADLoader(cfg).load()

    # Attach labels
    data["student"].y_grade      = meta["grade"].float()
    data["student"].y_dropout    = meta["dropout"].float()
    data["student"].y_engagement = meta["engagement"].long()

    # Cap accessed edges (same as train_fast.py)
    MAX_ACCESSED = 100_000
    _etype = ("student", "accessed", "resource")
    if hasattr(data[_etype], "edge_index"):
        ei = data[_etype].edge_index
        if ei.shape[1] > MAX_ACCESSED:
            perm = torch.randperm(ei.shape[1])[:MAX_ACCESSED]
            data[_etype].edge_index = ei[:, perm]
            print(f"  [graph cap] accessed edges → {MAX_ACCESSED:,}")

    cache_path = Path("data/bert_cache_oulad.pt")
    print(f"Loading BERT cache from {cache_path}...")
    bert_cache = torch.load(str(cache_path), weights_only=True)

    # Build loader once — shared across all variants and folds
    loader = build_encode_loader(data, cfg)

    Path("results").mkdir(exist_ok=True)
    all_results = {}

    for name, ModelClass in VARIANTS.items():
        all_results[name] = run_variant(
            name, ModelClass, data, meta, bert_cache, cfg, loader
        )
        # Save incrementally so a crash doesn't lose earlier variants
        with open("results/ablation_results.json", "w") as f:
            json.dump(all_results, f, indent=2)

    # Final table
    W = 58
    print(f"\n{'='*W}")
    print(f"  ABLATION TABLE  ({N_FOLDS}-fold, {N_EPOCHS} epochs each)")
    print(f"{'='*W}")
    print(f"  {'Variant':<28} {'AUC':>8}  {'±':>6}  {'RMSE':>8}  {'F1':>7}")
    print(f"  {'-'*W}")
    for name, r in all_results.items():
        marker = "  ←" if name == "MG-HGNN (full)" else ""
        print(f"  {name:<28} {r['auc_mean']:>8.4f}"
              f"  {r['auc_std']:>6.4f}  "
              f"{r['rmse_mean']:>8.4f}  "
              f"{r['f1_mean']:>7.4f}{marker}")
    print(f"{'='*W}")
    print(f"\nSaved → results/ablation_results.json")


if __name__ == "__main__":
    main()
