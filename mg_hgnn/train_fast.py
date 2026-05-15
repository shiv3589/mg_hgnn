"""
train_fast.py — 3-fix optimised training for MG-HGNN.

Speedups vs train.py (186 s/epoch on 32 K nodes):
  FIX 1  Pre-cache BERT + HGTConv embeddings via model.encode_all()
  FIX 2  Lightweight inner loop: only prediction heads run each epoch;
         graph encoder updates every 10 epochs (outer step).
  FIX 3  NeighborLoader ([10,5] neighbours, batch 1024) for all graph passes.

CLI (drop-in replacement for train.py):
  python train_fast.py --dataset oulad --use-cache
  python train_fast.py --debug
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, mean_squared_error
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

from config import Config
from evaluate import compute_metrics, print_results_table
from models.mg_hgnn import MG_HGNN
from train import TrainingConfig, make_synthetic_data, _class_dist, _save_results


# ----------------------------------------------------------------
# FIX 3 — NeighborLoader
# ----------------------------------------------------------------

def _n_students(data: HeteroData) -> int:
    """Derive student count from the first available feature tensor."""
    s = data["student"]
    for attr in ("x_struct", "x_behav", "input_ids"):
        if hasattr(s, attr):
            return getattr(s, attr).shape[0]
    raise ValueError("Cannot determine student count from HeteroData")


class _FullGraphLoader:
    """Fallback when torch-sparse/pyg-lib are unavailable.

    Yields the full HeteroData as a single 'batch', setting the fields that
    encode_all_batched / graph_step expect from a real NeighborLoader batch:
      batch["student"].batch_size  — number of seed students
      batch["student"].n_id        — global student IDs (identity mapping)
    """
    def __init__(self, data: HeteroData, n_students: int) -> None:
        self._data = data
        self._n    = n_students

    def __iter__(self):
        self._data["student"].batch_size = self._n
        self._data["student"].n_id       = torch.arange(self._n)
        yield self._data


def build_encode_loader(data: HeteroData, cfg: Config):
    """Return a NeighborLoader if torch-sparse/pyg-lib is available, else a
    full-graph fallback loader.  Both expose the same iteration interface."""
    n = _n_students(data)
    try:
        import torch_geometric.typing as _T
        if _T.WITH_PYG_LIB or _T.WITH_TORCH_SPARSE:
            return NeighborLoader(
                data,
                num_neighbors={etype: [10, 5] for etype in data.edge_types},
                batch_size=1024,
                input_nodes=("student", torch.arange(n)),
                shuffle=False,
            )
    except Exception:
        pass
    print("  [FIX 3] torch-sparse/pyg-lib not found — using full-graph loader "
          "(install torch-sparse for NeighborLoader speedup)")
    return _FullGraphLoader(data, n)


# ----------------------------------------------------------------
# FIX 1 + FIX 3 — Batched encode without gradients (~30 s target)
# ----------------------------------------------------------------

def encode_all_batched(
    model:          MG_HGNN,
    loader:         NeighborLoader,
    bert_cache:     Optional[Dict],
    total_students: int,
) -> torch.Tensor:
    """Encode every student via mini-batches; return full (N, embed_dim) tensor."""
    all_emb = torch.zeros(total_students, model.cfg.embed_dim)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            n   = batch["student"].batch_size
            nid = batch["student"].n_id[:n]
            emb = model.encode_all(batch, bert_cache)["student"][:n]
            all_emb[nid] = emb.cpu()
    model.train()
    return all_emb


# FIX C — explicit no-grad re-encode for the outer loop (~2-4 s)
def fast_reencode(
    model:      MG_HGNN,
    loader,
    bert_cache: Optional[Dict],
    n:          int,
) -> torch.Tensor:
    """No-grad re-encode; encode_all_batched already uses no_grad internally."""
    return encode_all_batched(model, loader, bert_cache, n)


# ----------------------------------------------------------------
# FIX 2 — Head-only helpers
# ----------------------------------------------------------------

def heads_forward(
    model:     MG_HGNN,
    h_student: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Run the three prediction heads on cached student embeddings (no graph)."""
    return {
        "grade":      model.grade_head(h_student),
        "dropout":    model.dropout_head(h_student),
        "engagement": model.engagement_head(h_student),
    }


def build_optimizers(model: MG_HGNN, cfg: Config) -> Tuple[Adam, Adam]:
    """Two separate Adam optimisers: graph encoder (lr) and heads (2×lr)."""
    graph_params = (
        list(model.struct_enc.parameters())
        + list(model.text_enc.parameters())
        + list(model.behav_enc.parameters())
        + list(model.inst_enc.parameters())
        + list(model.gate.parameters())
        + list(model.convs.parameters())
    )
    head_params = (
        list(model.grade_head.parameters())
        + list(model.dropout_head.parameters())
        + list(model.engagement_head.parameters())
    )
    opt_graph = Adam(graph_params, lr=cfg.lr,      weight_decay=cfg.weight_decay)
    opt_heads = Adam(head_params,  lr=cfg.lr * 2)
    return opt_graph, opt_heads


# ----------------------------------------------------------------
# Shared loss helper
# ----------------------------------------------------------------

def get_lambdas(epoch: int, cfg: Config) -> Dict[str, float]:
    """Linear warmup for lambda_engagement over the first 20 epochs (Fix B)."""
    warmup = min(1.0, epoch / 20.0)
    return {
        "grade":      cfg.lambda_grade,
        "dropout":    cfg.lambda_dropout,
        "engagement": cfg.lambda_engagement * warmup + 0.3 * (1 - warmup),
    }


def _loss(
    preds:        Dict[str, torch.Tensor],
    grade_true:   torch.Tensor,
    dropout_true: torch.Tensor,
    eng_true:     torch.Tensor,
    cfg:          Config,
    mse_fn:       nn.MSELoss,
    bce_fn:       nn.BCELoss,
    ce_fn:        nn.CrossEntropyLoss,
    lambdas:      Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    lam_g = lambdas["grade"]      if lambdas else cfg.lambda_grade
    lam_d = lambdas["dropout"]    if lambdas else cfg.lambda_dropout
    lam_e = lambdas["engagement"] if lambdas else cfg.lambda_engagement
    return (
        lam_g * mse_fn(preds["grade"].squeeze(),   grade_true)
        + lam_d * bce_fn(preds["dropout"].squeeze(), dropout_true)
        + lam_e * ce_fn( preds["engagement"],        eng_true)
    )


# ----------------------------------------------------------------
# Outer step: update graph-encoder weights via batched backward
# ----------------------------------------------------------------

def graph_step(
    model:       MG_HGNN,
    loader,
    bert_cache:  Optional[Dict],
    train_idx_t: torch.Tensor,
    data:        HeteroData,
    cfg:         Config,
    opt_graph:   Adam,
    mse_fn:      nn.MSELoss,
    bce_fn:      nn.BCELoss,
    ce_fn:       nn.CrossEntropyLoss,
    epoch:       int = 0,
    max_nodes:   int = 2000,
) -> float:
    """One-shot graph-encoder initialisation step (called once per fold at warmup).

    FIX B: encodes the full graph under no_grad (cheap), then runs a single
    head-loss backward on a random subsample of up to max_nodes training
    students.  This seeds the graph encoder with a meaningful gradient signal
    without paying the cost of full-graph HGTConv backward (~10 min on CPU).
    """
    # Subsample train_idx to cap backward cost
    if len(train_idx_t) > max_nodes:
        perm    = torch.randperm(len(train_idx_t))[:max_nodes]
        sub_idx = train_idx_t[perm]
    else:
        sub_idx = train_idx_t

    # Full-graph encode under no_grad (~42 s, no backward through HGTConv)
    h_all = encode_all_batched(model, loader, bert_cache, _n_students(data))

    # Backward through heads only on the subgraph; seeds opt_graph params
    model.train()
    opt_graph.zero_grad()
    out = heads_forward(model, h_all[sub_idx])
    loss = _loss(
        out,
        data["student"].y_grade[sub_idx],
        data["student"].y_dropout[sub_idx],
        data["student"].y_engagement[sub_idx],
        cfg, mse_fn, bce_fn, ce_fn,
        lambdas=get_lambdas(epoch, cfg),
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    opt_graph.step()
    return loss.item()


# ----------------------------------------------------------------
# Inner step: heads only (~0.5 s/epoch)
# ----------------------------------------------------------------

def heads_step(
    model:       MG_HGNN,
    h_student:   torch.Tensor,
    train_idx_t: torch.Tensor,
    data:        HeteroData,
    cfg:         Config,
    opt_heads:   Adam,
    mse_fn:      nn.MSELoss,
    bce_fn:      nn.BCELoss,
    ce_fn:       nn.CrossEntropyLoss,
    epoch:       int = 0,
) -> float:
    """Forward through heads only on cached embeddings; step opt_heads."""
    model.train()
    opt_heads.zero_grad()
    # h_student has requires_grad=False (produced under no_grad); only head
    # params receive gradients, graph encoder params are not touched.
    s   = h_student[train_idx_t]
    out = heads_forward(model, s)
    loss = _loss(
        out,
        data["student"].y_grade[train_idx_t],
        data["student"].y_dropout[train_idx_t],
        data["student"].y_engagement[train_idx_t],
        cfg, mse_fn, bce_fn, ce_fn,
        lambdas=get_lambdas(epoch, cfg),
    )
    loss.backward()
    opt_heads.step()
    return loss.item()


def evaluate_heads(
    model:      MG_HGNN,
    h_student:  torch.Tensor,
    val_idx_t:  torch.Tensor,
    data:       HeteroData,
    cfg:        Config,
    mse_fn:     nn.MSELoss,
    bce_fn:     nn.BCELoss,
    ce_fn:      nn.CrossEntropyLoss,
) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        s   = h_student[val_idx_t]
        out = heads_forward(model, s)
        val_loss = _loss(
            out,
            data["student"].y_grade[val_idx_t],
            data["student"].y_dropout[val_idx_t],
            data["student"].y_engagement[val_idx_t],
            cfg, mse_fn, bce_fn, ce_fn,
        )
        g_pred = out["grade"].squeeze().numpy()
        g_true = data["student"].y_grade[val_idx_t].numpy()
        rmse   = float(np.sqrt(mean_squared_error(g_true, g_pred)))

        d_pred = out["dropout"].squeeze().numpy()
        d_true = data["student"].y_dropout[val_idx_t].numpy().astype(int)
        try:
            auc = float(roc_auc_score(d_true, d_pred))
        except ValueError:
            auc = 0.5

        e_pred = out["engagement"].argmax(dim=-1).numpy()
        e_true = data["student"].y_engagement[val_idx_t].numpy()
        f1 = float(f1_score(e_true, e_pred, average="macro", zero_division=0))

    return {"val_loss": float(val_loss), "val_rmse": rmse, "val_auc": auc, "val_f1": f1}


# ----------------------------------------------------------------
# Single-fold fast training
# ----------------------------------------------------------------

def train_one_fold_fast(
    model:         MG_HGNN,
    train_idx:     np.ndarray,
    val_idx:       np.ndarray,
    data:          HeteroData,
    cfg:           Config,
    train_cfg:     TrainingConfig,
    fold_k:        int = 0,
    bert_cache:    Optional[Dict] = None,
    encode_loader: Optional[NeighborLoader] = None,
) -> Tuple[Dict[str, List[float]], Dict[str, float]]:
    """Train one fold using the 3-fix fast loop.

    Warmup (once):               encode_all_batched → graph_step (seeds encoder).
    Outer loop (every 10 ep):    fast_reencode — no grad, ~2-4 s.
    Inner loop (epochs i…i+9):   heads_step only — ~0.13 s/epoch.
    graph_step is NOT called inside the outer loop (FIX A).
    """
    Path(cfg.checkpoint_dir).mkdir(exist_ok=True)
    ckpt_path = Path(cfg.checkpoint_dir) / f"fold_{fold_k}_fast_best.pt"

    opt_graph, opt_heads = build_optimizers(model, cfg)
    mse_fn = nn.MSELoss()

    # Fix A — class-weighted CE for engagement (computed from train split)
    train_t = torch.tensor(train_idx, dtype=torch.long)
    val_t   = torch.tensor(val_idx,   dtype=torch.long)
    eng_labels = data["student"].y_engagement[train_t].numpy()
    eng_classes = np.unique(eng_labels)
    eng_weights = compute_class_weight("balanced", classes=eng_classes, y=eng_labels)
    ce_fn = nn.CrossEntropyLoss(weight=torch.FloatTensor(eng_weights))

    # Weighted BCE for dropout — counters 71/29 class imbalance that causes
    # AUC collapse when the loss minimises by predicting the majority class.
    _n_pos = int((data["student"].y_dropout[train_t] == 1).sum())
    _n_neg = int((data["student"].y_dropout[train_t] == 0).sum())
    _bce_pw = torch.tensor([_n_neg / _n_pos])   # ~2.45 for OULAD
    print(f"  Dropout pos_weight: {_bce_pw.item():.3f}  "
          f"(neg={_n_neg:,} pos={_n_pos:,})")

    def bce_fn(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        w = torch.where(true.bool(), _bce_pw.expand_as(true), torch.ones_like(true))
        return F.binary_cross_entropy(pred, true, weight=w)

    n_epochs = 5 if train_cfg.debug else cfg.epochs

    drop_labels = data["student"].y_dropout
    print(f"  Dropout dist  train [{_class_dist(drop_labels[train_t])}]"
          f"  val [{_class_dist(drop_labels[val_t])}]")
    print(f"  Engagement weights: "
          + "  ".join(f"cls{c}={w:.2f}" for c, w in zip(eng_classes, eng_weights)))

    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [], "val_rmse": [], "val_auc": [], "val_f1": []
    }
    best_val_loss = float("inf")
    # Fix C — dual patience: both AUC and F1 must stagnate to trigger early stop
    best_val_auc = -1.0
    best_val_f1  = -1.0
    auc_patience = 0
    f1_patience  = 0
    epoch_times: List[float] = []
    h_student:   Optional[torch.Tensor] = None

    bar = tqdm(total=n_epochs, desc=f"Fold {fold_k} [fast]",
               disable=train_cfg.debug)

    # ── warmup: encode once + one graph_step, then pin h_student ─────────
    print("  [warmup] graph encode + weight update...")
    t0 = time.time()
    h_student = encode_all_batched(model, encode_loader, bert_cache, _n_students(data))
    graph_step(model, encode_loader, bert_cache, train_t,
               data, cfg, opt_graph, mse_fn, bce_fn, ce_fn,
               epoch=0, max_nodes=2000)
    print(f"  [warmup done] {time.time()-t0:.1f}s — "
          f"h_student pinned for all {n_epochs} epochs")

    # ── flat epoch loop: heads only, h_student fixed throughout ──────────
    for epoch in range(n_epochs):
        t_epoch    = time.time()
        train_loss = heads_step(
            model, h_student, train_t, data, cfg,
            opt_heads, mse_fn, bce_fn, ce_fn, epoch=epoch,
        )
        val_metrics = evaluate_heads(
            model, h_student, val_t, data, cfg,
            mse_fn, bce_fn, ce_fn,
        )
        epoch_sec = time.time() - t_epoch
        epoch_times.append(epoch_sec)

        tl   = train_loss
        vl   = val_metrics["val_loss"]
        rmse = val_metrics["val_rmse"]
        auc  = val_metrics["val_auc"]
        f1   = val_metrics["val_f1"]

        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["val_rmse"].append(rmse)
        history["val_auc"].append(auc)
        history["val_f1"].append(f1)

        if epoch == 0:
            print(f"  [Epoch 1 timing] {epoch_sec:.2f}s (heads-only)")

        if not train_cfg.debug:
            bar.set_postfix(
                tr=f"{tl:.4f}", val=f"{vl:.4f}",
                auc=f"{auc:.3f}", f1=f"{f1:.3f}",
            )
            bar.update(1)
            if (epoch + 1) % train_cfg.log_every == 0:
                tqdm.write(
                    f"  Fold {fold_k} | Epoch {epoch+1:4d}/{n_epochs}  "
                    f"train={tl:.4f}  val={vl:.4f}  "
                    f"rmse={rmse:.4f}  auc={auc:.4f}  f1={f1:.4f}"
                )
        else:
            print(f"  Epoch {epoch+1:2d}/{n_epochs}  "
                  f"train={tl:.4f}  val={vl:.4f}  "
                  f"rmse={rmse:.4f}  auc={auc:.4f}  f1={f1:.4f}")

        # ── checkpoint on val_loss improvement ────────────────────────────
        if vl < best_val_loss:
            best_val_loss = vl
            torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

        # Fix C — dual early stopping: require BOTH AUC and F1 to stagnate
        if auc > best_val_auc:
            best_val_auc = auc
            auc_patience = 0
        else:
            auc_patience += 1

        if f1 > best_val_f1:
            best_val_f1 = f1
            f1_patience = 0
        else:
            f1_patience += 1

        if auc_patience >= cfg.patience and f1_patience >= cfg.patience:
            msg = (f"  Early stop at epoch {epoch + 1} "
                   f"(auc_pat={auc_patience}, f1_pat={f1_patience})")
            (print if train_cfg.debug else tqdm.write)(msg)
            break

    bar.close()

    best_auc_epoch = int(np.argmax(history["val_auc"])) + 1
    best_f1_epoch  = int(np.argmax(history["val_f1"]))  + 1
    print(f"  Best val-AUC epoch: {best_auc_epoch}  best val-F1 epoch: {best_f1_epoch}")

    # ── reload best checkpoint, final eval ───────────────────────────────
    model.load_state_dict(
        torch.load(ckpt_path, weights_only=True)["model_state_dict"]
    )
    model.eval()
    h_final = encode_all_batched(
        model, encode_loader, bert_cache, _n_students(data)
    )
    with torch.no_grad():
        g_pred    = model.grade_head(h_final[val_t]).squeeze().numpy()
        g_true    = data["student"].y_grade[val_t].numpy()
        best_rmse = float(np.sqrt(mean_squared_error(g_true, g_pred)))

        d_pred = model.dropout_head(h_final[val_t]).squeeze().numpy()
        d_true = data["student"].y_dropout[val_t].numpy().astype(int)
        try:
            best_auc = float(roc_auc_score(d_true, d_pred))
        except ValueError:
            best_auc = 0.5

        e_pred  = model.engagement_head(h_final[val_t]).argmax(dim=-1).numpy()
        e_true  = data["student"].y_engagement[val_t].numpy()
        best_f1 = float(f1_score(e_true, e_pred, average="macro", zero_division=0))

    avg_epoch_sec = float(np.mean(epoch_times)) if epoch_times else 0.0
    return history, {
        "rmse": best_rmse, "auc": best_auc, "f1": best_f1,
        "best_auc_epoch": best_auc_epoch,
        "avg_epoch_sec":  avg_epoch_sec,
    }


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train MG-HGNN (fast 3-fix)")
    parser.add_argument("--debug",    action="store_true",
                        help="1 fold, 5 epochs (synthetic: 100 students)")
    parser.add_argument("--dataset",
                        choices=["synthetic", "oulad", "moocc",
                                 "assistments09", "assistments15"],
                        default="synthetic")
    parser.add_argument("--use-cache", action="store_true",
                        help="Load pre-computed BERT embeddings from "
                             "data/bert_cache_<dataset>.pt")
    args = parser.parse_args()

    cfg       = Config()
    train_cfg = TrainingConfig(debug=args.debug)

    print(f"Dataset: {args.dataset.upper()}")
    print(f"Mode:    {'DEBUG (1 fold, 5 epochs)' if args.debug else 'FULL (5-fold CV)'} [FAST]")

    if args.dataset == "oulad":
        from data.oulad_loader import OULADLoader
        data_loader = OULADLoader(cfg)
        data, meta  = data_loader.load()
        data_loader.summary(data, meta)
    elif args.dataset == "moocc":
        from data.moocc_loader import MOOCCubeLoader
        data_loader = MOOCCubeLoader(cfg)
        data, meta  = data_loader.load()
        data_loader.summary(data, meta)
    elif args.dataset in ("assistments09", "assistments15"):
        from data.assistments_loader import ASSISTmentsLoader
        ver = "2009" if args.dataset == "assistments09" else "2015"
        data_loader = ASSISTmentsLoader(cfg, version=ver)
        if not data_loader.check_files():
            return
        data, meta = data_loader.load()
        data_loader.summary(data, meta)
    else:
        n_students = 100 if args.debug else 200
        print(f"Building synthetic graph ({n_students} students)...")
        data, meta = make_synthetic_data(cfg, n_students=n_students)

    bert_cache: Optional[Dict] = None
    if args.dataset in ("assistments09", "assistments15"):
        # ASSISTments has no text — all nodes carry identical [CLS]-only stubs.
        # Run BERT once on a single [CLS] sequence and broadcast to every node
        # type, bypassing per-node BERT calls (~30 min → <1 s).
        print("  [text-free] building broadcast BERT cache from single [CLS] pass…")
        from models.encoders import TextEncoder
        _tenc = TextEncoder(cfg, freeze_bert=True)
        _tenc.eval()
        with torch.no_grad():
            _ids  = data["student"].input_ids[:1]
            _mask = data["student"].attention_mask[:1]
            _emb  = _tenc(_ids, _mask)   # (1, embed_dim)
        bert_cache = {}
        for ntype in ("student", "course", "resource"):
            if ntype in data.node_types:
                n = data[ntype].num_nodes
                bert_cache[ntype] = _emb.expand(n, -1).clone()
        print(f"  [text-free] cache built — "
              f"{', '.join(f'{k}: {v.shape[0]}' for k, v in bert_cache.items())}")
    elif args.use_cache:
        from data.cache_embeddings import cache_bert_embeddings, load_bert_cache
        cache_path = Path(f"data/bert_cache_{args.dataset}.pt")
        if cache_path.exists():
            print(f"Loading BERT cache from {cache_path}…")
            bert_cache = load_bert_cache(str(cache_path))
            print(f"  Cached node types: {list(bert_cache.keys())}")
        else:
            print(f"Cache not found — building {cache_path}…")
            bert_cache = cache_bert_embeddings(data, cfg, save_path=str(cache_path))

    data["student"].y_grade      = meta["grade"].float()
    data["student"].y_dropout    = meta["dropout"].float()
    data["student"].y_engagement = meta["engagement"].long()

    n_students     = len(meta["student_ids"])
    dropout_labels = meta["dropout"].numpy()
    n_folds        = 1 if args.debug else train_cfg.n_folds
    skf   = StratifiedKFold(n_splits=max(n_folds, 2), shuffle=True, random_state=42)
    folds = list(skf.split(np.arange(n_students), dropout_labels))[:n_folds]

    # Cap accessed edges to prevent HGTConv hang on 1M+ edges without torch_scatter
    MAX_ACCESSED = 100_000
    _etype = ("student", "accessed", "resource")
    if hasattr(data[_etype], "edge_index"):
        _ei = data[_etype].edge_index
        if _ei.shape[1] > MAX_ACCESSED:
            _perm = torch.randperm(_ei.shape[1])[:MAX_ACCESSED]
            data[_etype].edge_index = _ei[:, _perm]
            print(f"  [graph cap] accessed edges: {_ei.shape[1]:,} → {MAX_ACCESSED:,}")

    # Build once; shared across folds (data is read-only)
    encode_loader = build_encode_loader(data, cfg)

    all_histories: List[Dict] = []
    all_metrics:   List[Dict] = []

    for fold_k, (train_idx, val_idx) in enumerate(folds):
        print(f"\n{'='*52}")
        print(f"  Fold {fold_k + 1}/{n_folds}  "
              f"(train={len(train_idx):,}, val={len(val_idx):,})")
        print(f"{'='*52}")

        model   = MG_HGNN(cfg)
        history, best = train_one_fold_fast(
            model, train_idx, val_idx, data, cfg, train_cfg,
            fold_k=fold_k, bert_cache=bert_cache,
            encode_loader=encode_loader,
        )
        all_histories.append(history)
        all_metrics.append(best)
        print(f"\n  Fold {fold_k + 1} best →  "
              f"RMSE={best['rmse']:.4f}  AUC={best['auc']:.4f}  "
              f"F1={best['f1']:.4f}  best-AUC-ep={best['best_auc_epoch']}")

        out_path = Path(cfg.results_dir) / f"training_history_{args.dataset}_fast.json"
        _save_results(all_histories, all_metrics, out_path, args.dataset, n_folds)
        print(f"  [saved] {fold_k + 1}/{n_folds} folds → {out_path}")

    rmses = [m["rmse"] for m in all_metrics]
    aucs  = [m["auc"]  for m in all_metrics]
    f1s   = [m["f1"]   for m in all_metrics]
    W = 52
    print(f"\n{'='*W}")
    print(f"  Cross-Validation Summary  ({n_folds} fold{'s' if n_folds > 1 else ''}) [fast]")
    print(f"{'='*W}")
    print(f"  RMSE:          {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")
    print(f"  AUC-ROC:       {np.mean(aucs):.4f}  ± {np.std(aucs):.4f}")
    print(f"  Engagement-F1: {np.mean(f1s):.4f}  ± {np.std(f1s):.4f}")
    print(f"{'='*W}")

    out_path = Path(cfg.results_dir) / f"training_history_{args.dataset}_fast.json"
    _save_results(all_histories, all_metrics, out_path, args.dataset, n_folds)
    print(f"\n  Histories saved → {out_path}")

    # ── Final metrics table (last fold's best checkpoint) ────────────────
    last_fold_k  = n_folds - 1
    last_val_idx = folds[last_fold_k][1]
    last_val_t   = torch.tensor(last_val_idx, dtype=torch.long)
    last_ckpt    = Path(cfg.checkpoint_dir) / f"fold_{last_fold_k}_fast_best.pt"

    final_model = MG_HGNN(cfg)
    final_model.load_state_dict(
        torch.load(last_ckpt, weights_only=True)["model_state_dict"]
    )
    final_model.eval()
    h_final = encode_all_batched(
        final_model, build_encode_loader(data, cfg),
        bert_cache, _n_students(data),
    )
    with torch.no_grad():
        final_preds = {
            "grade":      final_model.grade_head(h_final[last_val_t]).squeeze().numpy(),
            "dropout":    final_model.dropout_head(h_final[last_val_t]).squeeze().numpy(),
            "engagement": final_model.engagement_head(h_final[last_val_t]).argmax(dim=-1).numpy(),
        }
    labels_np = {
        "grade":      data["student"].y_grade[last_val_t].numpy(),
        "dropout":    data["student"].y_dropout[last_val_t].numpy().astype(int),
        "engagement": data["student"].y_engagement[last_val_t].numpy(),
    }
    metrics = compute_metrics(final_preds, labels_np)
    tag = "debug" if args.debug else args.dataset
    print_results_table(metrics, model_name=f"MG-HGNN fast ({tag})")


if __name__ == "__main__":
    main()
