"""
Pre-compute frozen BERT embeddings for all text-bearing node types
and save them to disk.  Training epochs then load from the cache
instead of running the BERT forward pass each time.

Usage (run from mg_hgnn/mg_hgnn/):
    python data/cache_embeddings.py --dataset oulad
    python data/cache_embeddings.py --dataset moocc
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import torch
from torch_geometric.data import HeteroData

# When run as `python data/cache_embeddings.py`, Python adds data/ to sys.path
# but config.py lives one level up.  Add the project root explicitly.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import Config


def cache_bert_embeddings(
    data:      HeteroData,
    config:    Config,
    save_path: str = "data/bert_cache.pt",
    batch_size: int = 64,
) -> Dict[str, torch.Tensor]:
    """
    Run TextEncoder once on all node text features and save to disk.

    Iterates over student / course / resource nodes, batching in groups
    of `batch_size` to stay within GPU/CPU memory.  Students in datasets
    without real text (OULAD) still get cached — their [CLS]-only embeddings
    are identical but pre-computing avoids 28 k BERT calls per epoch.

    Parameters
    ----------
    data      : HeteroData returned by any loader
    config    : Config (used to instantiate TextEncoder)
    save_path : where to write the cache .pt file
    batch_size: BERT mini-batch size (reduce if OOM)

    Returns
    -------
    cache dict  {node_type: FloatTensor (N, embed_dim)}
    """
    from models.encoders import TextEncoder

    encoder = TextEncoder(config, freeze_bert=True)
    encoder.eval()

    cache: Dict[str, torch.Tensor] = {}
    t0 = time.time()

    for node_type in ("student", "course", "resource"):
        store = data[node_type]
        # Both loaders use input_ids / attention_mask
        if not (hasattr(store, "input_ids") and hasattr(store, "attention_mask")):
            print(f"  Skipping {node_type} — no text attributes.")
            continue

        ids  = store.input_ids
        mask = store.attention_mask
        N    = ids.shape[0]

        print(f"  Encoding {node_type} ({N:,} nodes) in batches of {batch_size}…", end="", flush=True)
        t1 = time.time()

        # BROADCAST FIX (2026-08-14): node types without real per-node
        # text (e.g. MOOCCube's students -- no student text exists in
        # that dataset at all, so every row is the identical [CLS]-only
        # stub) were running N separate BERT forward passes for one
        # distinct input. Confirmed on the moocc run: this stalled the
        # cache job for 40+ min encoding 199,199 identical students and
        # drove free RAM from 18GB down to 4GB before it was killed.
        # Detect a uniform input across the whole node type and encode
        # it exactly once instead. Applies to any node type this is true
        # for (not just "student" specifically) -- OULAD's student stub
        # is uniform too (see this function's docstring), so it benefits
        # from this fix as well, with identical output values either way.
        uniform = N > 1 and bool((ids == ids[0]).all() and (mask == mask[0]).all())
        if uniform:
            print(f" uniform input detected — encoding once, broadcasting to {N:,}…",
                  end="", flush=True)
            with torch.no_grad():
                one_emb = encoder(ids[0:1], mask[0:1])       # (1, embed_dim)
            cache[node_type] = one_emb.expand(N, -1).contiguous()
        else:
            embs = []
            with torch.no_grad():
                for start in range(0, N, batch_size):
                    e = encoder(ids[start:start + batch_size],
                                mask[start:start + batch_size])
                    embs.append(e)
            cache[node_type] = torch.cat(embs, dim=0)   # (N, embed_dim)

        elapsed = time.time() - t1
        print(f"  done in {elapsed:.1f}s  shape={cache[node_type].shape}")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, save_path)
    total = time.time() - t0
    print(f"BERT cache saved → {save_path}  (total {total:.1f}s)")
    return cache


def cache_full_embeddings(
    data,
    model,
    config,
    save_path: str = "data/full_embed_cache.pt",
) -> dict:
    """
    Run ONE full forward pass through encoders + HGTConv.
    Cache the final student node embeddings (post-graph).
    Only the 3 prediction heads need to run each epoch.
    This is valid because BERT is frozen and the graph
    structure does not change between epochs.
    NOTE: invalidate this cache if HGTConv weights update
    significantly — rebuild every 10 epochs.
    """
    import os
    model.eval()
    with torch.no_grad():
        h = model.encode_all(data)   # returns dict of node embeddings
    torch.save(h, save_path)
    size_mb = os.path.getsize(save_path) / 1e6
    print(f"Full embed cache saved: {size_mb:.1f} MB → {save_path}")
    return h


def load_bert_cache(
    save_path: str,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    cache = torch.load(save_path, weights_only=True)
    if device is not None:
        cache = {k: v.to(device) for k, v in cache.items()}
    return cache


# ----------------------------------------------------------------
# CLI entry-point
# ----------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-compute BERT node embeddings")
    parser.add_argument("--dataset", choices=["oulad", "moocc", "synthetic"],
                        default="oulad")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out", default=None,
                        help="Override output path (default: data/bert_cache_<dataset>.pt)")
    args = parser.parse_args()

    cfg = Config()

    print(f"Dataset: {args.dataset.upper()}")

    if args.dataset == "oulad":
        from data.oulad_loader import OULADLoader
        loader = OULADLoader(cfg)
        data, _ = loader.load()
    elif args.dataset == "moocc":
        from data.moocc_loader import MOOCCubeLoader
        loader = MOOCCubeLoader(cfg)
        data, _ = loader.load()
    else:
        from train import make_synthetic_data
        data, _ = make_synthetic_data(cfg)

    out_path = args.out or f"data/bert_cache_{args.dataset}.pt"
    cache = cache_bert_embeddings(data, cfg,
                                  save_path=out_path,
                                  batch_size=args.batch_size)
    print("Done. Node types cached:", list(cache.keys()))
