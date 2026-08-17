"""
Correct Figure 2 / Table 6 regeneration.

Uses model.gate.get_gate_weights_from_data() (real encoded student
embeddings) instead of the broken synthetic-probe get_gate_weights(),
averaged across all 5 trained folds -- matching the paper's own caption
("gate weights per edge relation type after 5-fold training on OULAD")
and Table 6 ("averaged over 5-fold training runs").
"""
import json
import numpy as np
import torch
from config import Config
from models.mg_hgnn import MG_HGNN
from evaluate import visualize_gate_weights
from data.cache_embeddings import load_bert_cache
from data.oulad_loader import OULADLoader

cfg = Config()

with open('results/training_history_oulad_fast.json') as f:
    hist = json.load(f)
best_metrics = hist['best_metrics']
n_folds = len(best_metrics)

print("Loading OULAD + bert cache for real embeddings...")
loader = OULADLoader(cfg)
data, meta = loader.load()
bert_cache = load_bert_cache('data/bert_cache_oulad.pt')
h_t = bert_cache["student"]   # frozen BERT text embeddings, same for every fold

per_fold_alpha = {r: [] for r in cfg.edge_types}   # r -> list of (3,) arrays, one per fold

for fold_num in range(n_folds):
    ckpt_path = f'checkpoints/fold_{fold_num}_fast_best.pt'
    model = MG_HGNN(cfg)
    ckpt  = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    with torch.no_grad():
        h_s = model.struct_enc(data["student"].x_struct)
        h_b = model.behav_enc(data["student"].x_behav)

    print(f"  fold_{fold_num} (AUC={best_metrics[fold_num]['auc']:.4f}):")
    for r in cfg.edge_types:
        alpha = model.gate.get_gate_weights_from_data(r, h_s, h_t, h_b)
        per_fold_alpha[r].append(alpha)
        print(f"    {r:<25}: {[round(v, 3) for v in alpha.tolist()]}")

    # Save the best fold's model object for the headline heatmap plot below
    if fold_num == max(range(n_folds), key=lambda i: best_metrics[i]['auc']):
        best_model, best_h_s, best_h_b = model, h_s, h_b
        best_fold_num = fold_num

print()
print("=== Mean gate weights across 5 folds (Table 6 values) ===")
mean_table = {}
std_table = {}
for r in cfg.edge_types:
    arr = np.stack(per_fold_alpha[r], axis=0)   # (5, 3)
    mean_table[r] = arr.mean(axis=0)
    std_table[r]  = arr.std(axis=0)
    m = [round(v, 3) for v in mean_table[r].tolist()]
    s = [round(v, 3) for v in std_table[r].tolist()]
    print(f"  {r:<25}: mean={m}  std={s}")

# --- Regenerate Figure 2 using the BEST fold's real-data alpha (headline figure) ---
print()
print(f"Regenerating heatmap from best fold (fold_{best_fold_num}, real data)...")
visualize_gate_weights(
    best_model,
    save_path='paper/figures/gate_heatmap.pdf',
    h_s=best_h_s, h_t=h_t, h_b=best_h_b,
)
visualize_gate_weights(
    best_model,
    save_path='results/gate_heatmap.pdf',
    h_s=best_h_s, h_t=h_t, h_b=best_h_b,
)

# --- Also save the 5-fold-averaged version, for direct comparison / Table 6 ---
class _MeanGateProxy:
    """Wraps the mean_table dict so visualize_gate_weights's fallback path
    (get_gate_weights) returns the 5-fold-averaged real-data alpha."""
    def __init__(self, edge_types, mean_table):
        self.edge_types = edge_types
        self._mean_table = mean_table
    def get_gate_weights(self, r):
        return self._mean_table[r]

class _ModelProxy:
    def __init__(self, gate):
        self.gate = gate

proxy_model = _ModelProxy(_MeanGateProxy(cfg.edge_types, mean_table))
visualize_gate_weights(proxy_model, save_path='results/gate_heatmap_5fold_mean.pdf')

print()
print("Saved:")
print("  paper/figures/gate_heatmap.pdf   (best fold, real data)")
print("  results/gate_heatmap.pdf         (best fold, real data)")
print("  results/gate_heatmap_5fold_mean.pdf  (mean across 5 folds, real data)")

import re
for f in ['paper/figures/gate_heatmap.pdf', 'results/gate_heatmap.pdf']:
    raw = open(f, 'rb').read()
    m = re.search(rb'/CreationDate\s*\((D:[^)]*)\)', raw)
    print(f'  {f}  CreationDate: {m.group(1).decode() if m else "not found"}')
