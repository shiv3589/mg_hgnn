import json
import torch
from config import Config
from models.mg_hgnn import MG_HGNN
from evaluate import visualize_gate_weights

cfg = Config()

with open('results/training_history_oulad_fast.json') as f:
    hist = json.load(f)

best_metrics = hist['best_metrics']          # list, index == fold number
best_fold = max(range(len(best_metrics)), key=lambda i: best_metrics[i]['auc'])
best_auc = best_metrics[best_fold]['auc']
print(f'Best fold: fold_{best_fold}  AUC: {best_auc:.4f}')

ckpt_path = f'checkpoints/fold_{best_fold}_fast_best.pt'
print(f'Loading: {ckpt_path}')
model = MG_HGNN(cfg)
ckpt  = torch.load(ckpt_path, map_location='cpu', weights_only=True)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

print()
print('=== TRAINED gate weights (best fold, should be differentiated) ===')
for r in cfg.edge_types:
    alpha = model.gate.get_gate_weights(r)
    print(f'  {r:<25}: {alpha.round(3).tolist()}')

print()
print('=== Gate weights across all 5 folds ===')
for fold_num in range(len(best_metrics)):
    ckpt_path_f = f'checkpoints/fold_{fold_num}_fast_best.pt'
    model2 = MG_HGNN(cfg)
    ckpt2  = torch.load(ckpt_path_f, map_location='cpu', weights_only=True)
    model2.load_state_dict(ckpt2['model_state_dict'])
    model2.eval()
    print(f'  fold_{fold_num}  (AUC={best_metrics[fold_num]["auc"]:.4f}):')
    for r in cfg.edge_types:
        alpha = model2.gate.get_gate_weights(r)
        print(f'    {r:<25}: {alpha.round(3).tolist()}')

# Generate corrected heatmap from the best-fold TRAINED model — overwrite the wrong one
visualize_gate_weights(model, save_path='results/gate_heatmap_CORRECTED.pdf')
visualize_gate_weights(model, save_path='paper/figures/gate_heatmap.pdf')
visualize_gate_weights(model, save_path='results/gate_heatmap.pdf')

print()
print('Corrected heatmap saved to:')
print('  results/gate_heatmap_CORRECTED.pdf')
print('  paper/figures/gate_heatmap.pdf')
print('  results/gate_heatmap.pdf')

# Verify creation date is NOW (not the old 2026-05-06) — read PDF metadata directly,
# no pdfinfo binary on this machine.
import re
for f in ['paper/figures/gate_heatmap.pdf', 'results/gate_heatmap.pdf']:
    data = open(f, 'rb').read()
    m = re.search(rb'/CreationDate\s*\((D:[^)]*)\)', data)
    print(f'  {f}  CreationDate: {m.group(1).decode() if m else "not found"}')
