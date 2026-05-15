#!/usr/bin/env bash
set -euo pipefail
cd /Users/parul/mg_hgnn/mg_hgnn
mkdir -p logs results checkpoints paper/figures

# ── helper: generate gate heatmap from fold-0 checkpoint ──────────
heatmap() {
  local label=$1 outpath=$2
  python3 -c "
import torch
from config import Config
from models.mg_hgnn import MG_HGNN
from evaluate import visualize_gate_weights
cfg   = Config()
model = MG_HGNN(cfg)
ckpt  = torch.load('checkpoints/fold_0_fast_best.pt', map_location='cpu', weights_only=True)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
visualize_gate_weights(model, save_path='$outpath')
print('Saved: $outpath')
" 2>&1
}

# ── helper: archive checkpoints with dataset tag ───────────────────
archive() {
  local tag=$1
  for f in checkpoints/fold_*_fast_best.pt; do
    cp "$f" "${f/fold_/fold_${tag}_}" 2>/dev/null || true
  done
}

echo "========================================================"
echo " RUN 1 — ASSISTments 2009 full 5-fold training"
echo "========================================================"
python3 train_fast.py --dataset assistments09 2>&1 | tee logs/run_assist09.log
heatmap assist09 paper/figures/gate_heatmap_assist09.pdf
archive assist09

echo "========================================================"
echo " RUN 2 — ASSISTments 2015 full 5-fold training"
echo "========================================================"
python3 train_fast.py --dataset assistments15 2>&1 | tee logs/run_assist15.log
heatmap assist15 paper/figures/gate_heatmap_assist15.pdf
archive assist15

echo "========================================================"
echo " RUN 3 — Baselines ASSISTments 2009"
echo "========================================================"
python3 baselines.py --dataset assistments09 2>&1 | tee logs/baselines_assist09.log

echo "========================================================"
echo " RUN 4 — Baselines ASSISTments 2015"
echo "========================================================"
python3 baselines.py --dataset assistments15 2>&1 | tee logs/baselines_assist15.log

echo "========================================================"
echo " RUN 5 — Ablation ASSISTments 2009"
echo "========================================================"
python3 run_ablation.py --dataset assistments09 2>&1 | tee logs/ablation_assist09.log

echo "========================================================"
echo " RUN 6 — Ablation ASSISTments 2015"
echo "========================================================"
python3 run_ablation.py --dataset assistments15 2>&1 | tee logs/ablation_assist15.log

echo "========================================================"
echo " ALL RUNS DONE"
echo "========================================================"
