"""Plot per-fold val-AUC curves from training_history JSON."""
import json
import sys
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "results/training_history_oulad_fast.json"
d = json.load(open(path))
folds = d["folds"]
n = len(folds)

fig, ax = plt.subplots(figsize=(9, 5))
palette = plt.cm.tab10.colors

for i, h in enumerate(folds):
    aucs = h["val_auc"]
    epochs = list(range(1, len(aucs) + 1))
    peak = max(aucs)
    ax.plot(epochs, aucs, color=palette[i],
            label=f"Fold {i+1}  (peak {peak:.4f})", linewidth=1.6)
    ax.axhline(peak, color=palette[i], linewidth=0.6, linestyle="--", alpha=0.4)

mean_peaks = np.mean([max(h["val_auc"]) for h in folds])
ax.axhline(mean_peaks, color="black", linewidth=1.2, linestyle=":",
           label=f"Mean peak AUC = {mean_peaks:.4f}")

ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Val AUC-ROC", fontsize=12)
ax.set_title("MG-HGNN: Val AUC per fold (OULAD, weighted BCE)", fontsize=13)
ax.legend(fontsize=10)
ax.set_ylim(0.4, 1.0)
ax.grid(axis="y", alpha=0.3)

out = pathlib.Path("results/auc_curves.png")
out.parent.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(out, dpi=150)
print(f"Saved → {out}")

# Per-fold table
print(f"\n{'Fold':>5}  {'Epochs':>6}  {'Peak AUC':>9}  {'Final AUC':>10}  {'Peak F1':>8}")
print("-" * 48)
for i, h in enumerate(folds):
    aucs = h["val_auc"]
    f1s  = h["val_f1"]
    print(f"  {i+1:>3}  {len(aucs):>6}  {max(aucs):>9.4f}  {aucs[-1]:>10.4f}  {max(f1s):>8.4f}")
print("-" * 48)
print(f"  Mean         {np.mean([max(h['val_auc']) for h in folds]):.4f}  "
      f"             {np.mean([max(h['val_f1'])  for h in folds]):.4f}")
