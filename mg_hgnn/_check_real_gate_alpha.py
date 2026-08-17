"""
Cross-check: does model.gate.get_gate_weights() (synthetic probe) match
the actual alpha the model produces on REAL encoded student embeddings?
"""
import torch
from config import Config
from models.mg_hgnn import MG_HGNN
from data.cache_embeddings import load_bert_cache
from data.oulad_loader import OULADLoader

cfg = Config()
model = MG_HGNN(cfg)
ckpt = torch.load('checkpoints/fold_4_fast_best.pt', map_location='cpu', weights_only=True)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

print("Loading OULAD + bert cache for real embeddings...")
loader = OULADLoader(cfg)
data, meta = loader.load()
bert_cache = load_bert_cache('data/bert_cache_oulad.pt')

with torch.no_grad():
    h_s = model.struct_enc(data["student"].x_struct)
    h_t = bert_cache["student"]
    h_b = model.behav_enc(data["student"].x_behav)

    print()
    print(f"h_s stats: mean={h_s.mean():.4f} std={h_s.std():.4f}")
    print(f"h_t stats: mean={h_t.mean():.4f} std={h_t.std():.4f}")
    print(f"h_b stats: mean={h_b.mean():.4f} std={h_b.std():.4f}")

    print()
    print("=== REAL-DATA gate alpha (mean +/- std over all 28,785 students) ===")
    for r in cfg.edge_types:
        _, alpha = model.gate(r, h_s, h_t, h_b)   # (N, 3)
        mean_a = alpha.mean(dim=0)
        std_a  = alpha.std(dim=0)
        mean_r = [round(v, 3) for v in mean_a.tolist()]
        std_r  = [round(v, 3) for v in std_a.tolist()]
        print(f"  {r:<25}: mean={mean_r}  std={std_r}")

    print()
    print("=== PROBE-based gate alpha (what get_gate_weights() reports) ===")
    for r in cfg.edge_types:
        alpha = model.gate.get_gate_weights(r)
        print(f"  {r:<25}: {[round(v, 3) for v in alpha.tolist()]}")
