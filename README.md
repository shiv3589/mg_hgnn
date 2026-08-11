# MG-HGNN: Relation-Conditioned Modality Gating for Student Performance Prediction

## Overview
MG-HGNN is a Modality-Gated Heterogeneous Graph Neural Network that
conditions multi-modal feature fusion on the edge relation type during
message passing. For each relation type r, a dedicated softmax gate learns
which modality (structured, textual, behavioral) is most informative for
that specific relation.

**Submitted to ACM CIKM 2026** (under review)

## Requirements
- Python 3.9+
- PyTorch >= 2.2
- PyTorch Geometric >= 2.4

pip install -r mg_hgnn/requirements.txt

## Dataset Setup
Download and place at the following paths:

| Dataset | URL | Path |
|---|---|---|
| OULAD | https://analyse.kmi.open.ac.uk/open_dataset | mg_hgnn/data/raw/oulad/ |
| ASSISTments 2009 | https://sites.google.com/site/assistmentsdata/ | mg_hgnn/data/raw/assistments09/ |
| ASSISTments 2015 | https://sites.google.com/site/assistmentsdata/ | mg_hgnn/data/raw/assistments15/ |

## Reproducing Results

cd mg_hgnn

# Train on OULAD (5-fold CV)
python train_fast.py --dataset oulad

# Train on ASSISTments
python train_fast.py --dataset assistments09
python train_fast.py --dataset assistments15

# Run baselines
python baselines.py --dataset oulad

# Run ablation study (all 5 variants)
python run_ablation.py --dataset oulad

# Visualize gate weights
python evaluate.py --visualize-gates

## Project Structure

mg_hgnn/
├── models/
│   ├── encoders.py         Modality encoders (MLP, BERT, BiLSTM)
│   ├── gate.py             ModalityGate — core contribution
│   └── mg_hgnn.py          Full model + 4 ablation variants
├── data/
│   ├── oulad_loader.py     OULAD heterogeneous graph loader
│   ├── assistments_loader.py   ASSISTments 2009 + 2015 loader
│   └── moocc_loader.py     MOOC-Cube loader
├── train_fast.py           Optimised training loop
├── run_ablation.py         Ablation study
├── baselines.py            XGBoost, GAT, HGT, HAN baselines
├── evaluate.py             Metrics + gate weight visualization
├── config.py               All hyperparameters
└── paper/                  LaTeX source + compiled PDFs
