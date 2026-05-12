#!/usr/bin/env bash
# ================================================================
#  MG-HGNN Project Setup Script
#  Run this entire file as a single prompt in Claude Code:
#  "Run setup_mg_hgnn.sh to scaffold the project and write the
#   LaTeX paper with reduced abstract into the paper/ folder."
# ================================================================

set -e

PROJECT="mg_hgnn"

# ----------------------------------------------------------------
# 1. Create full folder structure
# ----------------------------------------------------------------
mkdir -p $PROJECT/paper
mkdir -p $PROJECT/data
mkdir -p $PROJECT/models
mkdir -p $PROJECT/results
mkdir -p $PROJECT/checkpoints

touch $PROJECT/data/__init__.py
touch $PROJECT/data/oulad_loader.py
touch $PROJECT/data/moocc_loader.py
touch $PROJECT/models/__init__.py
touch $PROJECT/models/encoders.py
touch $PROJECT/models/gate.py
touch $PROJECT/models/mg_hgnn.py
touch $PROJECT/train.py
touch $PROJECT/evaluate.py
touch $PROJECT/baselines.py
touch $PROJECT/config.py

# ----------------------------------------------------------------
# 2. requirements.txt
# ----------------------------------------------------------------
cat > $PROJECT/requirements.txt << 'EOF'
torch>=2.2.0
torch_geometric>=2.4.0
transformers>=4.38.0
scikit-learn>=1.4.0
pandas>=2.1.0
numpy>=1.26.0
tqdm>=4.66.0
matplotlib>=3.8.0
xgboost>=2.0.0
pytest>=8.0.0
EOF

# ----------------------------------------------------------------
# 3. config.py — central hyperparameter dataclass
# ----------------------------------------------------------------
cat > $PROJECT/config.py << 'EOF'
from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    # --- Graph ---
    node_types: List[str] = field(
        default_factory=lambda: ["student", "course", "instructor", "resource"]
    )
    edge_types: List[str] = field(
        default_factory=lambda: [
            "enrolled_in", "collaborated_with", "accessed", "submitted_to"
        ]
    )

    # --- Encoders ---
    structured_input_dim: int = 64
    text_model_name: str = "bert-base-uncased"
    behavioral_input_dim: int = 32
    behavioral_seq_len: int = 50
    embed_dim: int = 128

    # --- HGNN ---
    num_hgnn_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.3

    # --- Training ---
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 200
    patience: int = 20
    batch_size: int = 256

    # --- Multi-task loss weights ---
    lambda_grade: float = 0.4
    lambda_dropout: float = 0.4
    lambda_engagement: float = 0.2

    # --- Paths ---
    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results"
EOF

echo "[OK] Folder structure and config created."

# ----------------------------------------------------------------
# 4. Write the LaTeX paper with REDUCED abstract
# ----------------------------------------------------------------
cat > $PROJECT/paper/mg_hgnn_paper.tex << 'LATEX'
% ============================================================
%  MG-HGNN: Modality-Gated Heterogeneous Graph Neural Networks
%  for Student Academic Performance Prediction
%  Conference Paper -- ACM SIGCONF Format
% ============================================================

\documentclass[sigconf]{acmart}

% ---- Packages ----
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{algorithm2e}
\usepackage{tikz}
\usepackage{lipsum}

% ---- ACM Metadata ----
\acmConference[KDD '26]{ACM SIGKDD Conference on Knowledge Discovery
              and Data Mining}{August 2026}{Barcelona, Spain}
\acmDOI{}
\acmISBN{}

% ============================================================
\begin{document}

% ---- Title ----
\title{MG-HGNN: Modality-Gated Heterogeneous Graph Neural Networks\\
       for Student Academic Performance Prediction}

% ---- Authors ----
\author{[Author Name]}
\affiliation{%
  \institution{[Your University / Institution]}
  \city{[City]}
  \country{[Country]}
}
\email{[email@university.edu]}

% ---- CCS Concepts ----
\begin{CCSXML}
<ccs2012>
  <concept>
    <concept_id>10010147.10010257.10010293.10010294</concept_id>
    <concept_desc>Computing methodologies~Neural networks</concept_desc>
    <concept_significance>500</concept_significance>
  </concept>
  <concept>
    <concept_id>10010147.10010178</concept_id>
    <concept_desc>Computing methodologies~Artificial intelligence</concept_desc>
    <concept_significance>300</concept_significance>
  </concept>
</ccs2012>
\end{CCSXML}

\ccsdesc[500]{Computing methodologies~Neural networks}
\ccsdesc[300]{Computing methodologies~Artificial intelligence}

\keywords{heterogeneous graph neural networks, student performance
          prediction, multi-modal learning, modality gating,
          educational data mining, learning analytics}

% ============================================================
%  ABSTRACT  (~150 words -- concise version)
% ============================================================
\begin{abstract}
Predicting student academic performance requires integrating structured
records, textual content, and behavioral logs --- modalities that carry
different signals depending on the relational context in which they are
used. Existing heterogeneous graph neural networks (HGNNs) collapse these
modalities into a single node embedding and apply uniform message passing
regardless of edge type, discarding relation-specific modal relevance.
We propose \textbf{MG-HGNN}, a \emph{Modality-Gated HGNN} in which each
node maintains separate modality embeddings and a learnable, per-edge-type
softmax gate $\boldsymbol{\alpha}^{(r)}$ controls how much each modality
contributes to message passing along relation $r$. Evaluated on MOOC-Cube
and OULAD across grade prediction, dropout risk, and engagement
classification, MG-HGNN outperforms HAN, HGT, and R-GCN baselines.
Ablation and gate-weight analyses confirm that edge-conditioned modal
gating is the key driver of performance gains.
\end{abstract}

\maketitle

% ============================================================
%  1. INTRODUCTION
% ============================================================
\section{Introduction}
\label{sec:intro}
\lipsum[1-2]
% TODO: Motivation, problem statement, and listed contributions.

% ============================================================
%  2. RELATED WORK
% ============================================================
\section{Related Work}
\label{sec:related}

The prediction of student academic performance has evolved substantially,
progressing from classical machine learning on independent per-student
feature vectors toward relational deep learning methods that explicitly
encode the complex network structure of educational ecosystems.

\subsection{Graph Neural Networks for Student Performance Prediction}

Early graph-based approaches showed that modelling student similarity as
an explicit graph yields systematic gains over flat-feature baselines.
\citet{li2022studygnn} proposed Study-GNN, a multi-topology GNN (MTGNN)
that constructs multiple student similarity graphs using distinct distance
metrics and fuses them via attention, casting performance prediction as
semi-supervised node classification. Study-GNN outperformed SVMs, logistic
regression, shallow neural networks, and vanilla GCN, validating relational
modelling for education.

\citet{huang2024dual} introduced APP-DGNN, a dual graph neural network
that constructs two complementary graphs: one from interaction logs
capturing engagement dynamics, and one from student attributes such as
demographics and prior grades. Fusing local and global representations,
APP-DGNN achieved approximately 84\% accuracy on pass/fail and 90\% on
pass/withdraw tasks on a public MOOC dataset. This dual-graph paradigm
establishes a key principle: different information sources benefit from
distinct graph topologies rather than a single aggregated representation.

\subsection{Temporal and Dynamic Graph Models}

Static graph methods cannot capture behavioral evolution across time.
\citet{huang2024temporal} addressed this with APP-TGN, a temporal graph
network encoding student activity logs as a dynamic graph. Combining
low-pass and high-pass spectral filters with multi-head attention, APP-TGN
significantly improves both overall accuracy and early at-risk detection in
MOOC settings. \citet{zhang2025gnntinet} embedded GNN layers inside a
GNN-Transformer-InceptionNet hybrid for multi-label performance prediction,
reporting up to 98.5\% accuracy on a single-institution dataset.

Despite these advances, long-range temporal dependencies remain
under-exploited in several dual and multi-graph designs
\citep{huang2024dual, huang2024temporal, balachandar2024mspp}, motivating
continued work on temporally expressive architectures. MG-HGNN addresses
a complementary and previously unexplored axis: \emph{modal selectivity}
during heterogeneous message passing.

\subsection{Hypergraph and Higher-Order Models}

\citet{yang2024hypergraph} proposed FD-HGNN, a framelet-based dual
hypergraph GNN using low- and high-pass hypergraphs to capture higher-order,
multi-way student relationships. Evaluated on four datasets, FD-HGNN
outperforms both traditional ML and standard GNNs. While this line of work
captures structural complexity beyond pairwise edges, hypergraph
constructions do not address the problem of multi-modal feature
heterogeneity \emph{within} each node --- the gap targeted by MG-HGNN.

\subsection{Multi-source and Heterogeneous Data Fusion}

Several works highlight the need to handle sparse, multi-source data ---
behavioral logs, demographics, emotional indicators, and prior grades ---
more effectively than flat concatenation allows
\citep{huang2024dual, zhang2025gnntinet, huang2024temporal,
li2022studygnn, yang2024hypergraph, balachandar2024mspp}.
\citet{balachandar2024mspp} proposed MSPP, which applies advanced GNN
layers on heterogeneous academic data, achieving approximately 76\%
accuracy on complex multi-class tasks. Across the literature, reported
accuracy spans from $\sim$76\% on complex multi-class settings
\citep{balachandar2024mspp} to 84--90\% on binary outcomes
\citep{huang2024dual} and up to 98--99\% on richer, single-institution
benchmarks \citep{zhang2025gnttinet}. Shared limitations include high
computational cost on large cohorts and limited cross-institutional
generalizability \citep{huang2024dual, li2022studygnn, zhang2025gnttinet,
huang2024temporal, yang2024hypergraph, balachandar2024mspp}.

\subsection{Gap Addressed by This Work}

Despite the breadth of existing approaches, one assumption has remained
unexamined: that all edge relation types should draw on the same mixture of
input modalities when aggregating neighborhood information. No existing
HGNN framework conditions its feature fusion strategy on the edge type
being traversed. MG-HGNN fills this gap by maintaining separate per-modality
embeddings at each node and learning a distinct, softmax-normalized modality
gate $\boldsymbol{\alpha}^{(r)}$ for each relation $r \in \mathcal{R}$,
allowing the model to discover which information sources are most relevant
for each relational context without manual specification.

% ============================================================
%  3. PROBLEM DEFINITION
% ============================================================
\section{Problem Definition}
\label{sec:problem}
\lipsum[1]
% TODO: Formal graph G = (V, E, T_v, T_e), node/edge types, task heads.

% ============================================================
%  4. METHODOLOGY
% ============================================================
\section{Methodology: MG-HGNN}
\label{sec:method}
\lipsum[1-2]
% TODO: Modality encoders, gate formulation, HGT backbone, prediction heads.

% ============================================================
%  5. EXPERIMENTS
% ============================================================
\section{Experiments}
\label{sec:experiments}
\lipsum[1]
% TODO: Datasets, baselines (HAN/HGT/R-GCN/Study-GNN), RQ1-RQ4 tables.

% ============================================================
%  6. CONCLUSION
% ============================================================
\section{Conclusion}
\label{sec:conclusion}
\lipsum[1]
% TODO: Summary of contributions, limitations, future work.

% ============================================================
%  REFERENCES
% ============================================================
\bibliographystyle{ACM-Reference-Format}

\begin{thebibliography}{9}

\bibitem{balachandar2024mspp}
Balachandar, V., \& Venkatesh, K. (2024).
\newblock A multi-dimensional student performance prediction model ({MSPP}):
  An advanced framework for accurate academic classification and analysis.
\newblock \emph{MethodsX}, 14.
\newblock \url{https://doi.org/10.1016/j.mex.2024.103148}

\bibitem{huang2024dual}
Huang, Q., \& Zeng, Y. (2024).
\newblock Improving academic performance predictions with dual graph neural
  networks.
\newblock \emph{Complex \& Intelligent Systems}, 10, 3557--3575.
\newblock \url{https://doi.org/10.1007/s40747-024-01344-z}

\bibitem{huang2024temporal}
Huang, Q., \& Chen, J. (2024).
\newblock Enhancing academic performance prediction with temporal graph
  networks for massive open online courses.
\newblock \emph{Journal of Big Data}, 11, 1--26.
\newblock \url{https://doi.org/10.1186/s40537-024-00918-5}

\bibitem{li2022studygnn}
Li, M., Wang, X., Wang, Y., Chen, Y., \& Chen, Y. (2022).
\newblock {Study-GNN}: A novel pipeline for student performance prediction
  based on multi-topology graph neural networks.
\newblock \emph{Sustainability}.
\newblock \url{https://doi.org/10.3390/su14137965}

\bibitem{li2022campus}
Li, X., Zhang, Y., Cheng, H., Li, M., \& Yin, B. (2022).
\newblock Student achievement prediction using deep neural network from
  multi-source campus data.
\newblock \emph{Complex \& Intelligent Systems}, 8, 5143--5156.
\newblock \url{https://doi.org/10.1007/s40747-022-00731-8}

\bibitem{yang2024hypergraph}
Yang, Y., Shi, J., Li, M., \& Fujita, H. (2024).
\newblock Framelet-based dual hypergraph neural networks for student
  performance prediction.
\newblock \emph{International Journal of Machine Learning and Cybernetics},
  15, 3863--3877.
\newblock \url{https://doi.org/10.1007/s13042-024-02124-4}

\bibitem{zhang2025gnttinet}
Zhang, X., Zhang, Y., Chen, A., Yu, M., \& Zhang, L. (2025).
\newblock Optimizing multi-label student performance prediction with
  {GNN-TINet}: A contextual multidimensional deep learning framework.
\newblock \emph{PLOS ONE}, 20.
\newblock \url{https://doi.org/10.1371/journal.pone.0314823}

\end{thebibliography}

\end{document}
LATEX

echo "[OK] LaTeX paper written to $PROJECT/paper/mg_hgnn_paper.tex"

# ----------------------------------------------------------------
# 5. Verify structure
# ----------------------------------------------------------------
echo ""
echo "Project structure:"
find $PROJECT -type f | sort

echo ""
echo "================================================================"
echo " Setup complete. Next steps:"
echo "  1. cd $PROJECT"
echo "  2. pip install -r requirements.txt --break-system-packages"
echo "  3. Run: pdflatex paper/mg_hgnn_paper.tex  (needs TeX Live)"
echo "  4. Use Prompts 4-9 from the Claude Code guide to fill models/"
echo "================================================================"
