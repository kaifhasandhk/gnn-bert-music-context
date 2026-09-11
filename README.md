# GNN-Based BERT for Understanding Context from Music

**Course:** CSE425 / EEE474 / CSE715 Neural Networks  
**Dataset:** MagnaTagATune (3,000 clips, top-50 tags)  
**Architecture:** Multimodal Cross-Attention (DistilBERT + GraphSAGE GNN)

---

##  Overview
This repository implements a multimodal neural network architecture that combines structural audio graph representations (GraphSAGE on 3-second audio segment graphs) with natural language tag context (DistilBERT) to perform multi-label music tagging and context understanding.

---

##  Performance & Ablation Results

| Model / Architecture | Macro-F1 | Micro-F1 | Key Insight |
| :--- | :---: | :---: | :--- |
| **B1: Majority Class Baseline** | 0.021 | 0.161 | Naïve tag prior baseline |
| **B2: Mel-Spectrogram CNN** | 0.229 | 0.354 | Standard 2D spectral audio model |
| **Task 1: BERT-Only (Text Context)** | 0.190 | 0.336 | Text sequence representation |
| **Task 2: GraphSAGE GNN (Audio Graph)** | 0.216 | 0.413 | Temporal segment graph learning |
| **Task 3: Cross-Attention GNN-BERT Fusion** | **0.292** | **0.472** | **Best overall multimodal model** |

---

##  Latent Space Visualization
Below is the t-SNE projection of the cross-attention fused embeddings $z$, highlighting clear topological separation between contrasting genres (`classical` vs. `techno`):

![t-SNE Plot](results/plots/task3_tsne_fused.png)

---

##  Repository Directory Structure
```text
gnn-bert-music-context/
├── README.md
├── requirements.txt
├── config.yaml
├── data/
│   ├── raw/                      # FMA, MagnaTagATune audio downloads
│   ├── processed/                # Saved PyG .pt graphs & feature caches
│   └── splits/                   # Train / Val / Test metadata JSONs
├── notebooks/
│   ├── eda.ipynb                 # Tag distribution & dataset statistics
│   └── demo_context.ipynb        # End-to-end Kaggle training & ablation pipeline
├── src/
│   ├── audio_features.py         # Vectorized Mel + Chroma feature extraction (104-dim)
│   ├── graph_builder.py          # Dynamic audio segment graph builder
│   ├── bert_encoder.py           # Task 1: DistilBERT tag classifier
│   ├── gnn_model.py              # Task 2: 2-Layer GraphSAGE GNN
│   ├── fusion_model.py           # Task 3: Cross-Attention GNN-BERT fusion
│   ├── contrastive.py            # Task 4: InfoNCE cross-modal loss (Extension)
│   ├── train.py                  # Standalone execution pipeline
│   └── evaluate.py               # Threshold sweeping & Macro/Micro F1 metrics
├── results/
│   ├── metrics.json              # Programmatic ablation comparison metrics
│   ├── plots/                    # Exported t-SNE & loss curves
│   └── retrieval_examples/
│       ├── case_studies.json     # Quantitative test clip JSON logs
│       └── qualitative_samples.txt # Human-readable prediction comparisons
└── report/
    └── final_report.pdf          # Final project paper & documentation
```
##  Quickstart
1. Open `notebooks/demo_context.ipynb` in Google Colab or Kaggle.
2. Execute all cells sequentially to build segment graphs, train model variants, and reproduce evaluation metrics and case studies.
