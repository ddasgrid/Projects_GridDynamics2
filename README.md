#  Visual Entailment Analyzer (SNLI-VE)

> A multimodal Two-Tower deep learning system that predicts the logical relationship — **Entailment**, **Neutral**, or **Contradiction** — between an image and a natural language hypothesis. Built and fine-tuned locally on Apple Silicon using ViT + BERT, achieving a peak validation accuracy of **73.73%**, surpassing the 70% target.

---

##  Demo Videos

https://drive.google.com/drive/folders/1_OX-CSacoElP15tuaaJexJU-LPzLRy-5?usp=sharing

#### System Limitations
- The text pipeline truncates inputs strictly at 128 tokens, permanently deleting any excess words before processing.
- The model struggles with out-of-distribution (OOD) data because its ViT and BERT backbones were fine-tuned specifically on SNLI-VE and Flickr30k datasets.
- BERT requires full bidirectional context, meaning single-word prompts lack logical structure and cause confused, random classifications.
- ViT was trained exclusively on real-world photographs, so abstract art, simple shapes, or solid color backgrounds break its patch-based logic.
- While BERT's sub-word tokenization handles minor misspellings, severe typos destroy the root word and prevent the model from finding visual alignments.

#### Best Practices for Optimal Results
- Always upload natural, real-world photographs featuring distinct subjects, strictly avoiding digital art, solid colors, or abstract geometric shapes.
- Keep your text hypotheses strictly under 128 words to prevent automatic truncation.
- Write complete, grammatically correct sentences that make a specific logical claim (e.g., "Two people are playing a sport" instead of just "Sports").
- Rely on the mathematically stable Base Model (Concatenation + Linear Classifier) to run the pipeline, as it successfully absorbs gradient shock unlike the fragile SOTA model.

---

##  Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset--data-pipeline)
- [Architecture](#architecture)
- [Experiments & Baselines](#experiments--baselines)
- [Training Strategy](#training-strategy--optimization)
- [Evaluation & Results](#evaluation--results)
- [Hard Negatives & Representation Collapse](#hard-negatives--sota-collapse)
- [Demo (Streamlit App)](#demo--streamlit-app)
- [Reproducing This Project](#reproducing-this-project)
- [Project Structure](#project-structure)
- [Future Scope](#future-scope)
- [Acknowledgements](#acknowledgements)

---

##  Problem Statement

**Visual Entailment** is a multimodal NLI (Natural Language Inference) task: given a **Flickr30k image** as a premise and a **text hypothesis**, the model must classify their logical relationship into one of three categories:

| Label | ID | Meaning |
|---|---|---|
| Entailment | 0 | The hypothesis is definitely true given the image |
| Neutral | 1 | The hypothesis may or may not be true |
| Contradiction | 2 | The hypothesis is definitely false given the image |

**Why this is hard:** It requires genuine cross-modal reasoning — the model must ground linguistic semantics into visual features, not just pattern-match within a single modality.

**Success Criterion:** Surpass **70% validation accuracy** on the full SNLI-VE dataset.

---

##  Dataset & Data Pipeline

### Source
- **Annotations:** [`HuggingFaceM4/SNLI-VE`](https://huggingface.co/datasets/HuggingFaceM4/SNLI-VE) (JSONL format)
- **Images:** Flickr30k image corpus (~31,000 unique images)
- **Final Training Set:** 529,527 valid, **perfectly balanced** rows across all 3 classes

### Pipeline Steps

```
SNLI-VE JSONL Annotations + Flickr30k Images
         │
         ▼
prepare_dataframe()   ← Drops NaNs, validates image paths, maps labels → {0,1,2}, saves CSV
         │
         ▼
SNLIVEDataset         ← ViTImageProcessor (RGB) + BertTokenizer (max_length=128)
         │
         ▼
DataLoader            ← batch_size=32 (optimized for Mac MPS Mixed Precision)
```

### Data Quality Measures

- **Path Validation:** `os.path.exists()` check on every image before training — zero missing-file crashes.
- **Label Cleaning:** Custom `SNLIVEDataset` explicitly drops corrupted/unlabeled rows.
- **Perfect Class Balance:** All three classes are equally represented, eliminating class-imbalance bias.
- **Tokenization:** Text truncated/padded to `max_length=128` via `BertTokenizer` to standardize sequence length.

### Hard Negative Generation (Phase 3)

To stress-test the final model, the full dataset was infused with programmatically generated **hard negatives** — subtle contradictory pairs designed to break logical shortcuts:

| Method | Description |
|---|---|
| **Antonym Augmentation** | `nlpaug` swaps key words with antonyms, directly flipping logical meaning |
| **Cross-Row Mashing** | Pairs an image with an entailing sentence from a completely unrelated row |
| **Targeted POS Replacement** | `nltk` identifies high-frequency nouns/verbs and replaces them via a custom semantic confusion dictionary |

---

##  Architecture

### Two-Tower Design

```
         Image Input                    Text Input
             │                              │
    ViTImageProcessor               BertTokenizer
             │                              │
    Vision Encoder (ViT)         Text Encoder (BERT)
  (google/vit-base-patch16-224)   (bert-base-uncased)
  [Progressively Unfrozen]        [Progressively Unfrozen]
             │                              │
             └──────────┬───────────────────┘
                        │
               Fusion Mechanism
         (Concat vs. Cross-Attention vs. Math Merges)
                        │
               Reasoning Engine (FFNN)
          Linear → LayerNorm → GELU → Dropout
                        │
               Classifier Head
          (Linear vs. Deep MLP vs. SwiGLU)
                        │
               3-Class Output
         [Entailment | Neutral | Contradiction]
```

### Backbone Encoders

| Encoder | Model | Output Dim |
|---|---|---|
| Vision | `google/vit-base-patch16-224` | 768 |
| Text | `bert-base-uncased` | 768 |

### Fusion Mechanisms Explored

| Fusion | Math | Result |
|---|---|---|
| **Concatenation** | `[v; t]` → 1536-dim | ✅ Best stability |
| **Cross-Attention** | Text as Query over Image Keys/Values | ✅ Best proxy score |
| **Element-wise Addition** | `v + t` | ❌ Noisy representations |
| **Element-wise Multiplication** | `v ⊙ t` | ❌ Destroys unaligned features |

### Classifier Heads Explored

| Head | Mechanism | Proxy Score |
|---|---|---|
| Linear | Single affine layer | 56.53% |
| Deep MLP | Stacked linear + activations | 56.76% |
| **SwiGLU** | Multiplicative gating: `x ⊙ σ(gate)` | **59.26%** |

---

##  Experiments & Baselines

Full hyperparameter and architecture search across the Base Model + 9 experiments. All proxy runs use 20% data with frozen backbones to isolate fusion/head capacity.

| Experiment | Dataset | Backbone | Fusion | Depth/Dim/Dropout | Head | Val Acc | Notes |
|---|---|---|---|---|---|---|---|
| Base Model | 100% | Fully Frozen | Concat | 2 / 512 / 0.1 | Linear | 58.00% | Bottlenecked by frozen encoders |
| Base Model | 100% | Top 2 Unfrozen | Concat | 2 / 512 / 0.1 | Linear | 70.69% | +12% jump from unfreezing |
| **Base Model** | **100%** | **Top 6 Unfrozen** | **Concat** | **2 / 512 / 0.1** | **Linear** | **73.73%** | 🏆 Champion model |
| Exp 2 | 20% | Frozen | Multiplication | 2 / 512 / 0.1 | Linear | 44.22% | Destroyed unaligned features |
| Exp 3 | 20% | Frozen | Addition | 2 / 512 / 0.1 | Linear | 48.32% | Noisy representations |
| Exp 4 | 20% | Frozen | Concat | 2 / 512 / 0.1 | Linear | 57.65% | Lossless information baseline |
| Exp 1(a) | 20% | Frozen | Cross-Attention | 4 / 512 / 0.1 | Linear | 56.58% | Deep network overfit |
| Exp 1(b) | 20% | Frozen | Cross-Attention | 1 / 512 / 0.1 | Linear | 58.00% | Shallow = better generalization |
| Exp 1(c) | 20% | Frozen | Cross-Attention | 1 / 256 / 0.1 | Linear | 57.75% | Narrow dim restricted capacity |
| Exp 1(d) | 20% | Frozen | Cross-Attention | 1 / 512 / 0.3 | Linear | 58.23% | High dropout killed co-adaptation |
| Exp 5 | 20% | Frozen | Cross-Attention | 1 / 512 / 0.3 | Linear | 56.53% | Too simple to decode attention |
| Exp 6 | 20% | Frozen | Cross-Attention | 1 / 512 / 0.3 | Deep MLP | 56.76% | Failed on frozen noise |
| Exp 7 | 20% | Frozen | Cross-Attention | 1 / 512 / 0.3 | SwiGLU | 59.26% | Gating excelled on static features |
| Exp 8 | 50% | Top 2 Unfrozen | Cross-Attention | 1 / 512 / 0.3 | SwiGLU | 67.21% | Scaled smoothly |
| Exp 9 | 100% + Hard Negs | Top 4 Unfrozen | Cross-Attention | 1 / 512 / 0.3 | SwiGLU | 33.37% |  NaN Collapse |

### Key Experimental Insights

**Fusion:**
- Multiplication and Addition fail because ViT and BERT encode into **completely unaligned latent spaces** — merging them mathematically creates corrupted representations.
- Concatenation (`768 + 768 = 1536`) is lossless — it glues feature matrices side-by-side with zero mathematical assumptions.
- Cross-Attention uses text as a dynamic **Query** to scan image features, suppressing irrelevant visual background based on linguistic context.

**FFNN Depth:**
- Depth 4 (56.58%) < Depth 1 (58.00%) — attaching a deep network to static frozen features causes **over-parameterization** (memorization, not learning).
- `dim=512` acts as an "Information Bottleneck": wide enough for logical reasoning, narrow enough to discard background noise. `dim=256` was too restrictive.
- `dropout=0.3` outperformed `dropout=0.1` by destroying co-adaptation shortcuts.

**Classifier Head:**
- SwiGLU's multiplicative gate (`multiply by 0` on noise) decisively outperformed both Linear and Deep MLP under frozen-backbone conditions.

---

##  Training Strategy & Optimization

### Hardware & Precision

| Feature | Implementation |
|---|---|
| **Device Routing** | Auto-detects Apple MPS → CUDA → CPU |
| **Mixed Precision (AMP)** | `torch.autocast` wraps forward passes |
| **Gradient Scaling** | `GradScaler` for CUDA; intentionally bypassed for MPS (not required) |
| **Inference Optimization** | `torch.no_grad()` during validation |

### Optimizer & Scheduler

| Component | Config | Rationale |
|---|---|---|
| **Optimizer** | AdamW, `weight_decay=0.01` | Decoupled weight decay for stable regularization |
| **Backbone LR** | `1e-5` | Conservative — prevents gradient shock on pre-trained weights |
| **Head LR** | `1e-4` | Aggressive — untrained heads need faster convergence |
| **Scheduler** | Cosine Warmup (10% warmup → cosine decay) | Prevents early gradient spikes; precise convergence |

### Training Flow Control

```
Proxy Training (20% data, frozen backbones)
   → Architecture search (fusion + head + FFNN params)
   → Best config: Cross-Attention + SwiGLU

Progressive Unfreezing
   Phase 1: 0 layers unfrozen   → Proxy baseline
   Phase 2: Top 2 layers        → 70.69% (Base) / 67.21% (SOTA)
   Phase 3: Top 6 layers        → 73.73% (Base) ✅ / NaN collapse (SOTA) 

Early Stopping
   → patience=3 (halts if val loss doesn't improve for 3 epochs)
   → Prevents overfitting at every phase
```

**Why Progressive Unfreezing?**
Sequentially unfreezing transformer layers builds a mathematical "wall" that protects pre-trained weights from catastrophic forgetting, while allowing the model to safely adapt to the target domain.

---

##  Evaluation & Results

### Metrics

| Metric | Description |
|---|---|
| **Validation Accuracy** | % correctly classified across Entailment / Neutral / Contradiction on unseen eval set |
| **Cross-Entropy Loss** | Measures predictive confidence; triggers Early Stopping if it plateaus for 3 epochs |

### Final Results Summary

| Architecture | Dataset Scale | Final Accuracy | Goal (>70%) | Stability |
|---|---|---|---|---|
| 🏆 **Base (Concat + Linear, Top 6 Unfrozen)** | 100% | **73.73%** | ✅ Exceeded | Highly stable — absorbed massive gradients |
| Base (Concat + Linear, Top 2 Unfrozen) | 100% | 70.69% | ✅ Met | Stable intermediate step |
| SOTA (Attention + SwiGLU, Top 2 Unfrozen) | 50% | 67.21% | ❌ Missed | Stable on limited data |
| SOTA Proxy (Attention + SwiGLU, Frozen) | 20% | 59.26% | ❌ Missed | Best under controlled proxy |
| SOTA Collapse (Attention + SwiGLU, Top 4 Unfrozen) | 100% + Hard Negs | 33.37% | ❌ Failed | Total NaN network death |

---

##  Hard Negatives & SOTA Collapse

### What Happened in Experiment 9

The SOTA architecture (Cross-Attention + SwiGLU) was stress-tested with 4 unfrozen transformer layers + full hard-negative-infused dataset. In **Epoch 2**, the model suffered **complete representation collapse**:

- Validation loss → `NaN`
- Accuracy → `33.37%` (pure random chance for 3 classes)

### Root Cause: A Perfect Storm

```
Hard Negatives → Massive error signals (high loss)
       +
4 Unfrozen Layers → Internal Covariate Shift (ViT/BERT outputs shift violently)
       +
Cross-Attention Softmax + SwiGLU gates → Fragile math that breaks under large shifting inputs
       =
Gradient Explosion → NaN poisoning → Total network death
```

### Why the Base Model Survived

Under even heavier stress (6 unfrozen layers), the Base Model (Concatenation + Linear) peaked at **73.73%** because:
- **Concatenation** = basic matrix stitching. No Softmax, no gates. Mathematically indestructible.
- **Linear Classifier** = simple matrix multiplication. Absorbs gradient shocks without exploding.

> **Core Lesson:** Architectural simplicity is not a weakness — it is a stability superpower when scaling to full data with deep unfreezing.

---

##  Demo — Streamlit App

An interactive **Streamlit** UI (`app2.py`) allows inference on new image-text pairs without any code.

### Running the App

```bash
# 1. Clone the repository
git clone <(https://github.com/ddasgrid/Projects_GridDynamics2)>
cd visual-entailment-analyzer

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download model weights (hosted on Google Drive — exceeds GitHub's 100MB limit)
 Link to Drive where model weights are stored: https://drive.google.com/drive/folders/1G8HXVayxYRuoW8L-69jry8BzTed1pGZP?usp=sharing
 Place all .pth files in the same directory as app2.py

# 4. Launch the demo
streamlit run app2.py
```

### App Features
- **Upload any image** (Flickr30k or custom)
- **Type any hypothesis** as free text
- Outputs predicted label (**Entailment / Neutral / Contradiction**) with confidence scores
- Runs locally using the champion Base Model weights

---

##  Reproducing This Project

### Requirements

**Hardware:** Apple Silicon Mac (M1/M2/M3) recommended for MPS acceleration. Minimum **16GB RAM** required for unfreezing deep transformer layers.

**Software:**

```
torch>=2.0.0
transformers>=4.30.0
nltk>=3.8.1
nlpaug>=1.1.11
pandas>=1.5.0
numpy>=1.23.0
Pillow>=9.0.0
tqdm>=4.65.0
streamlit>=1.25.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

### Step-by-Step Reproduction

```
Step 1 → Run base_model.ipynb          (Data curation + baseline frozen training)
Step 2 → Run Experiment_1.ipynb        (Attention fusion hyperparameter tuning)
Step 3 → Run Experiments 2–7.ipynb     (Fusion & classifier head search on 20% proxy)
Step 4 → Run Experiment_8.ipynb        (50% data scale-up with Top 2 unfrozen)
Step 5 → Run Experiment_9.ipynb        (100% data + hard negatives — observe NaN collapse)
Step 6 → Return to base_model.ipynb    (Progressive unfreezing → Top 2 → Top 6, champion run)
Step 7 → streamlit run app2.py         (Launch interactive demo)
```

---

##  Project Structure

```
.
├── app.py                  ← Streamlit demo UI
├── base_model.ipynb        ← Baseline establishment & data curation
├── Experiment_1.ipynb      ← Attention Fusion hyperparameter tuning
├── Experiment_2.ipynb      ← Element-wise Multiplication test
├── Experiment_3.ipynb      ← Element-wise Addition test
├── Experiment_4.ipynb      ← Concatenation fusion test
├── Experiment_5.ipynb      ← Standard Linear Head test
├── Experiment_6.ipynb      ← Deep MLP Classifier test
├── Experiment_7.ipynb      ← SwiGLU Classifier test
├── Experiment_8.ipynb      ← 50% Data Scale-up
├── Experiment_9.ipynb      ← 100% Data Scale-up & NaN Collapse
├── read.md                 ← Experiment summaries
├── report_2.pdf            ← Theoretical logic and mathematical breakdown
└── requirements.txt        ← Python dependencies
```

> **Note:** Model weights (`.pth` files) are hosted on Google Drive due to GitHub's 100MB file size limit.

---

##  Future Scope

To push accuracy from **73.73% → 80%+**, two parallel improvement tracks are planned:

### 1. Maximize the Base Model
The champion model is currently under-trained at only **1 epoch per unfreezing phase**. Extending to **3–5 epochs per phase** with Cosine LR Decay + Early Stopping will allow the model to safely reach a deeper global minimum without forgetting.

### 2. Stabilize the SOTA Architecture
To prevent the NaN collapse observed in Experiment 9, the Cross-Attention + SwiGLU model requires:

| Fix | Purpose |
|---|---|
| **Gradient Clipping** (`max_norm=1.0`) | Caps exploding gradient magnitudes before they cause NaN |
| **Curriculum Learning** | Delay hard negatives until the model stabilizes in later epochs |
| **Lower Backbone LR** (`1e-6` vs `1e-5`) | Slower unfreezing to reduce covariate shift |

### 3. Transition to LoRA
Instead of fully unfreezing millions of backbone parameters (which causes instability + high VRAM usage), future versions will use **LoRA (Low-Rank Adaptation)**:
- Freeze original ViT/BERT weights entirely
- Inject tiny trainable adapter matrices (`rank=8` to `rank=32`)
- Eliminates gradient explosion while enabling complex fusion head scaling
- Reduces memory footprint by ~60–70% compared to full fine-tuning

---

##  Acknowledgements

| Resource | Source |
|---|---|
| **Dataset** | [HuggingFaceM4/SNLI-VE](https://huggingface.co/datasets/HuggingFaceM4/SNLI-VE) |
| **Image Corpus** | [Flickr30k](http://shannon.cs.illinois.edu/DenotationGraph/) |
| **Vision Encoder** | [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224) |
| **Text Encoder** | [bert-base-uncased](https://huggingface.co/bert-base-uncased) |
| **Development Assistance** | LLMs used as coding assistants during development and documentation |

**License:** This project is intended for educational and research purposes.

---

<div align="center">

**Built with ViT + BERT | Fine-tuned on Apple Silicon | Peak Accuracy: 73.73%**

</div>
