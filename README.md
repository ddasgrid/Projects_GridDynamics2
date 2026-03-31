#  Visual Entailment Analyzer (SNLI-VE)

> A multimodal Two-Tower deep learning system that predicts the logical relationship - **Entailment**, **Neutral**, or **Contradiction** - between an image and a natural language hypothesis. Built and fine-tuned locally on Apple Silicon using ViT + BERT, with a historical peak validation accuracy of **73.73%** and a deployment baseline chosen for stronger custom-data behavior.

---

##  Demo Videos

https://drive.google.com/drive/folders/1_OX-CSacoElP15tuaaJexJU-LPzLRy-5?usp=sharing

##  Deployed App

https://huggingface.co/spaces/DebDas02/Visual_Entailment

#### System Limitations
- The text pipeline truncates inputs strictly at 128 tokens, permanently deleting any excess words before processing.
- The model struggles with out-of-distribution (OOD) data because its ViT and BERT backbones were fine-tuned specifically on SNLI-VE and Flickr30k datasets.
- BERT requires full bidirectional context, meaning single-word prompts lack logical structure and cause confused, random classifications.
- ViT was trained exclusively on real-world photographs, so abstract art, simple shapes, or solid color backgrounds break its patch-based logic.
- While BERT sub-word tokenization handles minor misspellings, severe typos destroy the root word and prevent the model from finding visual alignments.

#### Best Practices for Optimal Results
- Always upload natural, real-world photographs featuring distinct subjects, strictly avoiding digital art, solid colors, or abstract geometric shapes.
- Keep your text hypotheses under 128 words to prevent automatic truncation.
- Write complete, grammatically correct sentences that make a specific logical claim (e.g., "Two people are playing a sport" instead of just "Sports").
- For app workflows in this repo, use the selected deployment baseline checkpoint (`final_sota_visual_entailment3.pth`) for best custom-data behavior.

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

**Why this is hard:** It requires genuine cross-modal reasoning - the model must ground linguistic semantics into visual features, not just pattern-match within a single modality.

**Success Criterion:** Surpass **70% validation accuracy** on the full SNLI-VE dataset.

---

##  Dataset & Data Pipeline

### Source
- **Annotations:** [`HuggingFaceM4/SNLI-VE`](https://huggingface.co/datasets/HuggingFaceM4/SNLI-VE) (JSONL format)
- **Images:** Flickr30k image corpus (~31,000 unique images)
- **Final Training Set:** 529,527 valid, balanced rows across all 3 classes

### Pipeline Steps

```text
SNLI-VE JSONL Annotations + Flickr30k Images
         |
         v
prepare_dataframe()   <- Drops NaNs, validates image paths, maps labels -> {0,1,2}, saves CSV
         |
         v
SNLIVEDataset         <- ViTImageProcessor (RGB) + BertTokenizer (max_length=128)
         |
         v
DataLoader            <- batch_size=32 (optimized for Mac MPS Mixed Precision)
```

### Data Quality Measures

- **Path Validation:** `os.path.exists()` check on every image before training.
- **Label Cleaning:** Custom `SNLIVEDataset` drops corrupted/unlabeled rows.
- **Class Balance:** All three classes are equally represented.
- **Tokenization:** Text truncated/padded to `max_length=128` via `BertTokenizer`.

### Hard Negative Generation (Phase 3)

To stress-test the final model, the dataset was infused with programmatically generated hard negatives.

| Method | Description |
|---|---|
| **Antonym Augmentation** | `nlpaug` swaps key words with antonyms, flipping logical meaning |
| **Cross-Row Mashing** | Pairs an image with an entailing sentence from an unrelated row |
| **Targeted POS Replacement** | `nltk` identifies high-frequency nouns/verbs and replaces them via a confusion dictionary |

---

##  Architecture

### Two-Tower Design

```text
         Image Input                    Text Input
             |                              |
    ViTImageProcessor               BertTokenizer
             |                              |
    Vision Encoder (ViT)         Text Encoder (BERT)
  (google/vit-base-patch16-224)   (bert-base-uncased)
             |                              |
             \-----------+------------------/
                         |
                Fusion Mechanism
       (Concat / Cross-Attention / Math Merges)
                         |
                Reasoning Engine
        Linear -> LayerNorm -> GELU -> Dropout
                         |
                Classifier Head
               (Linear / SwiGLU)
                         |
                3-Class Output
```

### Backbone Encoders

| Encoder | Model | Output Dim |
|---|---|---|
| Vision | `google/vit-base-patch16-224` | 768 |
| Text | `bert-base-uncased` | 768 |

### Fusion Mechanisms Explored

| Fusion | Math | Result |
|---|---|---|
| **Concatenation** | `[v; t]` -> 1536-dim | Best stability |
| **Cross-Attention** | Text as Query over Image Keys/Values | Best advanced reasoning |
| **Element-wise Addition** | `v + t` | Noisy representations |
| **Element-wise Multiplication** | `v * t` | Destroys unaligned features |

### Classifier Heads Explored

| Head | Mechanism | Proxy Score |
|---|---|---|
| Linear | Single affine layer | 56.53% |
| Deep MLP | Stacked linear + activations | 56.76% |
| **SwiGLU** | Multiplicative gating | **59.26%** |

---

##  Experiments & Baselines

| Experiment | Dataset | Backbone | Fusion | Depth/Dim/Dropout | Head | Val Acc | Notes |
|---|---|---|---|---|---|---|---|
| Base Model | 100% | Fully Frozen | Concat | 2 / 512 / 0.1 | Linear | 58.00% | Bottlenecked by frozen encoders |
| Base Model | 100% | Top 2 Unfrozen | Concat | 2 / 512 / 0.1 | Linear | 70.69% | Strong jump from unfreezing |
| **Base Model** | **100%** | **Top 6 Unfrozen** | **Concat** | **2 / 512 / 0.1** | **Linear** | **73.73%** | Historical best SNLI-VE val |
| Exp 8 | 50% | Top 2 Unfrozen | Cross-Attention | 1 / 512 / 0.3 | SwiGLU | 67.21% | Stable advanced model |
| Exp 9 | 100% + Hard Negs | Top 4 Unfrozen | Cross-Attention | 1 / 512 / 0.3 | SwiGLU | 33.37% | NaN collapse |

### Key Experimental Insights

- Concatenation is highly stable under heavy gradient stress.
- Cross-attention gives better explainability/control but can be fragile under aggressive unfreezing + hard negatives.
- Progressive unfreezing remains critical for stability.

---

##  Training Strategy & Optimization

### Hardware & Precision

| Feature | Implementation |
|---|---|
| **Device Routing** | Auto-detects Apple MPS -> CUDA -> CPU |
| **Mixed Precision (AMP)** | `torch.autocast` wraps forward passes |
| **Gradient Scaling** | `GradScaler` for CUDA; bypassed on MPS |
| **Inference Optimization** | `torch.no_grad()` during validation |

### Optimizer & Scheduler

| Component | Config | Rationale |
|---|---|---|
| **Optimizer** | AdamW, `weight_decay=0.01` | Stable regularization |
| **Backbone LR** | `1e-5` | Prevents gradient shock |
| **Head LR** | `1e-4` | Faster head convergence |
| **Scheduler** | Cosine warmup + decay | Stabilizes early steps |

### Training Flow Control

```text
Proxy Training -> Architecture search -> Progressive unfreezing -> Early stopping
```

---

##  Evaluation & Results

### Metrics

| Metric | Description |
|---|---|
| **Validation Accuracy** | % correctly classified across 3 classes |
| **Cross-Entropy Loss** | Confidence-aware training/eval loss |
| **Latency / Throughput** | Runtime performance for deployment |
| **Robustness Drop** | Accuracy drop under adversarial perturbations |

### Final Results Summary (Core Models)

| Architecture | Dataset Scale | Final Accuracy | Goal (>70%) | Stability |
|---|---|---|---|---|
| Base (Concat + Linear, Top 6 Unfrozen) | 100% | 73.73% | Met | Highly stable |
| Base (Concat + Linear, Top 2 Unfrozen) | 100% | 70.69% | Met | Stable |
| SOTA (Attention + SwiGLU, Top 2 Unfrozen) | 50% | 67.21% | Missed | Stable on limited data |
| SOTA Collapse (Attention + SwiGLU, Top 4 Unfrozen) | 100% + Hard Negs | 33.37% | Failed | Unstable |

---

##  Hard Negatives & SOTA Collapse

### What Happened in Experiment 9

The attention + SwiGLU architecture was stress-tested with hard negatives and deeper unfreezing. In epoch 2, validation loss became NaN and accuracy dropped to chance level.

### Root Cause (High-Level)

```text
Hard negatives + deeper unfreezing + fragile gated attention math
-> gradient explosion / representation drift
-> NaN collapse
```

### Why the Simpler Base Model Survived

- Concatenation + linear head has lower numerical fragility.
- It handles gradient spikes better in full-scale training.

---

### Baseline Checkpoint Decision for Task 1-5 Work

Before running Task 1-5 deliverables, we standardized on **`final_sota_visual_entailment3.pth` (64.18% clean-test accuracy)** as the practical baseline in app and analysis pipelines.

Even though a **73.7%** model exists on prior validation experiments, the **64.18% checkpoint produced better behavior on custom user data** (less brittle outputs and more consistent real-image predictions), so it was selected as the project baseline for downstream tasks.

### Task 1-5 Deliverables (Details, Results, Execution)

| Task | What was implemented | Key result | How to run |
|---|---|---|---|
| **Task 1** CoreML + Quantization | ONNX export (`opset=14`) + INT8 quantization + CoreML hybrid benchmarking | ONNX size 327.52 MB -> 83.01 MB (**74.65% reduction**) | `nenv/bin/python -c "from task1_coreml_pipeline import Task1Config, run_task1; run_task1(Task1Config())"` or `task_1.ipynb` |
| **Task 2** Cross-Attention Viz | Fusion cross-attention hooks + token/overall overlays + artifact indexing | 20 curated examples, selected-set accuracy **80.00%**, 40 attention tensors | `nenv/bin/python -c "from task2_cross_attention_viz import Task2Config, run_task2; run_task2(Task2Config())"` or `task_2.ipynb` |
| **Task 3** ViT Pruning | ViT depth ablations (12/9/6), 2-epoch head tuning, latency/param tradeoff | 9-layer gives best tradeoff: **62.18% val**, **234.05 ms** (1.87x faster than 12-layer) | `task_3.ipynb` + checkpoints (`final_sota_visual_entailment3.pth`, `task_3_9.pth`, `task_3_6.pth`) |
| **Task 4** Prompt Transfer | Frozen encoders + adapter head + concept-aware prompt template evaluation | Best mean zero-shot template: **`high_specificity` = 0.3997**; mean one-shot gain +0.0035 | `nenv/bin/python -c "from task4_pipeline import run_task4; run_task4(base_dir='.', train_sample=12000, dev_sample=3000, eval_sample=1500, batch_size=64, seed=42)"` or `task_4.ipynb` |
| **Task 5** Adversarial Robustness | 100 adversarial pairs, FGSM/PGD, text attacks, token/patch vulnerability | Clean test acc **0.6411**; PGD @4/255 drives adversarial accuracy to **0.0000** | `nenv/bin/python -c "from task5_adversarial import Task5Config, run_task5; run_task5(Task5Config())"` or `task_5.ipynb` |

For full per-task reports, see:
- [`README_task1.md`](README_task1.md)
- [`README_task2.md`](README_task2.md)
- [`README_task3.md`](README_task3.md)
- [`README_task4.md`](README_task4.md)
- [`README_task5.md`](README_task5.md)


---

##  Demo - Streamlit App

The project now ships a **single unified Streamlit app** (`app.py`) with all interfaces merged:
- Core entailment analyzer,
- Task 2 heatmap viewer/generator,
- Task 4 prompt transfer studio,
- Task 5 robustness explorer.

### Running the App

```bash
# 1. Clone the repository
git clone <https://github.com/ddasgrid/Projects_GridDynamics2>
cd visual-entailment-analyzer

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place model weights in repository root
# Recommended baseline:
#   final_sota_visual_entailment3.pth

# 4. Launch unified app
streamlit run app.py
```

---

##  Reproducing This Project

### Requirements

**Hardware:** Apple Silicon Mac recommended (MPS acceleration), 16GB RAM preferred.

**Software:**

```text
# Core runtime
numpy>=1.26
pandas>=2.1
Pillow>=10.0
torch>=2.2
transformers>=4.40
streamlit>=1.30
tqdm>=4.66
psutil>=5.9
huggingface_hub>=0.23

# Data + NLP utilities used in task pipelines
datasets>=2.16
nltk>=3.8
spacy>=3.7

# Visualization
matplotlib>=3.8

# Task 1 export / optimization stack
onnx>=1.15
onnxruntime>=1.17
coremltools>=8.0; platform_system == "Darwin"
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Step-by-Step Reproduction

```text
Step 1 -> Run task_1.ipynb (CoreML export + INT8 quantization + benchmark)
Step 2 -> Run task_2.ipynb (cross-attention hook extraction + heatmaps)
Step 3 -> Run task_3.ipynb (ViT depth pruning + latency/accuracy ablation)
Step 4 -> Run task_4.ipynb (prompt transfer + concept diagnostics)
Step 5 -> Run task_5.ipynb (FGSM/PGD + text attacks + vulnerability analysis)
Step 6 -> streamlit run app.py (unified UI for all tasks)
```

Direct script execution options are documented in each task README.

---

##  Project Structure

```text
.
├── README.md
├── README_task1.md
├── README_task2.md
├── README_task3.md
├── README_task4.md
├── README_task5.md
├── DEPLOY_APP2.md
├── requirements.txt
├── app.py                        <- unified Streamlit app (core + task2 + task4 + task5)
├── app_backbones.py              <- shared baseline checkpoint list
├── app_task2.py                  <- Task 2 UI module used by app.py
├── app_task4.py                  <- Task 4 UI module used by app.py
├── app_task5.py                  <- Task 5 UI module used by app.py
├── task_1.ipynb
├── task_2.ipynb
├── task_3.ipynb
├── task_4.ipynb
├── task_5.ipynb
├── task1_coreml_pipeline.py
├── task2_cross_attention_viz.py
├── task4_pipeline.py
├── task5_adversarial.py
├── task1_artifacts/
├── task2_artifacts/
├── task3_artifacts/
├── task4_results.csv
├── task4_run_summary.json
├── task4_concept_diagnostics.csv
├── task4_concept_lexicons.json
├── task4_swiglu_adapter_best.pth     <- download from Google Drive:https://drive.google.com/drive/folders/1G8HXVayxYRuoW8L-69jry8BzTed1pGZP?usp=drive_link
├── prompt_engineering_guide_task4.md
├── what_till_we_do_wrong.md
└── task5_artifacts/
```

---

##  Future Scope

To push performance and deployment quality further:

1. Improve custom-data generalization while preserving benchmark accuracy.
2. Add stronger adversarial training (image + text) to harden robustness.
3. Move more of the multimodal stack into optimized runtimes for Apple devices.
4. Expand adapter/LoRA style lightweight transfer for new domains.

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

**Built with ViT + BERT | Extended through Tasks 1-5 | Unified Streamlit Deployment**

</div>
