# Task 3: Vision Encoder Layer Pruning & Ablation

## Problem Statement
This task optimizes the Visual Entailment model for edge deployment by pruning ViT depth and measuring the tradeoff between latency, parameters, and accuracy.

Project baseline for all Task 1-5 runs:
- Backbone checkpoint: `final_sota_visual_entailment3.pth` (clean-test ~64.18%)
- Chosen for better custom-data behavior compared to a separate 73.7% validation checkpoint.

Target requirements:
- Compare ViT depths `[1-12]`, `[1-9]`, `[1-6]`.
- Freeze vision encoder and fine-tune fusion/head for 2 epochs per variant.
- Measure validation accuracy, latency, and parameter count.
- Find minimum viable depth with target `>65%` validation accuracy.

## Pipeline
1. Start from checkpoint `final_sota_visual_entailment3.pth`.
2. Build depth-pruned variants: 12, 9, 6 transformer blocks.
3. Freeze `vit` + `bert`; train only fusion/classifier modules for 2 epochs.
4. Record train/val metrics, latency (ms per batch), and parameters.
5. Save CSV and accuracy-latency tradeoff plot.

## Results

### Evaluation Protocol

| Setting | Value |
|---|---|
| Dataset split | Validation (`cleaned_snli_ve_dev.csv`) |
| Fine-tuning epochs | 2 |
| Frozen modules | `vit`, `bert` |
| Trainable modules | `cross_attention`, `fusion_head`, `classifier_head` |
| Latency metric | Avg ms per batch |

### Pruning Tradeoff

| Variant | ViT Depth | Train Loss / Acc | Val Loss / Acc | Latency (ms/batch) | Params | Param Reduction vs 12L | Speedup vs 12L |
|---|---:|---:|---:|---:|---:|---:|---:|
| vit_12_layers | 12 | 0.8914 / 58.38% | 0.8459 / 61.90% | 436.45 | 198,892,035 | 0.00% | 1.00x |
| vit_9_layers | 9 | 0.8819 / 59.14% | 0.8315 / 62.18% | 234.05 | 177,628,419 | 10.69% | 1.87x |
| vit_6_layers | 6 | 0.8842 / 58.88% | 0.8404 / 61.47% | 167.02 | 156,364,803 | 21.38% | 2.61x |

### Parameter Breakdown

| Variant | ViT | BERT | Cross-Attn | Fusion Head | Classifier Head |
|---|---|---|---|---|---|
| 12-layer | 86,389,248 (trainable 0) | 109,482,240 (trainable 0) | 2,362,368 | 394,752 | 263,427 |
| 9-layer | 65,125,632 (trainable 0) | 109,482,240 (trainable 0) | 2,362,368 | 394,752 | 263,427 |
| 6-layer | 43,862,016 (trainable 0) | 109,482,240 (trainable 0) | 2,362,368 | 394,752 | 263,427 |

### Interpretation
- `vit_9_layers` is the best balance in this run (accuracy + speed).
- `vit_6_layers` is fastest, but accuracy drops versus 9-layer.
- The strict `>65%` target was not reached by tested variants.

## How to Execute

### Option A: Notebook
Run `task_3.ipynb` and evaluate each checkpoint variant.

### Option B: Artifact-based report
Use saved checkpoints and regenerate:
- `task3_artifacts/task3_results.csv`
- `task3_artifacts/task3_accuracy_latency_tradeoff.png`

### Option C: Unified App (comparison/inference)
Task 3 has no dedicated UI page; use unified app for inference checks:
```bash
streamlit run app.py
```

### Dependencies
```bash
pip install -r requirements.txt
```

## Files Created and What They Contain

### Source / Checkpoints
- `task_3.ipynb`  
  Task notebook for pruning/eval workflow.
- `final_sota_visual_entailment3.pth`  
  12-layer baseline checkpoint used for pruning initialization.
- `task_3_9.pth`  
  9-layer pruned checkpoint.
- `task_3_6.pth`  
  6-layer pruned checkpoint.

### Output Artifacts (`task3_artifacts/`)
- `task3_results.csv`  
  Metrics table (train/val, latency, params, reductions, speedups).
- `task3_accuracy_latency_tradeoff.png`  
  Accuracy-vs-latency curve.
- `task3_eval_notes.json`  
  Evaluation metadata and checkpoint mapping.
