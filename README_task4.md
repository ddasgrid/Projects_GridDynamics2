# Task 4: Prompt-Based Transfer for Visual Entailment

## Problem Statement
This task builds prompt-based transfer for visual entailment without retraining the full multimodal backbone.

Project baseline for all Task 1-5 runs:
- Backbone checkpoint: `final_sota_visual_entailment3.pth` (clean-test ~64.18%)
- Chosen for stronger custom-data behavior than a separate 73.7% validation checkpoint.

Core requirements:
- Freeze ViT + BERT and train only a lightweight adapter head.
- Evaluate prompt-template framing on held-out + synthetic concept splits.
- Compare zero-shot vs one-shot behavior.
- Produce a prompt engineering guide with concept-wise recommendations.

## Pipeline
1. Load backbone checkpoint and freeze image/text encoders.
2. Train adapter/classifier head on fused features.
3. Evaluate template bank:
   - `base`
   - `low_specificity`
   - `high_specificity`
   - `concept_framing`
   - `negation_explicit`
4. Run concept-aware diagnostics across `action`, `count`, `spatial`, `negation`, `object_or_attribute`.
5. Export metrics, diagnostics, guide, and visuals.

## Results

### 1) Run Summary

| Metric | Value |
|---|---:|
| Device | `mps` |
| Adapter trainable params | 2,466,084 |
| Best validation accuracy (adapter training) | 44.30% |
| Final validation accuracy (adapter training) | 44.77% |
| Eval datasets | 6 |
| Prompt templates tested | 5 |

### 2) Template Ranking (mean zero-shot accuracy)

| Rank | Template | Mean Zero-Shot Acc |
|---:|---|---:|
| 1 | `high_specificity` | 0.3997 |
| 2 | `negation_explicit` | 0.3939 |
| 3 | `concept_framing` | 0.3840 |
| 4 | `low_specificity` | 0.3838 |
| 5 | `base` | 0.3739 |

### 3) Overall Transfer (`task4_results.csv`)

| Metric | Value |
|---|---:|
| Mean zero-shot accuracy | 0.3871 |
| Mean one-shot accuracy | 0.3905 |
| Mean gain (one-shot - zero-shot) | +0.0035 |

### 4) Best Template by Concept Type

| Concept Type | Best Template | Zero-Shot Acc |
|---|---|---:|
| action | `negation_explicit` | 0.3773 |
| count | `high_specificity` | 0.4840 |
| negation | `base` | 0.5144 |
| object_or_attribute | `concept_framing` | 0.4424 |
| spatial | `low_specificity` | 0.4449 |

### Notes on Current UI
- Task 4 UI is prompt-only (template analysis + custom playground).
- Attack logic is not part of Task 4 UI.

## How to Execute

### Option A: Notebook
Run `task_4.ipynb` top-to-bottom.

### Option B: Script
```bash
nenv/bin/python -c "from task4_pipeline import run_task4; run_task4(base_dir='.', train_sample=12000, dev_sample=3000, eval_sample=1500, batch_size=64, seed=42)"
```

### Option C: Unified App
```bash
streamlit run app.py
```
Then open **Task 4: Prompt Transfer** from the sidebar.

### Dependencies
```bash
pip install -r requirements.txt
```

## Files Created and What They Contain

### Source / Entry Files
- `task4_pipeline.py`  
  Full transfer pipeline (frozen encoders, adapter training, template eval, diagnostics, report generation).
- `task_4.ipynb`  
  Notebook runner for Task 4.
- `app_task4.py`  
  Task 4 UI module used inside unified `app.py`.

### Output Artifacts
- `task4_results.csv`  
  Main evaluation table (dataset x template x concept bucket).
- `task4_run_summary.json`  
  Run metadata, config, artifact paths.
- `task4_swiglu_adapter_best.pth`(Download from Google Drive:https://drive.google.com/drive/folders/1G8HXVayxYRuoW8L-69jry8BzTed1pGZP?usp=drive_link)  
  Trained adapter/head weights.
- `task4_concept_lexicons.json`  
  Concept lexicons.
- `task4_concept_diagnostics.csv`  
  Concept extraction/routing diagnostics.
- `prompt_engineering_guide_task4.md`  
  Prompt recommendations by concept type.
- `what_till_we_do_wrong.md`  
  Failure-analysis notes.
- `task4_visualizations.png`  
  Aggregated task visualizations.
