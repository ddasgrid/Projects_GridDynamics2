# Task 5: Adversarial Image-Text Pair Generation & Robustness Testing

## Problem Statement
This task evaluates brittleness of the visual entailment model under controlled adversarial perturbations before deployment.

Project baseline for all Task 1-5 runs:
- Backbone checkpoint: `final_sota_visual_entailment3.pth` (clean-test ~64.18%)
- Chosen over a separate 73.7% validation checkpoint due to better custom-data behavior.

Core requirements:
- Build 100 adversarial image-text pairs.
- Run image attacks: FGSM and PGD.
- Run text attacks: negation/synonym/object-substitution/paraphrase style changes.
- Evaluate hardness at epsilon budgets `1/255`, `4/255`, `8/255`.
- Identify vulnerable text tokens and image patches.
- Measure accuracy drop from clean to adversarial sets.

## Pipeline
1. Load checkpoint and evaluate clean test accuracy.
2. Build 100-pair clean base set from correctly classified entailment samples.
3. Generate text adversaries by attack mode.
4. Generate image adversaries with:
   - FGSM (`eps in [1/255, 4/255, 8/255]`)
   - PGD (`steps=10`, `alpha=eps*0.25`)
5. Evaluate accuracy/flip-rate per attack setting.
6. Rank vulnerability:
   - token gradient sensitivity
   - 14x14 patch saliency
7. Save all outputs in `task5_artifacts/`.

## Demo
https://drive.google.com/file/d/166SjJg-4xDE4kgvdUTpMp2hyQmoOiclh/view?usp=sharing

## Results

### 1) Clean vs Adversarial Summary

| Metric | Value |
|---|---:|
| Clean test accuracy (full test) | 0.6411 |
| Clean base accuracy (100 selected pairs) | 1.0000 |
| Text adversarial accuracy | 0.6700 |
| Text flip rate | 0.6900 |
| Combined adversarial accuracy (strongest image + text) | 0.8000 |
| Combined flip rate | 1.0000 |

### 2) Image Attack Hardness by Epsilon

| Attack | Epsilon | Accuracy | Accuracy Drop vs Clean Base | Attack Success Rate |
|---|---:|---:|---:|---:|
| FGSM | 1/255 | 0.4400 | 0.5600 | 0.5600 |
| FGSM | 4/255 | 0.3100 | 0.6900 | 0.6900 |
| FGSM | 8/255 | 0.3300 | 0.6700 | 0.6700 |
| PGD | 1/255 | 0.0500 | 0.9500 | 0.9500 |
| PGD | 4/255 | 0.0000 | 1.0000 | 1.0000 |
| PGD | 8/255 | 0.0000 | 1.0000 | 1.0000 |

### 3) Vulnerability Highlights
- Top vulnerable token (mean grad norm): `chefs` (0.0461)
- Most salient image patch: `(row=1, col=13)` with mean saliency `0.00430`

### Notes on Current UI
- Manual text-attack mode is removed.
- In custom robustness testing, text attack is auto-selected by concept type when enabled.
- FGSM is single-step (no step-scale parameter); PGD uses step-scale because it is iterative.

## How to Execute

### Option A: Notebook
Run `task_5.ipynb` top-to-bottom.

### Option B: Script
```bash
nenv/bin/python -c "from task5_adversarial import Task5Config, run_task5; run_task5(Task5Config())"
```

### Option C: Unified App
```bash
streamlit run app.py
```
Then open **Task 5: Robustness** from the sidebar.

### Dependencies
```bash
pip install -r requirements.txt
```

## Files Created and What They Contain

### Source / Entry Files
- `task5_adversarial.py`  
  Full Task 5 pipeline (pair generation, FGSM/PGD, text attacks, robustness + vulnerability analysis).
- `task_5.ipynb`  
  Notebook runner for Task 5.
- `app_task5.py`  
  Task 5 UI module used inside unified `app.py`.

### Output Artifacts (`task5_artifacts/`)
- `task5_robustness_summary.json`  
  Config + aggregate robustness metrics.
- `task5_base_pairs.csv`  
  Clean 100-pair base set.
- `task5_text_adversarial_pairs.csv`  
  Text adversarial samples and attack mode info.
- `task5_image_attack_results.csv`  
  FGSM/PGD results by epsilon.
- `task5_text_attack_accuracy_by_mode.csv`  
  Accuracy curve by text attack type.
- `task5_token_vulnerability.csv`  
  Token vulnerability ranking.
- `task5_image_region_vulnerability.csv`  
  Patch vulnerability ranking (14x14 grid).
- `task5_robustness_dashboard.png`  
  Robustness dashboard plot.
