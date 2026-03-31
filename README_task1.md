# Task 1: CoreML Export & Quantization Pipeline


##  Deployed App

https://huggingface.co/spaces/DebDas02/Visual_Entailment


## Problem Statement
This task focuses on optimizing a multimodal Visual Entailment model (ViT vision encoder + BERT text encoder + fusion head) for practical on-device inference on macOS/iOS.

Project baseline for all Task 1-5 runs:
- Backbone checkpoint: `final_sota_visual_entailment3.pth`
- Rationale: selected for better custom-data behavior (clean-test ~64.18%) even though a separate historical validation run reached 73.7%.

Primary goals:
- Decouple the architecture into modality-specific components.
- Export the vision encoder to ONNX (`opset=14`) and quantize it to INT8.
- Combine quantized vision features with a full-precision fusion head via CoreML.
- Benchmark latency, throughput, and memory behavior on Apple M-series hardware.

## Pipeline
1. Load SNLI-VE CSVs and checkpoint (`final_sota_visual_entailment3.pth`).
2. Build VE model:
   - Vision: `google/vit-base-patch16-224`
   - Text: `bert-base-uncased`
   - Cross-attention + fusion/classifier head
3. Export ViT encoder to ONNX (`vision_encoder_opset14.onnx`).
4. Apply post-training dynamic quantization (INT8) with ONNX Runtime (`vision_encoder_int8.onnx`).
5. Convert fusion/classifier head to CoreML (`fusion_head_fp16.mlpackage`).
6. Benchmark:
   - Baseline: end-to-end PyTorch
   - Optimized: quantized ONNX vision + CoreML fusion head (hybrid)
   - Batch sizes: 1, 4, 8

## Results

### 1) Model Artifact Size

| Artifact | Size (MB) |
|---|---:|
| FP32 ViT ONNX (`vision_encoder_opset14.onnx`) | 327.52 |
| INT8 ViT ONNX (`vision_encoder_int8.onnx`) | 83.01 |
| Size reduction | **74.65%** |

### 2) Benchmark (`task1_artifacts/benchmark_results.csv`)

| Engine | Batch | Mean Latency (ms) | Throughput (items/s) | RSS Delta (MB) |
|---|---:|---:|---:|---:|
| pytorch | 1 | 21.26 | 47.03 | 11.66 |
| quantized_hybrid_coreml | 1 | 31.75 | 31.50 | 4.27 |
| pytorch | 4 | 45.42 | 88.07 | 8.25 |
| quantized_hybrid_coreml | 4 | 90.69 | 44.11 | 48.42 |
| pytorch | 8 | 84.50 | 94.68 | 7.14 |
| quantized_hybrid_coreml | 8 | 184.67 | 43.32 | 68.64 |

### Observed Behavior
- INT8 significantly reduced ONNX artifact size.
- In this run, hybrid ONNX+CoreML was slower than end-to-end PyTorch due to runtime handoff overhead.

## How to Execute

### Option A: Notebook
Run `task_1.ipynb` top-to-bottom.

### Option B: Script
```bash
nenv/bin/python -c "from task1_coreml_pipeline import Task1Config, run_task1; run_task1(Task1Config())"
```

### Option C: Unified App (inference only)
Task 1 is an export/benchmark pipeline, so it does not have a dedicated UI page.
Use the unified app for inference checks:
```bash
streamlit run app.py
```

### Dependencies
```bash
pip install -r requirements.txt
```

## Files Created and What They Contain

### Source / Entry Files
- `task1_coreml_pipeline.py`  
  End-to-end Task 1 implementation (export, quantization, CoreML conversion, benchmarking).
- `task_1.ipynb`  
  Notebook runner for Task 1.

### Output Artifacts (`task1_artifacts/`)
- `vision_encoder_opset14.onnx` and `vision_encoder_opset14.onnx.data`  
  FP32 ONNX export of ViT vision encoder.
  (Download from Google Drive:https://drive.google.com/drive/folders/1G8HXVayxYRuoW8L-69jry8BzTed1pGZP?usp=drive_link)
- `vision_encoder_int8.onnx`  
  Quantized INT8 ONNX vision encoder.
- `fusion_head_fp16.mlpackage`  
  CoreML fusion/classifier head package.
- `benchmark_results.csv`  
  Latency/throughput/RSS comparison table.
- `task1_summary.json`  
  Run config, artifact paths, benchmark preview.
- `coreml_tmp/` (created at runtime)  
  Temporary CoreML compiler/runtime files.
