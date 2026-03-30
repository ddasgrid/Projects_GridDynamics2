from __future__ import annotations

import json
import random
import time
import os
import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

try:
    import onnx  # noqa: F401
    ONNX_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    ONNX_AVAILABLE = False

try:
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic
except Exception:  # pragma: no cover - optional dependency
    ort = None
    QuantType = None
    quantize_dynamic = None

try:
    import coremltools as ct
except Exception:  # pragma: no cover - optional dependency
    ct = None


LABEL2ID = {"entailment": 0, "neutral": 1, "contradiction": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


@dataclass
class Task1Config:
    train_csv: str = "master_augmented_snli_ve_train.csv"
    val_csv: str = "cleaned_snli_ve_dev.csv"
    test_csv: str = "cleaned_snli_ve_test.csv"
    checkpoint_path: str = "final_sota_visual_entailment3.pth"
    output_dir: str = "task1_artifacts"
    vision_model_name: str = "google/vit-base-patch16-224"
    text_model_name: str = "bert-base-uncased"
    max_text_length: int = 64
    batch_size: int = 8
    seed: int = 42
    # Training knobs (optional; defaults keep checkpoint usage fast).
    train_enabled: bool = False
    epochs: int = 2
    train_batch_size: int = 8
    lr: float = 2e-5
    weight_decay: float = 1e-4
    # Benchmark knobs.
    benchmark_repeats: int = 20
    benchmark_warmup: int = 5
    benchmark_batch_sizes: Tuple[int, ...] = (1, 4, 8)
    benchmark_pool_size: int = 128


class SwiGLU_MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int, dropout_rate: float = 0.3):
        super().__init__()
        self.w12 = nn.Linear(in_features, hidden_features * 2)
        self.w3 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        hidden = self.dropout(hidden)
        return self.w3(hidden)


class VisualEntailmentModel(nn.Module):
    def __init__(
        self,
        vision_model_name: str = "google/vit-base-patch16-224",
        text_model_name: str = "bert-base-uncased",
        hidden_dim: int = 512,
        dropout_rate: float = 0.3,
        depth: int = 1,
        freeze_mode: str = "partial",
        num_layers_to_freeze: int = 10,
    ):
        super().__init__()
        self.vit = AutoModel.from_pretrained(vision_model_name)
        self.bert = AutoModel.from_pretrained(text_model_name)
        self._apply_freezing(freeze_mode, num_layers_to_freeze)

        hidden_size = self.vit.config.hidden_size
        if hidden_size != self.bert.config.hidden_size:
            raise ValueError("ViT and BERT hidden sizes must match.")

        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True)

        fusion_layers: List[nn.Module] = []
        in_features = hidden_size
        for _ in range(depth):
            fusion_layers.append(nn.Linear(in_features, hidden_dim))
            fusion_layers.append(nn.LayerNorm(hidden_dim))
            fusion_layers.append(nn.GELU())
            fusion_layers.append(nn.Dropout(dropout_rate))
            in_features = hidden_dim
        self.fusion_head = nn.Sequential(*fusion_layers)
        self.classifier_head = SwiGLU_MLP(
            in_features=hidden_dim,
            hidden_features=hidden_dim // 2,
            out_features=3,
            dropout_rate=dropout_rate,
        )

    def _apply_freezing(self, mode: str, num_layers: int) -> None:
        if mode == "full":
            for p in self.vit.parameters():
                p.requires_grad = False
            for p in self.bert.parameters():
                p.requires_grad = False
            return

        if mode == "partial":
            def freeze_n_layers(model: nn.Module, n: int) -> None:
                if hasattr(model, "embeddings"):
                    for p in model.embeddings.parameters():
                        p.requires_grad = False
                if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
                    layers = model.encoder.layer
                    n_layers = min(n, len(layers))
                    for i in range(n_layers):
                        for p in layers[i].parameters():
                            p.requires_grad = False

            freeze_n_layers(self.vit, num_layers)
            freeze_n_layers(self.bert, num_layers)

    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        vit_outputs = self.vit(pixel_values=pixel_values)
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        vit_tokens = vit_outputs.last_hidden_state
        bert_cls = bert_outputs.last_hidden_state[:, 0, :]
        query = bert_cls.unsqueeze(1)
        attn_output, _ = self.cross_attention(query=query, key=vit_tokens, value=vit_tokens)
        base_fused = attn_output.squeeze(1)

        deep_fused = self.fusion_head(base_fused)
        logits = self.classifier_head(deep_fused)
        return logits


class SNLIVETrainDataset(Dataset):
    def __init__(self, csv_path: str, image_processor, tokenizer, max_text_length: int = 64):
        df = pd.read_csv(csv_path)
        df = df[df["label"].isin(LABEL2ID)].copy()
        df = df[df["image_path"].map(lambda p: Path(p).exists())].reset_index(drop=True)
        self.df = df
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        with Image.open(row["image_path"]) as im:
            image = im.convert("RGB")
        pixel_values = self.image_processor(images=[image], return_tensors="pt")["pixel_values"][0]
        text_tokens = self.tokenizer(
            str(row["hypothesis"]),
            truncation=True,
            padding="max_length",
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        return {
            "pixel_values": pixel_values,
            "input_ids": text_tokens["input_ids"][0],
            "attention_mask": text_tokens["attention_mask"][0],
            "label": torch.tensor(LABEL2ID[str(row["label"])], dtype=torch.long),
        }


class VisionEncoderONNXWrapper(nn.Module):
    def __init__(self, vit: nn.Module):
        super().__init__()
        self.vit = vit

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.vit(pixel_values=pixel_values).last_hidden_state


class FusionHeadOnly(nn.Module):
    def __init__(self, fusion_head: nn.Module, classifier_head: nn.Module):
        super().__init__()
        self.fusion_head = fusion_head
        self.classifier_head = classifier_head

    def forward(self, base_fused: torch.Tensor) -> torch.Tensor:
        return self.classifier_head(self.fusion_head(base_fused))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_processor_and_tokenizer(cfg: Task1Config):
    try:
        image_processor = AutoImageProcessor.from_pretrained(cfg.vision_model_name, local_files_only=True)
    except Exception:
        image_processor = AutoImageProcessor.from_pretrained(cfg.vision_model_name)
    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.text_model_name, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(cfg.text_model_name)
    return image_processor, tokenizer


def load_model_from_checkpoint(cfg: Task1Config, device: torch.device) -> VisualEntailmentModel:
    model = VisualEntailmentModel(
        vision_model_name=cfg.vision_model_name,
        text_model_name=cfg.text_model_name,
        hidden_dim=512,
        dropout_rate=0.3,
        depth=1,
        freeze_mode="partial",
        num_layers_to_freeze=10,
    )
    ckpt = Path(cfg.checkpoint_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


def train_model_if_requested(
    cfg: Task1Config,
    model: VisualEntailmentModel,
    image_processor,
    tokenizer,
    device: torch.device,
) -> VisualEntailmentModel:
    if not cfg.train_enabled:
        return model

    if not Path(cfg.train_csv).exists():
        raise FileNotFoundError(f"Training CSV not found: {cfg.train_csv}")

    train_ds = SNLIVETrainDataset(cfg.train_csv, image_processor, tokenizer, max_text_length=cfg.max_text_length)
    train_loader = DataLoader(train_ds, batch_size=cfg.train_batch_size, shuffle=True, num_workers=0)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        total = 0
        correct = 0
        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(pixel_values, input_ids, attention_mask)
            loss = criterion(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=-1)
            correct += int((preds == labels).sum().item())
            total += int(labels.size(0))

        avg_loss = epoch_loss / max(total, 1)
        acc = correct / max(total, 1)
        print(f"Epoch {epoch + 1}/{cfg.epochs} | train_loss={avg_loss:.4f} | train_acc={acc:.4f}")

    model.eval()
    torch.save(model.state_dict(), cfg.checkpoint_path)
    print(f"Saved trained checkpoint to {cfg.checkpoint_path}")
    return model


def export_vision_encoder_to_onnx(model: VisualEntailmentModel, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Use a detached copy so export device changes do not mutate the live model.
    wrapper = VisionEncoderONNXWrapper(copy.deepcopy(model.vit)).eval().cpu()
    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    export_kwargs = dict(
        input_names=["pixel_values"],
        output_names=["last_hidden_state"],
        dynamic_axes={"pixel_values": {0: "batch"}, "last_hidden_state": {0: "batch"}},
        opset_version=14,
        do_constant_folding=True,
    )

    # Prefer the legacy exporter to avoid a hard dependency on onnxscript.
    try:
        torch.onnx.export(
            wrapper,
            dummy_input,
            output_path.as_posix(),
            dynamo=False,
            **export_kwargs,
        )
    except TypeError:
        # Older torch versions do not expose `dynamo`; retry without it.
        torch.onnx.export(
            wrapper,
            dummy_input,
            output_path.as_posix(),
            **export_kwargs,
        )
    except ModuleNotFoundError as exc:
        if "onnxscript" in str(exc):
            raise RuntimeError(
                "ONNX export requires `onnxscript` with the new exporter. "
                "Install it (`pip install onnxscript`) or keep `dynamo=False`."
            ) from exc
        raise
    return output_path


def quantize_onnx_int8(onnx_input_path: Path, onnx_output_path: Path) -> Path:
    if quantize_dynamic is None or QuantType is None:
        raise RuntimeError("onnxruntime quantization is unavailable. Install onnxruntime.")
    onnx_output_path.parent.mkdir(parents=True, exist_ok=True)
    base_kwargs = {
        "model_input": onnx_input_path.as_posix(),
        "model_output": onnx_output_path.as_posix(),
        "weight_type": QuantType.QInt8,
    }
    # Handle both ORT APIs: some versions support optimize_model, others reject it.
    try:
        quantize_dynamic(**base_kwargs, optimize_model=True)
    except TypeError as exc:
        if "optimize_model" not in str(exc):
            raise
        quantize_dynamic(**base_kwargs)
    return onnx_output_path


def convert_fusion_head_to_coreml(model: VisualEntailmentModel, output_path: Path) -> Optional[Path]:
    if ct is None:
        print("coremltools not installed. Skipping CoreML export.")
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use detached copies so CoreML tracing stays CPU-only without altering the live model.
    fusion_module = FusionHeadOnly(
        copy.deepcopy(model.fusion_head),
        copy.deepcopy(model.classifier_head),
    ).eval().cpu()
    traced = torch.jit.trace(fusion_module, (torch.randn(1, model.vit.config.hidden_size),))
    traced.eval()

    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name="base_fused", shape=(ct.RangeDim(1, 8), model.vit.config.hidden_size))],
        outputs=[ct.TensorType(name="logits")],
        minimum_deployment_target=ct.target.macOS13,
    )
    mlmodel.save(output_path.as_posix())
    return output_path


def _configure_coreml_tmpdir(base_dir: Path) -> Path:
    tmp_dir = base_dir / "coreml_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = tmp_dir.as_posix()
    os.environ["TMP"] = tmp_dir.as_posix()
    os.environ["TEMP"] = tmp_dir.as_posix()
    return tmp_dir


def _load_coreml_model_if_available(
    coreml_path: Optional[Path], hidden_size: int
):
    if coreml_path is None or ct is None:
        return None
    try:
        coreml_model = ct.models.MLModel(coreml_path.as_posix())
        # Probe predict once so downstream loops can cleanly fallback.
        _ = coreml_model.predict(
            {"base_fused": np.zeros((1, hidden_size), dtype=np.float32)}
        )
        return coreml_model
    except Exception as exc:
        print(
            "CoreML runtime unavailable for predict(); "
            f"falling back to PyTorch fusion. Error: {exc}"
        )
        return None


def _open_rgb(path: str) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def prepare_batch_inputs(
    image_paths: Sequence[str],
    texts: Sequence[str],
    image_processor,
    tokenizer,
    max_text_length: int,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    images = [_open_rgb(p) for p in image_paths]
    pixel_values = image_processor(images=images, return_tensors="pt")["pixel_values"]
    toks = tokenizer(
        list(texts),
        truncation=True,
        padding=True,
        max_length=max_text_length,
        return_tensors="pt",
    )
    return pixel_values, toks


@torch.no_grad()
def pytorch_infer_batch(
    model: VisualEntailmentModel,
    pixel_values: torch.Tensor,
    toks: Dict[str, torch.Tensor],
    device: torch.device,
) -> np.ndarray:
    logits = model(
        pixel_values.to(device),
        toks["input_ids"].to(device),
        toks["attention_mask"].to(device),
    )
    return logits.detach().cpu().numpy()


@torch.no_grad()
def hybrid_infer_batch(
    model: VisualEntailmentModel,
    ort_session,
    coreml_model,
    pixel_values: torch.Tensor,
    toks: Dict[str, torch.Tensor],
    device: torch.device,
) -> np.ndarray:
    # 1) Quantized vision branch (ONNXRuntime INT8 model).
    ort_out = ort_session.run(None, {"pixel_values": pixel_values.numpy().astype(np.float32)})[0]
    vit_tokens = torch.from_numpy(ort_out).to(device=device, dtype=torch.float32)

    # 2) Full-precision text branch.
    bert_outputs = model.bert(
        input_ids=toks["input_ids"].to(device),
        attention_mask=toks["attention_mask"].to(device),
    )
    bert_cls = bert_outputs.last_hidden_state[:, 0, :]

    # 3) Full-precision fusion query and base_fused tensor.
    query = bert_cls.unsqueeze(1)
    attn_output, _ = model.cross_attention(query=query, key=vit_tokens, value=vit_tokens)
    base_fused = attn_output.squeeze(1)

    # 4) Fusion head via CoreML if available, else fallback to PyTorch.
    if coreml_model is not None:
        try:
            out = coreml_model.predict(
                {"base_fused": base_fused.detach().cpu().numpy().astype(np.float32)}
            )
            logits = np.array(out["logits"], dtype=np.float32)
            return logits
        except Exception as exc:
            if not getattr(hybrid_infer_batch, "_coreml_fallback_warned", False):
                print(
                    "CoreML predict failed during inference; "
                    f"using PyTorch fusion fallback. Error: {exc}"
                )
                hybrid_infer_batch._coreml_fallback_warned = True

    logits = model.classifier_head(model.fusion_head(base_fused)).detach().cpu().numpy()
    return logits


def sample_benchmark_pool(test_csv: str, pool_size: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv(test_csv)
    df = df[df["label"].isin(LABEL2ID)].copy()
    df = df[df["image_path"].map(lambda p: Path(p).exists())]
    if len(df) == 0:
        raise RuntimeError("No benchmark rows with valid image paths.")
    return df.sample(n=min(pool_size, len(df)), random_state=seed).reset_index(drop=True)


def _measure_latency(func, warmup: int, repeats: int) -> Tuple[float, float, float]:
    for _ in range(warmup):
        func()
    times_ms = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        func()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)
    arr = np.array(times_ms, dtype=np.float64)
    return float(arr.mean()), float(np.percentile(arr, 50)), float(np.percentile(arr, 90))


def benchmark_pipeline(
    cfg: Task1Config,
    model: VisualEntailmentModel,
    image_processor,
    tokenizer,
    device: torch.device,
    quantized_onnx_path: Optional[Path],
    coreml_path: Optional[Path],
) -> pd.DataFrame:
    pool = sample_benchmark_pool(cfg.test_csv, cfg.benchmark_pool_size, cfg.seed)
    process = psutil.Process()
    results = []

    ort_session = None
    if quantized_onnx_path is not None:
        if ort is None:
            print("onnxruntime not installed. Skipping hybrid benchmark.")
        else:
            ort_session = ort.InferenceSession(
                quantized_onnx_path.as_posix(),
                providers=["CPUExecutionProvider"],
            )

    coreml_model = _load_coreml_model_if_available(
        coreml_path, model.vit.config.hidden_size
    )

    for batch_size in cfg.benchmark_batch_sizes:
        batch = pool.sample(n=min(batch_size, len(pool)), random_state=cfg.seed + batch_size)
        pixel_values, toks = prepare_batch_inputs(
            image_paths=batch["image_path"].tolist(),
            texts=batch["hypothesis"].astype(str).tolist(),
            image_processor=image_processor,
            tokenizer=tokenizer,
            max_text_length=cfg.max_text_length,
        )

        rss_before = process.memory_info().rss / (1024 ** 2)
        pt_mean, pt_p50, pt_p90 = _measure_latency(
            lambda: pytorch_infer_batch(model, pixel_values, toks, device),
            warmup=cfg.benchmark_warmup,
            repeats=cfg.benchmark_repeats,
        )
        rss_after = process.memory_info().rss / (1024 ** 2)
        results.append(
            {
                "engine": "pytorch",
                "batch_size": int(batch_size),
                "latency_mean_ms": pt_mean,
                "latency_p50_ms": pt_p50,
                "latency_p90_ms": pt_p90,
                "throughput_items_per_sec": float(1000.0 * batch_size / max(pt_mean, 1e-9)),
                "rss_delta_mb": float(rss_after - rss_before),
            }
        )

        if ort_session is not None:
            rss_before = process.memory_info().rss / (1024 ** 2)
            hy_mean, hy_p50, hy_p90 = _measure_latency(
                lambda: hybrid_infer_batch(model, ort_session, coreml_model, pixel_values, toks, device),
                warmup=cfg.benchmark_warmup,
                repeats=cfg.benchmark_repeats,
            )
            rss_after = process.memory_info().rss / (1024 ** 2)
            results.append(
                {
                    "engine": "quantized_hybrid_coreml" if coreml_model is not None else "quantized_hybrid_pytorch_fusion",
                    "batch_size": int(batch_size),
                    "latency_mean_ms": hy_mean,
                    "latency_p50_ms": hy_p50,
                    "latency_p90_ms": hy_p90,
                    "throughput_items_per_sec": float(1000.0 * batch_size / max(hy_mean, 1e-9)),
                    "rss_delta_mb": float(rss_after - rss_before),
                }
            )

    return pd.DataFrame(results)


def run_task1(config: Optional[Task1Config] = None) -> Dict[str, object]:
    cfg = config or Task1Config()
    set_seed(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _configure_coreml_tmpdir(out_dir)

    if not ONNX_AVAILABLE:
        raise RuntimeError(
            "The `onnx` package is required for Task 1 export. "
            "Install dependencies first: pip install onnx onnxruntime coremltools"
        )

    device = pick_device()
    print(f"Using device: {device}")

    image_processor, tokenizer = load_processor_and_tokenizer(cfg)
    model = load_model_from_checkpoint(cfg, device)
    model = train_model_if_requested(cfg, model, image_processor, tokenizer, device)
    model.eval()

    onnx_fp_path = export_vision_encoder_to_onnx(model, out_dir / "vision_encoder_opset14.onnx")
    print(f"Exported ONNX vision encoder: {onnx_fp_path}")

    onnx_int8_path: Optional[Path] = None
    if quantize_dynamic is not None:
        onnx_int8_path = quantize_onnx_int8(onnx_fp_path, out_dir / "vision_encoder_int8.onnx")
        print(f"Exported INT8 ONNX: {onnx_int8_path}")
    else:
        print("onnxruntime quantization unavailable. Skipping INT8 export.")

    coreml_path: Optional[Path] = None
    if ct is not None:
        coreml_path = convert_fusion_head_to_coreml(model, out_dir / "fusion_head_fp16.mlpackage")
        print(f"Exported CoreML fusion head: {coreml_path}")
    else:
        print("coremltools unavailable. Skipping CoreML conversion.")

    # Reassert device placement before benchmarking in case any export path touched module devices.
    model.to(device).eval()

    benchmark_df = benchmark_pipeline(
        cfg=cfg,
        model=model,
        image_processor=image_processor,
        tokenizer=tokenizer,
        device=device,
        quantized_onnx_path=onnx_int8_path,
        coreml_path=coreml_path,
    )
    bench_csv = out_dir / "benchmark_results.csv"
    benchmark_df.to_csv(bench_csv, index=False)

    if onnx_int8_path is not None:
        verify_quantization_accuracy(
            cfg=cfg, 
            model=model, 
            image_processor=image_processor, 
            tokenizer=tokenizer, 
            device=device, 
            quantized_onnx_path=onnx_int8_path, 
            coreml_path=coreml_path
        )

    summary = {
        "config": asdict(cfg),
        "artifacts": {
            "onnx_fp32": str(onnx_fp_path),
            "onnx_int8": None if onnx_int8_path is None else str(onnx_int8_path),
            "coreml_fusion_head": None if coreml_path is None else str(coreml_path),
            "benchmark_csv": str(bench_csv),
        },
        "benchmark_preview": benchmark_df.to_dict(orient="records"),
    }

    summary_path = out_dir / "task1_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {summary_path}")
    return summary

def verify_quantization_accuracy(
    cfg: Task1Config,
    model: VisualEntailmentModel,
    image_processor,
    tokenizer,
    device: torch.device,
    quantized_onnx_path: Path,
    coreml_path: Optional[Path]
):
    print("\n--- STEP 5: QUANTIZATION ACCURACY VERIFICATION ---")

    # Load the optimized pipeline.
    if ort is None:
        raise RuntimeError("onnxruntime is required for quantized accuracy verification.")
    ort_session = ort.InferenceSession(quantized_onnx_path.as_posix(), providers=["CPUExecutionProvider"])
    coreml_model = _load_coreml_model_if_available(
        coreml_path, model.vit.config.hidden_size
    )
    if coreml_model is None:
        print("Using PyTorch fusion fallback for hybrid accuracy verification.")

    # Grab 100 random samples from the test set.
    pool = sample_benchmark_pool(cfg.test_csv, pool_size=100, seed=cfg.seed)
    
    correct_pytorch = 0
    correct_quantized = 0
    matches_between_models = 0
    total = len(pool)
    
    print(f"Evaluating {total} samples to verify accuracy retention...")

    # Process in batches of 10 to limit memory usage.
    batch_size = 10
    for start in range(0, total, batch_size):
        batch = pool.iloc[start:start+batch_size]
        
        pixel_values, toks = prepare_batch_inputs(
            image_paths=batch["image_path"].tolist(),
            texts=batch["hypothesis"].astype(str).tolist(),
            image_processor=image_processor,
            tokenizer=tokenizer,
            max_text_length=cfg.max_text_length,
        )
        
        true_labels = np.array([LABEL2ID[str(l)] for l in batch["label"]])
        
        # 1) Get PyTorch predictions.
        pt_logits = pytorch_infer_batch(model, pixel_values, toks, device)
        pt_preds = np.argmax(pt_logits, axis=-1)

        # 2) Get quantized hybrid predictions (CoreML or PyTorch fusion fallback).
        hy_logits = hybrid_infer_batch(model, ort_session, coreml_model, pixel_values, toks, device)
        hy_preds = np.argmax(hy_logits, axis=-1)

        # Tally scores.
        correct_pytorch += np.sum(pt_preds == true_labels)
        correct_quantized += np.sum(hy_preds == true_labels)
        matches_between_models += np.sum(pt_preds == hy_preds)
        
    pt_acc = (correct_pytorch / total) * 100
    hy_acc = (correct_quantized / total) * 100
    retention = (matches_between_models / total) * 100
    
    print("\n--- ACCURACY RESULTS ---")
    print(f"Original PyTorch Accuracy : {pt_acc:.1f}%")
    engine_name = "Quantized Apple Accuracy" if coreml_model is not None else "Quantized Hybrid Accuracy"
    print(f"{engine_name:<26}: {hy_acc:.1f}%")
    print(f"Model Agreement (Retention): {retention:.1f}%")

    if retention > 95.0:
        print("SUCCESS: Quantized model maintained >95% identical predictions.")
    else:
        print("WARNING: Quantization caused significant prediction drift.")

    return pt_acc, hy_acc, retention

if __name__ == "__main__":
    run_task1()
