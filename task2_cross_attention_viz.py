from __future__ import annotations
import textwrap

import json
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl_cache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LABEL2ID = {"entailment": 0, "neutral": 1, "contradiction": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
DIFFICULTY_ORDER = ("easy", "medium", "hard")


@dataclass
class Task2Config:
    test_csv: str = "cleaned_snli_ve_test.csv"
    checkpoint_path: str = "final_sota_visual_entailment3.pth"
    vision_model_name: str = "google/vit-base-patch16-224"
    text_model_name: str = "bert-base-uncased"
    output_dir: str = "task2_artifacts"
    num_examples: int = 20
    pool_size: int = 2500
    batch_size: int = 8
    max_text_length: int = 128
    fusion_attention_layers: int = 2
    fusion_attention_heads: int = 8
    top_tokens_per_example: int = 4
    seed: int = 42


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


class VisualEntailmentWithFusionAttention(nn.Module):
    def __init__(
        self,
        vision_model_name: str,
        text_model_name: str,
        hidden_dim: int = 512,
        dropout_rate: float = 0.3,
        depth: int = 1,
        fusion_attention_layers: int = 2,
        fusion_attention_heads: int = 8,
    ):
        super().__init__()
        self.vit = AutoModel.from_pretrained(vision_model_name)
        self.bert = AutoModel.from_pretrained(text_model_name)

        vit_hidden = self.vit.config.hidden_size
        bert_hidden = self.bert.config.hidden_size
        if vit_hidden != bert_hidden:
            raise ValueError("ViT and BERT hidden sizes must match.")
        self.hidden_size = vit_hidden

        # Original task head path (checkpoint-compatible).
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            batch_first=True,
        )
        fusion_layers: List[nn.Module] = []
        in_features = self.hidden_size
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

        # New fusion attention stack: image patches query text tokens.
        self.fusion_attn_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=self.hidden_size,
                    num_heads=fusion_attention_heads,
                    batch_first=True,
                )
                for _ in range(fusion_attention_layers)
            ]
        )
        self.fusion_attn_norms = nn.ModuleList([nn.LayerNorm(self.hidden_size) for _ in range(fusion_attention_layers)])
        self.fusion_attn_drop = nn.Dropout(dropout_rate)
        self.fusion_projection = nn.Linear(self.hidden_size, hidden_dim)
        # Gate starts at zero so checkpoint predictions are stable before finetuning.
        self.fusion_gate = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        vit_outputs = self.vit(pixel_values=pixel_values)
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        vit_all = vit_outputs.last_hidden_state  # [B, 197, H]
        bert_all = bert_outputs.last_hidden_state  # [B, T, H]

        vit_cls = vit_all[:, 0, :]
        bert_cls = bert_all[:, 0, :]

        # New fusion stack with patch-token cross-attention (hooks read weights here).
        img_tokens = vit_all[:, 1:, :]  # [B, 196, H]
        key_padding_mask = ~attention_mask.bool()  # True means "ignore this key position"
        for attn_layer, norm_layer in zip(self.fusion_attn_layers, self.fusion_attn_norms):
            attn_out, _ = attn_layer(
                query=img_tokens,
                key=bert_all,
                value=bert_all,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                average_attn_weights=False,
            )
            img_tokens = norm_layer(img_tokens + self.fusion_attn_drop(attn_out))

        # Original trained path.
        query = bert_cls.unsqueeze(1)
        key_value = vit_all
        attn_output, _ = self.cross_attention(query=query, key=key_value, value=key_value)
        base_fused = attn_output.squeeze(1)
        deep_fused = self.fusion_head(base_fused)

        # Blend in the new fusion stack summary.
        fusion_context = img_tokens.mean(dim=1)
        fusion_bonus = self.fusion_projection(fusion_context)
        deep_fused = deep_fused + torch.tanh(self.fusion_gate) * fusion_bonus

        logits = self.classifier_head(deep_fused)
        return logits


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


def _load_processor(vision_model_name: str):
    try:
        return AutoImageProcessor.from_pretrained(vision_model_name, local_files_only=True)
    except Exception:
        return AutoImageProcessor.from_pretrained(vision_model_name)


def _load_tokenizer(text_model_name: str):
    try:
        return AutoTokenizer.from_pretrained(text_model_name, local_files_only=True)
    except Exception:
        return AutoTokenizer.from_pretrained(text_model_name)


def _open_rgb(path: str) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def _prepare_pixels(image_paths: List[str], processor) -> torch.Tensor:
    images = [_open_rgb(p) for p in image_paths]
    return processor(images=images, return_tensors="pt")["pixel_values"]


def _prepare_text(texts: List[str], tokenizer, max_length: int) -> Dict[str, torch.Tensor]:
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


@torch.no_grad()
def predict_from_tensors(
    model: nn.Module,
    pixel_values: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    pred_chunks = []
    conf_chunks = []
    for start in range(0, len(pixel_values), batch_size):
        end = start + batch_size
        x = pixel_values[start:end].to(device)
        ids = input_ids[start:end].to(device)
        mask = attention_mask[start:end].to(device)
        logits = model(x, ids, mask)
        probs = torch.softmax(logits, dim=-1)
        pred_chunks.append(torch.argmax(probs, dim=-1).cpu())
        conf_chunks.append(torch.max(probs, dim=-1).values.cpu())
    return torch.cat(pred_chunks), torch.cat(conf_chunks)


def _difficulty(pred_id: int, true_id: int, confidence: float) -> str:
    if pred_id == true_id and confidence >= 0.80:
        return "easy"
    if pred_id == true_id and confidence >= 0.55:
        return "medium"
    return "hard"


def build_prediction_pool(
    model: nn.Module,
    df: pd.DataFrame,
    processor,
    tokenizer,
    cfg: Task2Config,
    device: torch.device,
) -> pd.DataFrame:
    valid = df[df["label"].isin(LABEL2ID)].copy()
    valid = valid[valid["image_path"].map(lambda p: Path(p).exists())]
    pool = valid.sample(n=min(cfg.pool_size, len(valid)), random_state=cfg.seed).reset_index(drop=False)
    pool = pool.rename(columns={"index": "source_index"})

    rows = []
    for start in range(0, len(pool), cfg.batch_size):
        batch = pool.iloc[start : start + cfg.batch_size]
        px = _prepare_pixels(batch["image_path"].tolist(), processor)
        toks = _prepare_text(batch["hypothesis"].astype(str).tolist(), tokenizer, max_length=cfg.max_text_length)
        preds, confs = predict_from_tensors(
            model=model,
            pixel_values=px,
            input_ids=toks["input_ids"],
            attention_mask=toks["attention_mask"],
            device=device,
            batch_size=cfg.batch_size,
        )

        for i, (_, row) in enumerate(batch.iterrows()):
            true_id = LABEL2ID[str(row["label"])]
            pred_id = int(preds[i].item())
            conf = float(confs[i].item())
            rows.append(
                {
                    "source_index": int(row["source_index"]),
                    "image_path": row["image_path"],
                    "hypothesis": row["hypothesis"],
                    "true_label": str(row["label"]),
                    "pred_label": ID2LABEL[pred_id],
                    "pred_confidence": conf,
                    "difficulty": _difficulty(pred_id, true_id, conf),
                    "is_correct": int(pred_id == true_id),
                }
            )
    return pd.DataFrame(rows)


def select_20_examples(pool_df: pd.DataFrame, num_examples: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    picked_indices = set()
    picked_rows = []

    # First pass: aim for class x difficulty coverage.
    for label in ("entailment", "neutral", "contradiction"):
        for diff in DIFFICULTY_ORDER:
            candidates = pool_df[(pool_df["true_label"] == label) & (pool_df["difficulty"] == diff)]
            if len(candidates) == 0:
                continue
            take = min(2, len(candidates))
            sample_rows = candidates.sample(n=take, random_state=rng.randint(0, 10_000_000))
            for idx, row in sample_rows.iterrows():
                if idx in picked_indices:
                    continue
                picked_indices.add(idx)
                picked_rows.append(row)
                if len(picked_rows) >= num_examples:
                    return pd.DataFrame(picked_rows).reset_index(drop=True)

    # Fill remaining slots.
    remaining = pool_df.drop(index=list(picked_indices), errors="ignore")
    if len(remaining) > 0 and len(picked_rows) < num_examples:
        extra = remaining.sample(
            n=min(num_examples - len(picked_rows), len(remaining)),
            random_state=seed + 123,
        )
        for _, row in extra.iterrows():
            picked_rows.append(row)

    out = pd.DataFrame(picked_rows).head(num_examples).reset_index(drop=True)
    return out


def register_attention_hooks(model: VisualEntailmentWithFusionAttention):
    cache: Dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(layer_name: str):
        def hook(_module, _inp, out):
            if isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
                cache[layer_name] = out[1].detach().cpu()
        return hook

    for i, layer in enumerate(model.fusion_attn_layers):
        handles.append(layer.register_forward_hook(make_hook(f"layer_{i}")))
    return cache, handles


def _normalized_heatmap(flat_scores: torch.Tensor, grid_size: int = 14) -> np.ndarray:
    arr = flat_scores.detach().cpu().numpy().reshape(grid_size, grid_size)
    arr = arr - arr.min()
    denom = arr.max() + 1e-9
    return arr / denom


#def _overlay_heatmap(ax, image: Image.Image, heat: np.ndarray, title: str) -> None:
    #heat_img = Image.fromarray((heat * 255).astype(np.uint8)).resize(image.size, resample=Image.BILINEAR)
    #heat_resized = np.array(heat_img, dtype=np.float32) / 255.0
    #ax.imshow(image)
    #ax.imshow(heat_resized, cmap="jet", alpha=0.45)
    #ax.set_title(title, fontsize=9)
    #ax.axis("off")

def _overlay_heatmap(ax, image: Image.Image, heat: np.ndarray, title: str, text: str = "") -> None:
    heat_img = Image.fromarray((heat * 255).astype(np.uint8)).resize(image.size, resample=Image.BILINEAR)
    heat_resized = np.array(heat_img, dtype=np.float32) / 255.0
    ax.imshow(image)
    ax.imshow(heat_resized, cmap="jet", alpha=0.45)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    
    # Add the hypothesis text to the image
    if text:
        # Wrap the text so it doesn't run off the edges
        wrapped_text = textwrap.fill(text, width=30) 
        
        # Add text with a background box for readability
        ax.text(
            0.5, 0.12, wrapped_text, 
            transform=ax.transAxes, 
            fontsize=14, 
            color='white', 
            ha='center', 
            va='bottom',
            bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.3')
        )


def _select_token_indices(
    token_ids: List[int],
    attention_mask: List[int],
    tokenizer,
    token_scores: torch.Tensor,
    top_k: int,
) -> List[int]:
    special_ids = set(tokenizer.all_special_ids)
    candidates = []
    valid_len = int(sum(attention_mask))
    for i in range(valid_len):
        tok_id = int(token_ids[i])
        if tok_id in special_ids:
            continue
        score = float(token_scores[i].item())
        candidates.append((i, score))
    candidates = sorted(candidates, key=lambda it: it[1], reverse=True)
    return [idx for idx, _ in candidates[:top_k]]


def extract_and_visualize_example(
    model: VisualEntailmentWithFusionAttention,
    processor,
    tokenizer,
    device: torch.device,
    cfg: Task2Config,
    row: pd.Series,
    example_idx: int,
    output_dir: Path,
) -> Dict[str, object]:
    image = _open_rgb(str(row["image_path"]))
    pixel_values = processor(images=[image], return_tensors="pt")["pixel_values"]
    tokens = _prepare_text([str(row["hypothesis"])], tokenizer, max_length=cfg.max_text_length)

    hook_cache, handles = register_attention_hooks(model)
    try:
        with torch.no_grad():
            logits = model(
                pixel_values.to(device),
                tokens["input_ids"].to(device),
                tokens["attention_mask"].to(device),
            )
            probs = torch.softmax(logits, dim=-1).cpu().squeeze(0)
    finally:
        for h in handles:
            h.remove()

    pred_id = int(torch.argmax(probs).item())
    pred_label = ID2LABEL[pred_id]
    pred_conf = float(torch.max(probs).item())

    tensor_dir = output_dir / "attention_tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)

    saved_tensors = {}
    for layer_name, tensor in hook_cache.items():
        # [B, heads, seq_img, seq_text] -> [heads, seq_img, seq_text]
        t = tensor[0].numpy()
        save_path = tensor_dir / f"example_{example_idx:02d}_{layer_name}.npy"
        np.save(save_path, t)
        saved_tensors[layer_name] = str(save_path)

    # Visualization with last fusion layer attention.
    last_layer = f"layer_{len(model.fusion_attn_layers) - 1}"
    attn = hook_cache[last_layer][0]  # [heads, seq_img, seq_text]
    mean_attn = attn.mean(dim=0)  # [seq_img, seq_text]

    token_ids = tokens["input_ids"][0].tolist()
    attn_mask = tokens["attention_mask"][0].tolist()
    token_scores = mean_attn.max(dim=0).values  # importance of each text token
    token_indices = _select_token_indices(
        token_ids=token_ids,
        attention_mask=attn_mask,
        tokenizer=tokenizer,
        token_scores=token_scores,
        top_k=cfg.top_tokens_per_example,
    )

    token_strs = tokenizer.convert_ids_to_tokens(token_ids)
    overlay_dir = output_dir / "token_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    ncols = max(1, len(token_indices))
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
    if ncols == 1:
        axes = [axes]

    token_maps = []
    for ax, tok_idx in zip(axes, token_indices):
        heat = _normalized_heatmap(mean_attn[:, tok_idx], grid_size=14)
        token = token_strs[tok_idx]
        _overlay_heatmap(ax, image, heat, title=f"token={token}")
        token_maps.append({"token_index": int(tok_idx), "token": token, "heatmap": heat})

    fig.suptitle(
        f"Example {example_idx:02d} | true={row['true_label']} | pred={pred_label} ({pred_conf:.2f})",
        fontsize=11,
    )
    fig.tight_layout()
    overlay_path = overlay_dir / f"example_{example_idx:02d}.png"
    fig.savefig(overlay_path, dpi=150)
    plt.close(fig)

    # Class-gallery map: average over valid non-special text tokens.
    valid_token_indices = _select_token_indices(
        token_ids=token_ids,
        attention_mask=attn_mask,
        tokenizer=tokenizer,
        token_scores=torch.ones_like(token_scores),
        top_k=int(sum(attn_mask)),
    )
    if valid_token_indices:
        class_map = mean_attn[:, valid_token_indices].mean(dim=1)
    else:
        class_map = mean_attn.mean(dim=1)

    return {
        "example_index": example_idx,
        "source_index": int(row["source_index"]),
        "image_path": row["image_path"],
        "hypothesis": row["hypothesis"],
        "true_label": row["true_label"],
        "pred_label": pred_label,
        "pred_confidence": pred_conf,
        "difficulty": row["difficulty"],
        "attention_tensors": saved_tensors,
        "overlay_path": str(overlay_path),
        "class_heatmap": _normalized_heatmap(class_map, grid_size=14),
    }


def build_class_galleries(
    records: List[Dict[str, object]],
    output_dir: Path,
) -> Dict[str, str]:
    gallery_paths = {}
    for label in ("entailment", "neutral", "contradiction"):
        subset = [r for r in records if r["pred_label"] == label]
        if not subset:
            continue

        n = len(subset)
        cols = min(4, n)
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = np.array(axes).reshape(rows, cols)

        for ax in axes.flatten():
            ax.axis("off")

        for i, rec in enumerate(subset):
            ax = axes.flatten()[i]
            image = _open_rgb(str(rec["image_path"]))
            #_overlay_heatmap(
                #ax=ax,
                #image=image,
                #heat=rec["class_heatmap"],
                #title=f"ex={rec['example_index']:02d} {rec['difficulty']} {rec['pred_confidence']:.2f}",
            #)
            _overlay_heatmap(
                ax=ax,
                image=image,
                heat=rec["class_heatmap"],
                title=f"ex={rec['example_index']:02d} {rec['difficulty']} {rec['pred_confidence']:.2f}",
                text=rec["hypothesis"]  # Add this line!
            )

        fig.suptitle(f"Predicted class gallery: {label}", fontsize=12)
        fig.tight_layout()
        out_path = output_dir / f"class_gallery_{label}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        gallery_paths[label] = str(out_path)

    return gallery_paths


def run_task2(config: Optional[Task2Config] = None) -> Dict[str, object]:
    cfg = config or Task2Config()
    set_seed(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(cfg.test_csv)
    device = pick_device()
    print(f"Using device: {device}")

    processor = _load_processor(cfg.vision_model_name)
    tokenizer = _load_tokenizer(cfg.text_model_name)

    model = VisualEntailmentWithFusionAttention(
        vision_model_name=cfg.vision_model_name,
        text_model_name=cfg.text_model_name,
        hidden_dim=512,
        dropout_rate=0.3,
        depth=1,
        fusion_attention_layers=cfg.fusion_attention_layers,
        fusion_attention_heads=cfg.fusion_attention_heads,
    )

    ckpt_path = Path(cfg.checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint with strict=False | missing={len(missing)} unexpected={len(unexpected)}")
    model.to(device).eval()

    pool_df = build_prediction_pool(
        model=model,
        df=df,
        processor=processor,
        tokenizer=tokenizer,
        cfg=cfg,
        device=device,
    )
    pool_df.to_csv(out_dir / "task2_prediction_pool.csv", index=False)

    selected_df = select_20_examples(pool_df, num_examples=cfg.num_examples, seed=cfg.seed)
    selected_df.to_csv(out_dir / "task2_selected_examples.csv", index=False)
    if selected_df.empty:
        raise RuntimeError("No examples selected.")

    records = []
    for i, row in selected_df.iterrows():
        rec = extract_and_visualize_example(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            device=device,
            cfg=cfg,
            row=row,
            example_idx=i,
            output_dir=out_dir,
        )
        records.append(rec)

    gallery_paths = build_class_galleries(records, out_dir)

    # Flatten a CSV index of saved attention tensors.
    tensor_rows = []
    for rec in records:
        for layer_name, path in rec["attention_tensors"].items():
            tensor_rows.append(
                {
                    "example_index": rec["example_index"],
                    "source_index": rec["source_index"],
                    "layer_name": layer_name,
                    "tensor_path": path,
                    "pred_label": rec["pred_label"],
                    "true_label": rec["true_label"],
                    "difficulty": rec["difficulty"],
                }
            )
    tensor_index_df = pd.DataFrame(tensor_rows)
    tensor_index_df.to_csv(out_dir / "task2_attention_tensor_index.csv", index=False)

    summary = {
        "config": asdict(cfg),
        "num_selected_examples": int(len(selected_df)),
        "selected_label_counts": selected_df["true_label"].value_counts().to_dict(),
        "selected_difficulty_counts": selected_df["difficulty"].value_counts().to_dict(),
        "gallery_paths": gallery_paths,
        "artifacts": {
            "prediction_pool_csv": str(out_dir / "task2_prediction_pool.csv"),
            "selected_examples_csv": str(out_dir / "task2_selected_examples.csv"),
            "attention_tensor_index_csv": str(out_dir / "task2_attention_tensor_index.csv"),
            "token_overlay_dir": str(out_dir / "token_overlays"),
            "attention_tensor_dir": str(out_dir / "attention_tensors"),
        },
    }

    with (out_dir / "task2_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved Task 2 artifacts to: {out_dir}")
    return summary


if __name__ == "__main__":
    run_task2()
