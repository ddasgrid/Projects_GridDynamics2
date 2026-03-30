from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from app_backbones import list_backbone_checkpoints
from task2_cross_attention_viz import (
    ID2LABEL,
    Task2Config,
    VisualEntailmentWithFusionAttention,
    _load_processor,
    _load_tokenizer,
    _normalized_heatmap,
    pick_device,
    register_attention_hooks,
)


DEFAULT_ARTIFACT_DIR = "task2_artifacts"
DEFAULT_SUMMARY = "task2_artifacts/task2_summary.json"
DEFAULT_SELECTED = "task2_artifacts/task2_selected_examples.csv"
DEFAULT_TENSOR_INDEX = "task2_artifacts/task2_attention_tensor_index.csv"
DEFAULT_CHECKPOINT = "final_sota_visual_entailment3.pth"

BADGE_CLASS_MAP = {
    "entailment": "ve-badge-entailment",
    "neutral": "ve-badge-neutral",
    "contradiction": "ve-badge-contradiction",
}


def apply_minimal_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 12% 10%, rgba(56, 189, 248, 0.12), transparent 30%),
                radial-gradient(circle at 88% 20%, rgba(20, 184, 166, 0.1), transparent 28%),
                linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%);
            font-family: "Avenir Next", "Segoe UI", "Trebuchet MS", sans-serif;
        }
        .main .block-container {
            max-width: 1100px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .ve-title {
            margin-bottom: 0.2rem;
            font-size: clamp(1.6rem, 3vw, 2.2rem);
            color: #111827;
            font-weight: 800;
            letter-spacing: -0.02em;
            animation: ve-rise 420ms ease-out both;
        }
        .ve-title-emphasis {
            background: linear-gradient(90deg, #0f766e, #2563eb);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }
        .ve-subtitle {
            margin-bottom: 1.5rem;
            color: #334155;
            font-size: 0.96rem;
            max-width: 72ch;
            animation: ve-rise 620ms ease-out both;
        }
        div[data-testid="column"] div[data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(255, 255, 255, 0.86);
            border: 1px solid #dbe2ea;
            border-radius: 14px;
            padding: 0.45rem 0.55rem;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
            backdrop-filter: blur(4px);
            transition: transform 160ms ease, box-shadow 160ms ease;
            animation: ve-rise 460ms ease-out both;
        }
        div[data-testid="column"] div[data-testid="stVerticalBlockBorderWrapper"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 14px 28px rgba(15, 23, 42, 0.1);
        }
        .ve-card-title {
            margin: 0 0 0.15rem 0;
            color: #111827;
            font-size: 1.1rem;
            font-weight: 700;
        }
        .ve-card-subtitle {
            margin: 0 0 0.9rem 0;
            color: #475569;
            font-size: 0.9rem;
        }
        .ve-badge {
            display: inline-block;
            border-radius: 999px;
            padding: 0.25rem 0.65rem;
            font-size: 0.78rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            border: 1px solid transparent;
        }
        .ve-badge-entailment {
            color: #166534;
            background: #dcfce7;
            border-color: #86efac;
        }
        .ve-badge-neutral {
            color: #854d0e;
            background: #fef9c3;
            border-color: #fde047;
        }
        .ve-badge-contradiction {
            color: #991b1b;
            background: #fee2e2;
            border-color: #fca5a5;
        }
        .ve-image-wrap {
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            overflow: hidden;
            margin: 0.6rem 0 0.9rem 0;
        }
        [data-testid="stFileUploader"] section {
            border-radius: 10px;
            border: 1px dashed #c6d3e1;
            background: #f8fbff;
        }
        .stButton > button {
            border-radius: 10px;
            font-weight: 700;
            border: 1px solid #c7d2fe;
            transition: transform 120ms ease, box-shadow 120ms ease;
        }
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 8px 16px rgba(37, 99, 235, 0.18);
        }
        .stTextInput > div > div > input {
            background: #fbfdff;
        }
        @keyframes ve-rise {
            from {
                opacity: 0;
                transform: translateY(6px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _list_checkpoints() -> list[str]:
    return list_backbone_checkpoints()


def _read_json(path: str) -> Optional[Dict[str, object]]:
    p = Path(path)
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_csv(path: str) -> Optional[pd.DataFrame]:
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    return df if not df.empty else None


def _resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (Path.cwd() / p)


def _valid_token_indices(token_ids: list[int], attn_mask: list[int], tokenizer) -> list[int]:
    valid_len = int(sum(attn_mask))
    special_ids = set(tokenizer.all_special_ids)
    out: list[int] = []
    for idx in range(valid_len):
        if int(token_ids[idx]) in special_ids:
            continue
        out.append(idx)
    return out


def _overlay_heat_on_image(image: Image.Image, heat_14x14: np.ndarray, alpha: float = 0.45) -> Image.Image:
    heat_img = Image.fromarray((heat_14x14 * 255).astype(np.uint8)).resize(image.size, resample=Image.BILINEAR)
    heat = np.asarray(heat_img, dtype=np.float32) / 255.0

    # Lightweight JET-like map without depending on mpl rendering.
    r = np.clip(1.5 - np.abs(4.0 * heat - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * heat - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * heat - 1.0), 0.0, 1.0)
    color = np.stack([r, g, b], axis=-1)

    base = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    blended = np.clip((1.0 - alpha) * base + alpha * color, 0.0, 1.0)
    return Image.fromarray((blended * 255).astype(np.uint8))


def _heatmaps_from_attention(
    layer_tensor: np.ndarray,
    valid_token_indices: list[int],
    token_index: int,
) -> Tuple[np.ndarray, np.ndarray]:
    # layer_tensor: [heads, 196, seq_text]
    t = torch.tensor(layer_tensor, dtype=torch.float32)
    mean_attn = t.mean(dim=0)  # [196, seq_text]
    token_heat = _normalized_heatmap(mean_attn[:, token_index], grid_size=14)
    if valid_token_indices:
        overall_vec = mean_attn[:, valid_token_indices].mean(dim=1)
    else:
        overall_vec = mean_attn.mean(dim=1)
    overall_heat = _normalized_heatmap(overall_vec, grid_size=14)
    return token_heat, overall_heat


def _layer_display_name(layer_name: str) -> str:
    if layer_name.startswith("layer_"):
        try:
            idx = int(layer_name.split("_")[1])
            return f"Fusion attention layer {idx + 1} ({layer_name})"
        except Exception:
            return layer_name
    return layer_name


@st.cache_resource
def load_task2_pipeline(
    checkpoint_path: str,
    vision_model_name: str,
    text_model_name: str,
    fusion_attention_layers: int,
    fusion_attention_heads: int,
):
    device = pick_device()
    processor = _load_processor(vision_model_name)
    tokenizer = _load_tokenizer(text_model_name)

    model = VisualEntailmentWithFusionAttention(
        vision_model_name=vision_model_name,
        text_model_name=text_model_name,
        hidden_dim=512,
        dropout_rate=0.3,
        depth=1,
        fusion_attention_layers=fusion_attention_layers,
        fusion_attention_heads=fusion_attention_heads,
    )
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model, processor, tokenizer, device


@torch.no_grad()
def run_custom_attention(
    model: VisualEntailmentWithFusionAttention,
    processor,
    tokenizer,
    device: torch.device,
    image: Image.Image,
    hypothesis: str,
    max_text_length: int,
) -> Dict[str, object]:
    pixel_values = processor(images=[image], return_tensors="pt")["pixel_values"]
    toks = tokenizer(
        [hypothesis],
        padding=True,
        truncation=True,
        max_length=max_text_length,
        return_tensors="pt",
    )

    cache, handles = register_attention_hooks(model)
    try:
        logits = model(
            pixel_values.to(device),
            toks["input_ids"].to(device),
            toks["attention_mask"].to(device),
        )
        probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
    finally:
        for h in handles:
            h.remove()

    pred_idx = int(np.argmax(probs))
    token_ids = toks["input_ids"][0].tolist()
    attn_mask = toks["attention_mask"][0].tolist()
    token_strs = tokenizer.convert_ids_to_tokens(token_ids)
    valid_indices = _valid_token_indices(token_ids, attn_mask, tokenizer)

    layer_tensors = {name: tensor[0].detach().cpu().numpy() for name, tensor in cache.items()}
    return {
        "pred_label": ID2LABEL[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probs": probs,
        "layer_tensors": layer_tensors,
        "token_ids": token_ids,
        "attn_mask": attn_mask,
        "tokens": token_strs,
        "valid_indices": valid_indices,
    }


def _render_probs(probs: np.ndarray) -> None:
    st.markdown("**Class probabilities**")
    for i, p in enumerate(probs.tolist()):
        st.progress(float(p), text=f"{ID2LABEL[i].capitalize()}: {100.0 * float(p):.1f}%")


def render_saved_heatmaps_tab(
    artifact_dir: str,
    summary_path: str,
    selected_csv: str,
    tensor_index_csv: str,
) -> None:
    summary = _read_json(summary_path)
    selected_df = _read_csv(selected_csv)
    tensor_idx = _read_csv(tensor_index_csv)

    if selected_df is None:
        st.warning(f"Could not find selected examples file: `{selected_csv}`")
        return

    cfg = Task2Config()
    if summary and isinstance(summary.get("config"), dict):
        cfg.max_text_length = int(summary["config"].get("max_text_length", cfg.max_text_length))
        cfg.text_model_name = str(summary["config"].get("text_model_name", cfg.text_model_name))
        cfg.top_tokens_per_example = int(
            summary["config"].get("top_tokens_per_example", cfg.top_tokens_per_example)
        )

    tokenizer = _load_tokenizer(cfg.text_model_name)

    with st.container(border=True):
        st.markdown("<h2 class='ve-card-title'>Saved Example Browser</h2>", unsafe_allow_html=True)
        st.markdown(
            "<p class='ve-card-subtitle'>Choose one of your original 20 image-text pairs and inspect precomputed + dynamic attention maps.</p>",
            unsafe_allow_html=True,
        )

        options = []
        for i, row in selected_df.iterrows():
            label = f"ex {i:02d} | {row['true_label']} | {row['difficulty']} | {str(row['hypothesis'])[:55]}"
            options.append((i, label))
        selected_idx = st.selectbox("Select example", options=options, format_func=lambda x: x[1])[0]

    row = selected_df.iloc[int(selected_idx)]
    image_path = _resolve_path(str(row["image_path"]))
    if not image_path.exists():
        st.error(f"Image not found: `{image_path}`")
        return

    image = Image.open(image_path).convert("RGB")

    # Pre-generated overlay image path.
    pre_overlay = _resolve_path(f"{artifact_dir}/token_overlays/example_{int(selected_idx):02d}.png")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        with st.container(border=True):
            st.markdown("<h2 class='ve-card-title'>Input Pair</h2>", unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.caption(
                f"True: `{row['true_label']}` | Pred: `{row['pred_label']}` | "
                f"Conf: {100.0 * float(row['pred_confidence']):.2f}% | Difficulty: `{row['difficulty']}`"
            )
            st.write(f"Hypothesis: {row['hypothesis']}")

    with c2:
        with st.container(border=True):
            st.markdown("<h2 class='ve-card-title'>Pre-generated Token Overlay</h2>", unsafe_allow_html=True)
            st.caption(
                f"Generated from top {cfg.top_tokens_per_example} tokens with highest attention scores "
                "for the selected example."
            )
            if pre_overlay.exists():
                st.image(str(pre_overlay), use_container_width=True)
            else:
                st.info("No pre-generated token overlay found for this example.")

    if tensor_idx is None:
        st.warning(f"Could not find tensor index CSV: `{tensor_index_csv}`")
        return

    rows = tensor_idx[tensor_idx["example_index"] == int(selected_idx)].copy()
    if rows.empty:
        st.info("No saved attention tensors found for this example.")
        return

    layer_to_path = {str(r["layer_name"]): str(r["tensor_path"]) for _, r in rows.iterrows()}
    layer_name = st.selectbox(
        "Layer",
        options=sorted(layer_to_path.keys()),
        format_func=_layer_display_name,
    )
    tensor_path = _resolve_path(layer_to_path[layer_name])
    if not tensor_path.exists():
        st.error(f"Tensor file missing: `{tensor_path}`")
        return

    layer_tensor = np.load(tensor_path)  # [heads, 196, seq_text]

    text = str(row["hypothesis"])
    toks = tokenizer([text], padding=True, truncation=True, max_length=cfg.max_text_length, return_tensors="pt")
    token_ids = toks["input_ids"][0].tolist()
    attn_mask = toks["attention_mask"][0].tolist()
    token_strs = tokenizer.convert_ids_to_tokens(token_ids)
    valid_indices = _valid_token_indices(token_ids, attn_mask, tokenizer)

    if not valid_indices:
        st.warning("No valid text tokens found after tokenization.")
        return

    token_options = [(idx, f"{idx}: {token_strs[idx]}") for idx in valid_indices]
    selected_token_idx = st.selectbox("Token", options=token_options, format_func=lambda x: x[1])[0]

    token_heat, overall_heat = _heatmaps_from_attention(layer_tensor, valid_indices, int(selected_token_idx))
    token_overlay = _overlay_heat_on_image(image, token_heat)
    overall_overlay = _overlay_heat_on_image(image, overall_heat)

    h1, h2 = st.columns(2, gap="large")
    with h1:
        with st.container(border=True):
            st.markdown("<h2 class='ve-card-title'>Token-wise Heatmap</h2>", unsafe_allow_html=True)
            st.image(token_overlay, use_container_width=True)
            st.caption(f"Layer `{layer_name}` | Token `{token_strs[int(selected_token_idx)]}`")
    with h2:
        with st.container(border=True):
            st.markdown("<h2 class='ve-card-title'>Overall Heatmap</h2>", unsafe_allow_html=True)
            st.image(overall_overlay, use_container_width=True)
            st.caption(f"Layer `{layer_name}` | Averaged over all valid tokens")

def render_custom_heatmap_tab(summary_path: str, checkpoints: Optional[list[str]] = None) -> None:
    summary = _read_json(summary_path)

    cfg = Task2Config()
    if summary and isinstance(summary.get("config"), dict):
        c = summary["config"]
        cfg.checkpoint_path = str(c.get("checkpoint_path", cfg.checkpoint_path))
        cfg.vision_model_name = str(c.get("vision_model_name", cfg.vision_model_name))
        cfg.text_model_name = str(c.get("text_model_name", cfg.text_model_name))
        cfg.max_text_length = int(c.get("max_text_length", cfg.max_text_length))
        cfg.fusion_attention_layers = int(c.get("fusion_attention_layers", cfg.fusion_attention_layers))
        cfg.fusion_attention_heads = int(c.get("fusion_attention_heads", cfg.fusion_attention_heads))

    checkpoints = checkpoints if checkpoints is not None else _list_checkpoints()
    if checkpoints:
        default_idx = checkpoints.index(DEFAULT_CHECKPOINT) if DEFAULT_CHECKPOINT in checkpoints else 0
    else:
        default_idx = 0

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        with st.container(border=True):
            st.markdown("<h2 class='ve-card-title'>1. Input</h2>", unsafe_allow_html=True)
            st.markdown(
                "<p class='ve-card-subtitle'>Upload your own image-text pair to generate token-wise and overall cross-attention heatmaps.</p>",
                unsafe_allow_html=True,
            )

            if checkpoints:
                checkpoint = st.selectbox("Checkpoint", checkpoints, index=default_idx)
            else:
                checkpoint = st.text_input("Checkpoint path", value=cfg.checkpoint_path)

            max_text_length = st.slider("Max text length", min_value=16, max_value=128, value=int(cfg.max_text_length), step=8)
            uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"], key="task2_custom_uploader")
            hypothesis = st.text_input("Hypothesis", placeholder="Example: A person is riding a bicycle.")

            if uploaded_file is not None:
                preview = Image.open(uploaded_file).convert("RGB")
                st.markdown("<div class='ve-image-wrap'>", unsafe_allow_html=True)
                st.image(preview, caption="Uploaded image preview", use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            run_btn = st.button("Generate Attention Heatmaps", use_container_width=True, type="primary")

            if run_btn:
                if uploaded_file is None or not hypothesis.strip():
                    st.session_state["task2_custom_result"] = {"error": "Please provide both image and hypothesis."}
                else:
                    try:
                        with st.spinner("Loading model + computing attention..."):
                            model, processor, tokenizer, device = load_task2_pipeline(
                                checkpoint_path=checkpoint,
                                vision_model_name=cfg.vision_model_name,
                                text_model_name=cfg.text_model_name,
                                fusion_attention_layers=cfg.fusion_attention_layers,
                                fusion_attention_heads=cfg.fusion_attention_heads,
                            )
                            image = Image.open(uploaded_file).convert("RGB")
                            out = run_custom_attention(
                                model=model,
                                processor=processor,
                                tokenizer=tokenizer,
                                device=device,
                                image=image,
                                hypothesis=hypothesis.strip(),
                                max_text_length=max_text_length,
                            )
                        st.session_state["task2_custom_result"] = {
                            "image": image,
                            "hypothesis": hypothesis.strip(),
                            "output": out,
                        }
                    except Exception as exc:
                        st.session_state["task2_custom_result"] = {"error": str(exc)}

    with col2:
        with st.container(border=True):
            st.markdown("<h2 class='ve-card-title'>2. Heatmaps</h2>", unsafe_allow_html=True)
            st.markdown(
                "<p class='ve-card-subtitle'>Inspect predicted label, probabilities, and attention overlays for your input.</p>",
                unsafe_allow_html=True,
            )

            state = st.session_state.get("task2_custom_result")
            if not state:
                st.info("Run generation to view custom heatmaps.")
                return
            if "error" in state:
                st.error(state["error"])
                return

            image = state["image"]
            out = state["output"]
            label = str(out["pred_label"])
            badge_class = BADGE_CLASS_MAP.get(label, "ve-badge-neutral")
            st.markdown(f"<span class='ve-badge {badge_class}'>{label.capitalize()}</span>", unsafe_allow_html=True)
            st.metric("Confidence", f"{100.0 * float(out['confidence']):.2f}%")
            _render_probs(np.array(out["probs"]))
            st.image(image, caption="Input image", use_container_width=True)

            layer_names = sorted(out["layer_tensors"].keys())
            if not layer_names:
                st.warning("No attention tensors captured from hooks.")
                return

            layer_name = st.selectbox(
                "Layer",
                options=layer_names,
                format_func=_layer_display_name,
                key="task2_custom_layer",
            )
            valid_indices = list(out["valid_indices"])
            tokens = list(out["tokens"])
            if not valid_indices:
                st.warning("No valid tokens available for token-wise map.")
                return

            token_options = [(idx, f"{idx}: {tokens[idx]}") for idx in valid_indices]
            selected_token_idx = st.selectbox("Token", options=token_options, format_func=lambda x: x[1], key="task2_custom_token")[0]

            token_heat, overall_heat = _heatmaps_from_attention(
                out["layer_tensors"][layer_name],
                valid_indices,
                int(selected_token_idx),
            )
            token_overlay = _overlay_heat_on_image(image, token_heat)
            overall_overlay = _overlay_heat_on_image(image, overall_heat)

            h1, h2 = st.columns(2)
            with h1:
                st.image(token_overlay, caption=f"Token: {tokens[int(selected_token_idx)]}", use_container_width=True)
            with h2:
                st.image(overall_overlay, caption="Overall token average", use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Visual Entailment", page_icon="👁️", layout="wide")
    apply_minimal_styles()

    st.markdown(
        "<h1 class='ve-title'>Task 2 <span class='ve-title-emphasis'>Cross-Attention</span> Viewer</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p class='ve-subtitle'>Browse your generated attention heatmaps and create token-wise/overall heatmaps for new image-text pairs.</p>",
        unsafe_allow_html=True,
    )

    tab_saved, tab_custom = st.tabs(["Saved Heatmaps", "Custom Heatmap Generator"])

    with tab_saved:
        c1, c2 = st.columns([1, 1], gap="large")
        with c1:
            artifact_dir = st.text_input("Artifact directory", value=DEFAULT_ARTIFACT_DIR)
            selected_csv = st.text_input("Selected examples CSV", value=DEFAULT_SELECTED)
        with c2:
            summary_path = st.text_input("Summary JSON", value=DEFAULT_SUMMARY)
            tensor_index_csv = st.text_input("Tensor index CSV", value=DEFAULT_TENSOR_INDEX)

        render_saved_heatmaps_tab(
            artifact_dir=artifact_dir,
            summary_path=summary_path,
            selected_csv=selected_csv,
            tensor_index_csv=tensor_index_csv,
        )

    with tab_custom:
        summary_path_custom = st.text_input("Summary JSON (for default config)", value=DEFAULT_SUMMARY, key="task2_summary_custom")
        render_custom_heatmap_tab(summary_path=summary_path_custom)


if __name__ == "__main__":
    main()
