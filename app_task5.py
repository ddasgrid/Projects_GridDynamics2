from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from app_backbones import list_backbone_checkpoints
from task4_pipeline import analyze_concept
from task5_adversarial import (
    ID2LABEL,
    TEXT_ATTACK_MODES,
    VisualEntailmentModel,
    _load_processor,
    _load_tokenizer,
    _norm_params,
    evaluate_dataframe_accuracy,
    fgsm_attack,
    generate_text_adversary,
    pgd_attack,
    pick_device,
)


DEFAULT_CHECKPOINT = "final_sota_visual_entailment3.pth"
DEFAULT_ARTIFACT_DIR = "task5_artifacts"
TEXT_CURVE_FILENAME = "task5_text_attack_accuracy_by_mode.csv"

BADGE_CLASS_MAP = {
    "entailment": "ve-badge-entailment",
    "neutral": "ve-badge-neutral",
    "contradiction": "ve-badge-contradiction",
}

CONCEPT_TO_ATTACK = {
    "negation": "negation",
    "object_or_attribute": "object_substitution",
    "action": "paraphrase",
    "spatial": "synonym_substitution",
    "count": "synonym_substitution",
}


def _list_checkpoints() -> list[str]:
    checkpoints = list_backbone_checkpoints()
    return [ckpt for ckpt in checkpoints if Path(ckpt).name == DEFAULT_CHECKPOINT]


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
        @media (max-width: 900px) {
            div[data-testid="column"] div[data-testid="stVerticalBlockBorderWrapper"] {
                min-height: auto;
            }
        }
        @media (prefers-reduced-motion: reduce) {
            .ve-title,
            .ve-subtitle,
            div[data-testid="column"] div[data-testid="stVerticalBlockBorderWrapper"] {
                animation: none;
            }
            div[data-testid="column"] div[data-testid="stVerticalBlockBorderWrapper"],
            .stButton > button {
                transition: none;
            }
            div[data-testid="column"] div[data-testid="stVerticalBlockBorderWrapper"]:hover,
            .stButton > button:hover {
                transform: none;
                box-shadow: none;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_pipeline(
    checkpoint_path: str,
    vision_model_name: str = "google/vit-base-patch16-224",
    text_model_name: str = "bert-base-uncased",
) -> Tuple[VisualEntailmentModel, object, object, torch.device, torch.Tensor, torch.Tensor]:
    device = pick_device()
    model = VisualEntailmentModel(
        vision_model_name=vision_model_name,
        text_model_name=text_model_name,
        hidden_dim=512,
        dropout_rate=0.3,
        depth=1,
        fusion_type="attention",
        freeze_mode="full",
        num_layers_to_freeze=12,
    )
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    processor = _load_processor(vision_model_name)
    tokenizer = _load_tokenizer(text_model_name)
    mean, std = _norm_params(processor, device=device)
    return model, processor, tokenizer, device, mean, std


def _prepare_single_image(processor, image: Image.Image) -> torch.Tensor:
    return processor(images=[image], return_tensors="pt")["pixel_values"]


def _prepare_single_text(tokenizer, text: str, max_len: int = 64) -> Dict[str, torch.Tensor]:
    return tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )


def _predict(
    model: VisualEntailmentModel,
    pixel_values: torch.Tensor,
    toks: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, object]:
    with torch.no_grad():
        logits = model(
            pixel_values.to(device),
            toks["input_ids"].to(device),
            toks["attention_mask"].to(device),
        )
        probs = torch.softmax(logits, dim=-1)[0].detach().cpu()
    pred_idx = int(torch.argmax(probs).item())
    return {
        "pred_idx": pred_idx,
        "pred_label": ID2LABEL[pred_idx],
        "confidence": float(torch.max(probs).item()),
        "probs": probs.numpy(),
    }


def _to_pil(pixel_values_norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> Image.Image:
    x = pixel_values_norm.detach().cpu()
    mean_cpu = mean.detach().cpu()
    std_cpu = std.detach().cpu()
    img = (x * std_cpu + mean_cpu).clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0).numpy()
    arr = (img * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def _render_probabilities(title: str, probs: np.ndarray) -> None:
    st.markdown(f"**{title}**")
    for idx, p in enumerate(probs.tolist()):
        label = ID2LABEL[idx].capitalize()
        st.progress(float(p), text=f"{label}: {100.0 * float(p):.1f}%")


def apply_text_attack_with_concept(
    text: str,
    seed: int = 42,
) -> Dict[str, str]:
    concept = str(analyze_concept(text).get("primary_concept", "object_or_attribute"))
    mapped_mode = CONCEPT_TO_ATTACK.get(concept, "paraphrase")

    attacked_text, applied_mode, anchor = generate_text_adversary(
        text=text,
        mode=mapped_mode,
        rng=random.Random(seed),
    )
    return {
        "concept_type": concept,
        "mapped_mode": mapped_mode,
        "applied_mode": applied_mode,
        "anchor": anchor,
        "attacked_text": attacked_text,
    }


def json_load(path: Path) -> Dict[str, object]:
    import json

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_previous_results(artifact_dir: str) -> Optional[Dict[str, object]]:
    summary_path = Path(artifact_dir) / "task5_robustness_summary.json"
    if not summary_path.exists():
        return None
    return json_load(summary_path)


def _load_image_attack_df(artifact_dir: str, summary: Optional[Dict[str, object]]) -> Optional[pd.DataFrame]:
    image_results_path = Path(artifact_dir) / "task5_image_attack_results.csv"
    if image_results_path.exists():
        df = pd.read_csv(image_results_path)
    elif summary is not None and isinstance(summary.get("image_attack_results"), list):
        df = pd.DataFrame(summary["image_attack_results"])
    else:
        return None

    if df.empty:
        return None

    df = df.copy()
    df["epsilon"] = df["epsilon"].astype(float)
    df["epsilon_label"] = df["epsilon"].map(lambda e: f"{int(round(e * 255))}/255")
    return df.sort_values(["attack_method", "epsilon"]).reset_index(drop=True)


def _compute_text_attack_curve(checkpoint: str, artifact_dir: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    text_pairs_path = Path(artifact_dir) / "task5_text_adversarial_pairs.csv"
    if not text_pairs_path.exists():
        return None, f"Missing file: `{text_pairs_path}`"

    text_df = pd.read_csv(text_pairs_path)
    required_cols = {
        "image_path",
        "clean_hypothesis",
        "adversarial_hypothesis",
        "attack_mode",
        "expected_label",
    }
    if not required_cols.issubset(text_df.columns):
        return None, "Text adversarial CSV is missing required columns for curve generation."

    try:
        model, processor, tokenizer, device, _, _ = load_pipeline(checkpoint)
    except Exception as exc:
        return None, f"Could not load checkpoint `{checkpoint}`: {exc}"

    rows = []
    for mode in sorted(text_df["attack_mode"].dropna().unique().tolist()):
        subset = text_df[text_df["attack_mode"] == mode].copy()
        if subset.empty:
            continue

        clean_eval_df = pd.DataFrame(
            {
                "image_path": subset["image_path"].astype(str).tolist(),
                "hypothesis": subset["clean_hypothesis"].astype(str).tolist(),
                "label": ["entailment"] * len(subset),
            }
        )
        adv_eval_df = pd.DataFrame(
            {
                "image_path": subset["image_path"].astype(str).tolist(),
                "hypothesis": subset["adversarial_hypothesis"].astype(str).tolist(),
                "label": subset["expected_label"].astype(str).str.lower().tolist(),
            }
        )

        clean_stats = evaluate_dataframe_accuracy(
            model=model,
            df=clean_eval_df,
            processor=processor,
            tokenizer=tokenizer,
            device=device,
            batch_size=8,
            max_text_length=64,
        )
        adv_stats = evaluate_dataframe_accuracy(
            model=model,
            df=adv_eval_df,
            processor=processor,
            tokenizer=tokenizer,
            device=device,
            batch_size=8,
            max_text_length=64,
        )

        rows.append(
            {
                "attack_mode": mode,
                "samples": int(len(subset)),
                "clean_accuracy": float(clean_stats["accuracy"]),
                "adversarial_accuracy": float(adv_stats["accuracy"]),
                "accuracy_drop": float(clean_stats["accuracy"] - adv_stats["accuracy"]),
            }
        )

    if not rows:
        return None, "No rows were available to compute text attack curves."

    curve_df = pd.DataFrame(rows).sort_values("samples", ascending=False).reset_index(drop=True)
    curve_path = Path(artifact_dir) / TEXT_CURVE_FILENAME
    curve_df.to_csv(curve_path, index=False)
    return curve_df, None


def _load_text_attack_curve(artifact_dir: str) -> Optional[pd.DataFrame]:
    curve_path = Path(artifact_dir) / TEXT_CURVE_FILENAME
    if not curve_path.exists():
        return None
    df = pd.read_csv(curve_path)
    return df if not df.empty else None


def _render_robustness_report(artifact_dir: str, checkpoint: Optional[str]) -> None:
    summary = _load_previous_results(artifact_dir)
    if summary is None:
        st.warning(f"No summary found at `{artifact_dir}/task5_robustness_summary.json`.")
        return

    with st.container(border=True):
        st.markdown("<h2 class='ve-card-title'>Robustness Summary</h2>", unsafe_allow_html=True)
        st.markdown(
            "<p class='ve-card-subtitle'>Clean vs adversarial performance from your Task 5 evaluation.</p>",
            unsafe_allow_html=True,
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Clean test acc", f"{100 * summary['clean_test_accuracy']:.2f}%")
        c2.metric("Text adv acc", f"{100 * summary['text_adversarial_accuracy']:.2f}%")
        c3.metric("Combined adv acc", f"{100 * summary['combined_adversarial_accuracy']:.2f}%")
        c4.metric("Combined flip rate", f"{100 * summary['combined_flip_rate']:.2f}%")

    image_df = _load_image_attack_df(artifact_dir, summary)
    text_curve_df = _load_text_attack_curve(artifact_dir)

    graph_col1, graph_col2 = st.columns(2, gap="large")

    with graph_col1:
        with st.container(border=True):
            st.markdown("<h2 class='ve-card-title'>Image Noise vs Accuracy</h2>", unsafe_allow_html=True)
            st.markdown(
                "<p class='ve-card-subtitle'>Accuracy trend across epsilon budgets for FGSM and PGD.</p>",
                unsafe_allow_html=True,
            )
            if image_df is None:
                st.info("No image attack results available.")
            else:
                acc_curve = (
                    image_df.pivot(index="epsilon", columns="attack_method", values="accuracy")
                    .sort_index()
                    .mul(100.0)
                )
                acc_curve.index = [f"{int(round(e * 255))}/255" for e in acc_curve.index]
                st.line_chart(acc_curve, use_container_width=True)

                success_curve = (
                    image_df.pivot(index="epsilon", columns="attack_method", values="attack_success_rate")
                    .sort_index()
                    .mul(100.0)
                )
                success_curve.index = [f"{int(round(e * 255))}/255" for e in success_curve.index]
                st.caption("Attack success rate (%)")
                st.line_chart(success_curve, use_container_width=True)

    with graph_col2:
        with st.container(border=True):
            st.markdown("<h2 class='ve-card-title'>Text Noise vs Accuracy</h2>", unsafe_allow_html=True)
            st.markdown(
                "<p class='ve-card-subtitle'>Accuracy change by text perturbation mode (negation, swap, paraphrase).</p>",
                unsafe_allow_html=True,
            )

            if text_curve_df is None:
                st.info("Text attack accuracy curve is not generated yet.")
                if checkpoint is None:
                    st.warning("Select a checkpoint to generate the text-noise accuracy graph.")
                else:
                    if st.button("Generate text-noise accuracy graph", use_container_width=True):
                        with st.spinner("Computing text attack accuracy by mode..."):
                            text_curve_df, err = _compute_text_attack_curve(checkpoint, artifact_dir)
                        if err:
                            st.error(err)
                        else:
                            st.success(f"Saved `{TEXT_CURVE_FILENAME}` in `{artifact_dir}`.")
            if text_curve_df is not None:
                plot_df = text_curve_df.copy()
                comp_df = plot_df.set_index("attack_mode")[["clean_accuracy", "adversarial_accuracy"]].mul(100.0)
                st.bar_chart(comp_df, use_container_width=True)

                drop_df = plot_df.set_index("attack_mode")[["accuracy_drop"]].mul(100.0)
                drop_df.columns = ["accuracy_drop_pct"]
                st.caption("Accuracy drop (%) from clean text to attacked text")
                st.bar_chart(drop_df, use_container_width=True)

    details_col1, details_col2 = st.columns(2, gap="large")
    token_path = Path(artifact_dir) / "task5_token_vulnerability.csv"
    region_path = Path(artifact_dir) / "task5_image_region_vulnerability.csv"
    dash_path = Path(artifact_dir) / "task5_robustness_dashboard.png"

    with details_col1:
        with st.container(border=True):
            st.markdown("<h2 class='ve-card-title'>Token Vulnerability</h2>", unsafe_allow_html=True)
            st.caption("Higher mean gradient norm means the model prediction is more sensitive to edits on that token.")
            if token_path.exists():
                st.dataframe(pd.read_csv(token_path), use_container_width=True, height=300)
            else:
                st.info("`task5_token_vulnerability.csv` not found.")

    with details_col2:
        with st.container(border=True):
            st.markdown("<h2 class='ve-card-title'>Patch Vulnerability</h2>", unsafe_allow_html=True)
            st.caption("Higher mean saliency means image patches where small perturbations most strongly affect predictions.")
            if region_path.exists():
                st.dataframe(pd.read_csv(region_path), use_container_width=True, height=300)
            else:
                st.info("`task5_image_region_vulnerability.csv` not found.")

    if dash_path.exists():
        with st.container(border=True):
            st.markdown("<h2 class='ve-card-title'>Saved Dashboard Snapshot</h2>", unsafe_allow_html=True)
            st.image(str(dash_path), use_container_width=True)


def _render_custom_test(checkpoints: list[str]) -> None:
    notice: Optional[Tuple[str, str]] = None
    result: Optional[Dict[str, object]] = None

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        with st.container(border=True):
            st.markdown("<h2 class='ve-card-title'>1. Input</h2>", unsafe_allow_html=True)
            st.markdown(
                "<p class='ve-card-subtitle'>Upload your own sample and choose image/text perturbation settings.</p>",
                unsafe_allow_html=True,
            )

            if checkpoints:
                default_idx = checkpoints.index(DEFAULT_CHECKPOINT) if DEFAULT_CHECKPOINT in checkpoints else 0
                checkpoint = st.selectbox("Model checkpoint", checkpoints, index=default_idx)
            else:
                checkpoint = None
                st.error("No `.pth` checkpoints found in current directory.")

            uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
            image = None
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert("RGB")
                st.markdown("<div class='ve-image-wrap'>", unsafe_allow_html=True)
                st.image(image, caption="Preview", use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            hypothesis = st.text_input("Hypothesis", placeholder="Example: A dog is running in the park.")

            st.markdown("**Text attack controls**")
            apply_text_attack = st.checkbox("Apply concept-aware text attack", value=False)
            if apply_text_attack:
                st.caption("Attack mode is selected automatically from detected concept type.")

            st.markdown("**Image attack controls**")
            apply_image_attack = st.checkbox("Apply image attack", value=True)
            attack_method = st.selectbox(
                "Image attack method",
                options=["fgsm", "pgd"],
                index=1,
                disabled=not apply_image_attack,
            )
            eps_preset = st.selectbox(
                "Epsilon preset",
                options=["1/255", "4/255", "8/255", "custom"],
                index=1,
                disabled=not apply_image_attack,
            )
            if eps_preset == "custom":
                epsilon = st.number_input(
                    "Custom epsilon",
                    min_value=0.0,
                    max_value=0.1,
                    value=float(4 / 255),
                    step=0.001,
                    format="%.6f",
                )
            else:
                epsilon = {"1/255": 1 / 255, "4/255": 4 / 255, "8/255": 8 / 255}[eps_preset]

            pgd_steps = st.slider(
                "PGD steps",
                min_value=1,
                max_value=40,
                value=10,
                disabled=not (apply_image_attack and attack_method == "pgd"),
            )
            pgd_step_scale = st.slider(
                "PGD step scale (alpha = epsilon * scale)",
                min_value=0.05,
                max_value=1.0,
                value=0.25,
                step=0.05,
                disabled=not (apply_image_attack and attack_method == "pgd"),
            )
            if apply_image_attack and attack_method == "fgsm":
                st.caption("FGSM is a single-step attack, so it does not use a step-size scale.")

            run_btn = st.button("Run Robustness Test", use_container_width=True, type="primary")

            if run_btn:
                if checkpoint is None:
                    notice = ("error", "Please select a valid checkpoint.")
                elif image is None or not hypothesis.strip():
                    notice = ("warning", "Please provide both an image and a hypothesis.")
                else:
                    try:
                        with st.spinner("Loading model..."):
                            model, processor, tokenizer, device, mean, std = load_pipeline(checkpoint)
                    except Exception as exc:
                        notice = ("error", f"Could not load checkpoint `{checkpoint}`: {exc}")
                    else:
                        clean_pixels = _prepare_single_image(processor, image)
                        clean_text = hypothesis.strip()
                        attacked_text = clean_text
                        text_attack_used = "none"
                        text_attack_mapped = "none"
                        text_anchor = ""
                        text_concept_type = ""

                        if apply_text_attack:
                            attack_meta = apply_text_attack_with_concept(
                                text=clean_text,
                                seed=42,
                            )
                            attacked_text = attack_meta["attacked_text"]
                            text_attack_used = attack_meta["applied_mode"]
                            text_attack_mapped = attack_meta["mapped_mode"]
                            text_anchor = attack_meta["anchor"]
                            text_concept_type = attack_meta["concept_type"]

                        clean_toks = _prepare_single_text(tokenizer, clean_text, max_len=64)
                        attacked_toks = _prepare_single_text(tokenizer, attacked_text, max_len=64)
                        clean_pred = _predict(model, clean_pixels, clean_toks, device)

                        attacked_pixels = clean_pixels.clone()
                        if apply_image_attack:
                            ids = attacked_toks["input_ids"].to(device)
                            mask = attacked_toks["attention_mask"].to(device)
                            labels = torch.tensor([clean_pred["pred_idx"]], dtype=torch.long, device=device)
                            x = clean_pixels.to(device)

                            if attack_method == "fgsm":
                                attacked_pixels, _ = fgsm_attack(
                                    model=model,
                                    x_orig=x,
                                    input_ids=ids,
                                    attention_mask=mask,
                                    labels=labels,
                                    epsilon=float(epsilon),
                                    mean=mean,
                                    std=std,
                                )
                            else:
                                alpha = float(epsilon) * float(pgd_step_scale)
                                attacked_pixels = pgd_attack(
                                    model=model,
                                    x_orig=x,
                                    input_ids=ids,
                                    attention_mask=mask,
                                    labels=labels,
                                    epsilon=float(epsilon),
                                    alpha=alpha,
                                    steps=int(pgd_steps),
                                    mean=mean,
                                    std=std,
                                )
                            attacked_pixels = attacked_pixels.detach().cpu()

                        attacked_pred = _predict(model, attacked_pixels, attacked_toks, device)

                        delta = attacked_pixels - clean_pixels
                        result = {
                            "clean_image": image,
                            "attacked_image": _to_pil(attacked_pixels, mean, std),
                            "clean_text": clean_text,
                            "attacked_text": attacked_text,
                            "text_attack_used": text_attack_used,
                            "text_attack_mapped": text_attack_mapped,
                            "text_concept_type": text_concept_type,
                            "text_anchor": text_anchor,
                            "clean_pred": clean_pred,
                            "attacked_pred": attacked_pred,
                            "apply_image_attack": apply_image_attack,
                            "attack_method": attack_method,
                            "epsilon": float(epsilon),
                            "l_inf": float(delta.abs().max().item()),
                            "l2": float(torch.norm(delta.view(-1), p=2).item()),
                        }

    with col2:
        with st.container(border=True):
            st.markdown("<h2 class='ve-card-title'>2. Verdict</h2>", unsafe_allow_html=True)
            st.markdown(
                "<p class='ve-card-subtitle'>Compare clean vs adversarial outputs and confidence shifts.</p>",
                unsafe_allow_html=True,
            )

            if notice:
                if notice[0] == "warning":
                    st.warning(notice[1])
                else:
                    st.error(notice[1])
            elif result is not None:
                adv_label = str(result["attacked_pred"]["pred_label"])
                badge_class = BADGE_CLASS_MAP.get(adv_label, "ve-badge-neutral")
                st.markdown(
                    f"<span class='ve-badge {badge_class}'>{adv_label.capitalize()}</span>",
                    unsafe_allow_html=True,
                )

                m1, m2, m3 = st.columns(3)
                m1.metric(
                    "Clean prediction",
                    str(result["clean_pred"]["pred_label"]).capitalize(),
                    f"{100 * float(result['clean_pred']['confidence']):.2f}% conf",
                )
                m2.metric(
                    "Attacked prediction",
                    str(result["attacked_pred"]["pred_label"]).capitalize(),
                    f"{100 * float(result['attacked_pred']['confidence']):.2f}% conf",
                )
                changed = result["clean_pred"]["pred_label"] != result["attacked_pred"]["pred_label"]
                m3.metric("Prediction changed?", "Yes" if changed else "No", "flip" if changed else "stable")

                img_col1, img_col2 = st.columns(2)
                with img_col1:
                    st.markdown("**Clean image**")
                    st.image(result["clean_image"], use_container_width=True)
                with img_col2:
                    st.markdown("**Attacked image**")
                    st.image(result["attacked_image"], use_container_width=True)

                st.markdown("**Text used**")
                st.write(f"- Clean: `{result['clean_text']}`")
                st.write(f"- Attacked: `{result['attacked_text']}`")
                st.write(f"- Text concept type: `{result['text_concept_type'] or 'not_applied'}`")
                st.write(f"- Concept-mapped attack: `{result['text_attack_mapped']}`")
                st.write(f"- Applied text attack: `{result['text_attack_used']}`")
                if result["text_anchor"]:
                    st.write(f"- Anchor token/rule: `{result['text_anchor']}`")

                probs_col1, probs_col2 = st.columns(2)
                with probs_col1:
                    _render_probabilities("Clean probabilities", result["clean_pred"]["probs"])
                with probs_col2:
                    _render_probabilities("Attacked probabilities", result["attacked_pred"]["probs"])

                if result["apply_image_attack"]:
                    st.caption(
                        f"Applied image attack: `{result['attack_method']}` | epsilon={result['epsilon']:.6f} | "
                        f"L_inf={result['l_inf']:.6f} | L2={result['l2']:.6f}"
                    )
            else:
                st.info("Run a custom robustness test to see prediction shifts.")


def main() -> None:
    st.set_page_config(page_title="Visual Entailment", page_icon="👁️", layout="wide")
    apply_minimal_styles()
    st.markdown(
        "<h1 class='ve-title'>Task 5 <span class='ve-title-emphasis'>Robustness</span> Explorer</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p class='ve-subtitle'>See robustness curves from your experiment and run FGSM/PGD + text adversaries on your own image-text inputs.</p>",
        unsafe_allow_html=True,
    )

    checkpoints = _list_checkpoints()

    tab_report, tab_custom = st.tabs(["Robustness Report", "Custom Robustness Test"])

    with tab_report:
        c1, c2 = st.columns([2, 1], gap="large")
        with c1:
            artifact_dir = st.text_input("Artifacts folder", value=DEFAULT_ARTIFACT_DIR)
        with c2:
            if checkpoints:
                default_idx = checkpoints.index(DEFAULT_CHECKPOINT) if DEFAULT_CHECKPOINT in checkpoints else 0
                report_checkpoint = st.selectbox("Checkpoint for text-curve eval", checkpoints, index=default_idx)
            else:
                report_checkpoint = None
                st.warning("No `.pth` checkpoints found for report-side text curve generation.")

        _render_robustness_report(artifact_dir, report_checkpoint)

    with tab_custom:
        _render_custom_test(checkpoints)


if __name__ == "__main__":
    main()
