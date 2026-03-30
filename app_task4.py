from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

from app_backbones import list_backbone_checkpoints
from task4_pipeline import (
    AdapterClassifier,
    EncoderConfig,
    FrozenViTTextEncoder,
    ID2LABEL,
    LABEL2ID,
    PROMPT_TEMPLATES,
    build_prompt,
    get_device,
)


DEFAULT_CHECKPOINT = "final_sota_visual_entailment3.pth"
DEFAULT_ADAPTER = "task4_swiglu_adapter_best.pth"
DEFAULT_RESULTS = "task4_results.csv"
DEFAULT_SUMMARY = "task4_run_summary.json"
DEFAULT_DIAGNOSTICS = "task4_concept_diagnostics.csv"
DEFAULT_PROMPT_GUIDE = "prompt_engineering_guide_task4.md"
DEFAULT_FAILURE_REPORT = "what_till_we_do_wrong.md"

BADGE_CLASS_MAP = {
    "entailment": "ve-badge-entailment",
    "neutral": "ve-badge-neutral",
    "contradiction": "ve-badge-contradiction",
}


def _list_checkpoints() -> list[str]:
    checkpoints = list_backbone_checkpoints()
    return [ckpt for ckpt in checkpoints if Path(ckpt).name == DEFAULT_CHECKPOINT]


def _list_adapter_weights() -> list[str]:
    preferred = sorted([p.name for p in Path(".").glob("task4*adapter*.pth")])
    if DEFAULT_ADAPTER in _list_checkpoints() and DEFAULT_ADAPTER not in preferred:
        preferred = [DEFAULT_ADAPTER] + preferred
    if not preferred:
        preferred = sorted([p.name for p in Path(".").glob("*.pth")])
    return preferred


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
def load_task4_models(
    checkpoint_path: str,
    adapter_path: str,
) -> Tuple[FrozenViTTextEncoder, AdapterClassifier, torch.device, Dict[str, int]]:
    device = get_device()

    encoder = FrozenViTTextEncoder(EncoderConfig())
    loaded_counts = encoder.load_encoder_weights_from_checkpoint(Path(checkpoint_path), device=device)

    head = AdapterClassifier(
        in_dim=encoder.out_dim,
        bottleneck_dim=32,
        dropout=0.1,
        num_classes=3,
    )
    adapter_state = torch.load(adapter_path, map_location=device)
    if isinstance(adapter_state, dict) and "state_dict" in adapter_state:
        adapter_state = adapter_state["state_dict"]
    head.load_state_dict(adapter_state, strict=False)

    encoder.to(device).eval()
    head.to(device).eval()
    return encoder, head, device, loaded_counts


@torch.no_grad()
def infer_single_prompt(
    encoder: FrozenViTTextEncoder,
    head: AdapterClassifier,
    device: torch.device,
    image: Image.Image,
    hypothesis: str,
    template_text: str,
) -> Dict[str, object]:
    prompt = build_prompt(template_text, hypothesis)

    px = encoder.image_processor(images=[image], return_tensors="pt")["pixel_values"].to(device)
    toks = encoder.tokenizer(
        [prompt],
        padding=True,
        truncation=True,
        max_length=encoder.cfg.max_text_len,
        return_tensors="pt",
    )
    toks = {k: v.to(device) for k, v in toks.items()}

    vit_cls = encoder.vit(pixel_values=px).last_hidden_state[:, 0, :]
    bert_cls = encoder.bert(**toks).last_hidden_state[:, 0, :]

    feat = torch.cat([vit_cls, bert_cls], dim=1)
    feat = F.normalize(feat.float(), p=2, dim=1)

    logits = head(feat)
    probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()

    pred_idx = int(np.argmax(probs))
    return {
        "prompt": prompt,
        "pred_idx": pred_idx,
        "pred_label": ID2LABEL[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probs": probs,
    }


def _render_probs(title: str, probs: np.ndarray) -> None:
    st.markdown(f"**{title}**")
    for i, p in enumerate(probs.tolist()):
        st.progress(float(p), text=f"{ID2LABEL[i].capitalize()}: {100.0 * float(p):.1f}%")


def _safe_read_json(path: str) -> Optional[Dict[str, object]]:
    p = Path(path)
    if not p.exists():
        return None
    import json

    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_read_csv(path: str) -> Optional[pd.DataFrame]:
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    return df if not df.empty else None


def render_analysis_report(
    results_path: str,
    summary_path: str,
    diagnostics_path: str,
    prompt_guide_path: str,
    failure_report_path: str,
) -> None:
    summary = _safe_read_json(summary_path)
    results = _safe_read_csv(results_path)
    diagnostics = _safe_read_csv(diagnostics_path)

    if results is None:
        st.warning(f"Could not find results at `{results_path}`.")
        return

    template_perf = (
        results.groupby("template")[["zero_shot_acc", "one_shot_acc"]].mean().sort_values("zero_shot_acc", ascending=False)
    )
    concept_perf = (
        results.groupby("concept_type")[["zero_shot_acc", "one_shot_acc"]].mean().sort_values("zero_shot_acc", ascending=False)
    )
    dataset_template = (
        results.groupby(["dataset", "template"])["zero_shot_acc"].mean().unstack("template")
    )

    best_template = template_perf.index[0]
    best_zero = float(template_perf.iloc[0]["zero_shot_acc"])
    overall_zero = float(results["zero_shot_acc"].mean())
    overall_one = float(results["one_shot_acc"].mean())

    with st.container(border=True):
        st.markdown("<h2 class='ve-card-title'>Prompt Analysis Summary</h2>", unsafe_allow_html=True)
        st.markdown(
            "<p class='ve-card-subtitle'>Your Task 4 prompt-engineering outcomes across held-out and synthetic sets.</p>",
            unsafe_allow_html=True,
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Best template", best_template)
        m2.metric("Best zero-shot", f"{100 * best_zero:.2f}%")
        m3.metric("Mean zero-shot", f"{100 * overall_zero:.2f}%")
        m4.metric("Mean one-shot", f"{100 * overall_one:.2f}%")

        if summary is not None:
            st.caption(
                f"Checkpoint: `{summary.get('used_checkpoint')}` | "
                f"Adapter: `{Path(str(summary.get('adapter_weights_path', ''))).name}` | "
                f"Concept mode: `{summary.get('concept_mode_used')}`"
            )

    c1, c2 = st.columns(2, gap="large")

    with c1:
        with st.container(border=True):
            st.markdown("<h2 class='ve-card-title'>Template Performance</h2>", unsafe_allow_html=True)
            st.markdown("<p class='ve-card-subtitle'>Zero-shot and one-shot accuracy by prompt template.</p>", unsafe_allow_html=True)
            st.bar_chart(template_perf.mul(100.0), use_container_width=True)
            st.dataframe((template_perf.mul(100.0)).round(2), use_container_width=True)

    with c2:
        with st.container(border=True):
            st.markdown("<h2 class='ve-card-title'>Concept-Type Performance</h2>", unsafe_allow_html=True)
            st.markdown("<p class='ve-card-subtitle'>How each concept bucket responds to prompting.</p>", unsafe_allow_html=True)
            st.bar_chart(concept_perf.mul(100.0), use_container_width=True)
            st.dataframe((concept_perf.mul(100.0)).round(2), use_container_width=True)

    with st.container(border=True):
        st.markdown("<h2 class='ve-card-title'>Dataset x Template (Zero-shot Accuracy)</h2>", unsafe_allow_html=True)
        st.markdown("<p class='ve-card-subtitle'>Prompt transfer behavior across held-out and synthetic datasets.</p>", unsafe_allow_html=True)
        st.dataframe((dataset_template.mul(100.0)).round(2), use_container_width=True)

    if diagnostics is not None:
        err_by_template = diagnostics.groupby("template")["is_error"].mean().sort_values(ascending=False)
        confusions = (
            diagnostics.groupby(["true_label", "pred_zero_label"]).size().reset_index(name="count").sort_values("count", ascending=False).head(12)
        )

        d1, d2 = st.columns(2, gap="large")
        with d1:
            with st.container(border=True):
                st.markdown("<h2 class='ve-card-title'>Error Rate by Template</h2>", unsafe_allow_html=True)
                st.bar_chart((err_by_template.to_frame("error_rate") * 100.0), use_container_width=True)
        with d2:
            with st.container(border=True):
                st.markdown("<h2 class='ve-card-title'>Top Label Confusions</h2>", unsafe_allow_html=True)
                st.dataframe(confusions, use_container_width=True, height=280)

    guide = Path(prompt_guide_path)
    st.caption(
        "Prompt guide is auto-generated from `task4_results.csv`: global template ranking, "
        "concept-wise winners, reliability flags, and recommended usage patterns."
    )
    if guide.exists():
        with st.expander("Prompt Engineering Guide", expanded=False):
            st.markdown(guide.read_text(encoding="utf-8"))

    failure = Path(failure_report_path)
    if failure.exists():
        with st.expander("Failure Analysis Notes", expanded=False):
            st.markdown(failure.read_text(encoding="utf-8"))


def render_custom_playground(checkpoints: list[str], adapters: list[str]) -> None:
    notice: Optional[Tuple[str, str]] = None
    result: Optional[Dict[str, object]] = None
    compare_df: Optional[pd.DataFrame] = None

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        with st.container(border=True):
            st.markdown("<h2 class='ve-card-title'>1. Input</h2>", unsafe_allow_html=True)
            st.markdown(
                "<p class='ve-card-subtitle'>Upload your sample and test how template choice changes the entailment verdict.</p>",
                unsafe_allow_html=True,
            )

            if checkpoints:
                ckpt_idx = checkpoints.index(DEFAULT_CHECKPOINT) if DEFAULT_CHECKPOINT in checkpoints else 0
                checkpoint = st.selectbox("Backbone checkpoint", checkpoints, index=ckpt_idx)
            else:
                checkpoint = None
                st.error("No `.pth` checkpoints found in current directory.")

            if adapters:
                adapter_idx = adapters.index(DEFAULT_ADAPTER) if DEFAULT_ADAPTER in adapters else 0
                adapter_weights = st.selectbox("Task 4 adapter weights", adapters, index=adapter_idx)
            else:
                adapter_weights = None
                st.error("No adapter weight file found. Expected `task4_swiglu_adapter_best.pth`.")

            uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
            image = None
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert("RGB")
                st.markdown("<div class='ve-image-wrap'>", unsafe_allow_html=True)
                st.image(image, caption="Preview", use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            hypothesis = st.text_input("Hypothesis", placeholder="Example: Two people are riding bicycles on a street.")
            expected_label = st.selectbox(
                "Expected label (optional)",
                options=["(none)", "entailment", "neutral", "contradiction"],
                index=0,
            )

            use_custom_template = st.checkbox("Use custom template", value=False)
            if use_custom_template:
                template_name = "custom"
                template_text = st.text_area(
                    "Custom template",
                    value="Image context: {concept}. Claim: {text}. Decide: Entail, Contradict, or Neutral.",
                    height=110,
                )
            else:
                template_name = st.selectbox("Prompt template", options=list(PROMPT_TEMPLATES.keys()), index=0)
                template_text = PROMPT_TEMPLATES[template_name]
                st.caption(f"Template preview: `{template_text}`")

            compare_all = st.checkbox("Compare all preset templates", value=True)
            run_btn = st.button("Analyze Prompt Behavior", use_container_width=True, type="primary")

            if run_btn:
                if checkpoint is None or adapter_weights is None:
                    notice = ("error", "Please select valid checkpoint and adapter weights.")
                elif image is None or not hypothesis.strip():
                    notice = ("warning", "Please provide both an image and a hypothesis.")
                else:
                    try:
                        with st.spinner("Loading Task 4 pipeline..."):
                            encoder, head, device, loaded_counts = load_task4_models(checkpoint, adapter_weights)
                    except Exception as exc:
                        notice = ("error", f"Could not load Task 4 pipeline: {exc}")
                    else:
                        working_hypothesis = hypothesis.strip()

                        eval_templates: Dict[str, str] = {}
                        if compare_all:
                            eval_templates.update(PROMPT_TEMPLATES)
                            if use_custom_template:
                                eval_templates["custom"] = template_text
                        else:
                            eval_templates[template_name] = template_text

                        rows = []
                        selected_out = None
                        for t_name, t_text in eval_templates.items():
                            try:
                                out = infer_single_prompt(
                                    encoder=encoder,
                                    head=head,
                                    device=device,
                                    image=image,
                                    hypothesis=working_hypothesis,
                                    template_text=t_text,
                                )
                            except Exception as exc:
                                notice = ("error", f"Template `{t_name}` failed: {exc}")
                                rows = []
                                selected_out = None
                                break

                            if t_name == template_name or (template_name == "custom" and t_name == "custom"):
                                selected_out = out

                            row = {
                                "template": t_name,
                                "pred_label": str(out["pred_label"]),
                                "confidence_pct": 100.0 * float(out["confidence"]),
                                "entailment_pct": 100.0 * float(out["probs"][LABEL2ID["entailment"]]),
                                "neutral_pct": 100.0 * float(out["probs"][LABEL2ID["neutral"]]),
                                "contradiction_pct": 100.0 * float(out["probs"][LABEL2ID["contradiction"]]),
                                "prompt": str(out["prompt"]),
                            }
                            if expected_label != "(none)":
                                row["is_correct"] = int(row["pred_label"] == expected_label)
                            rows.append(row)

                        if selected_out is None and rows:
                            selected_out = infer_single_prompt(
                                encoder=encoder,
                                head=head,
                                device=device,
                                image=image,
                                hypothesis=working_hypothesis,
                                template_text=template_text,
                            )

                        if rows and selected_out is not None and notice is None:
                            compare_df = pd.DataFrame(rows)
                            result = {
                                "checkpoint": checkpoint,
                                "adapter": adapter_weights,
                                "loaded_counts": loaded_counts,
                                "hypothesis": hypothesis.strip(),
                                "expected_label": expected_label,
                                "selected_template": template_name,
                                "selected_output": selected_out,
                            }

    with col2:
        with st.container(border=True):
            st.markdown("<h2 class='ve-card-title'>2. Verdict</h2>", unsafe_allow_html=True)
            st.markdown(
                "<p class='ve-card-subtitle'>Prediction details for your selected prompt and optional template comparison.</p>",
                unsafe_allow_html=True,
            )

            if notice:
                if notice[0] == "warning":
                    st.warning(notice[1])
                else:
                    st.error(notice[1])
            elif result is not None:
                out = result["selected_output"]
                label = str(out["pred_label"])
                badge_class = BADGE_CLASS_MAP.get(label, "ve-badge-neutral")
                st.markdown(
                    f"<span class='ve-badge {badge_class}'>{label.capitalize()}</span>",
                    unsafe_allow_html=True,
                )

                m1, m2, m3 = st.columns(3)
                m1.metric("Selected template", str(result["selected_template"]))
                m2.metric("Confidence", f"{100 * float(out['confidence']):.2f}%")
                if result["expected_label"] != "(none)":
                    is_ok = "Yes" if label == result["expected_label"] else "No"
                    m3.metric("Matches expected?", is_ok)
                else:
                    m3.metric("Device", str(get_device()))

                st.caption(
                    f"Checkpoint: `{result['checkpoint']}` | Adapter: `{result['adapter']}` | "
                    f"Loaded keys: vit={result['loaded_counts'].get('vit_loaded', 0)}, "
                    f"bert={result['loaded_counts'].get('bert_loaded', 0)}"
                )

                st.markdown("**Rendered prompt**")
                st.code(str(out["prompt"]), language="text")
                _render_probs("Class probabilities", np.array(out["probs"]))

                if compare_df is not None and len(compare_df) > 1:
                    st.markdown("**Template comparison on this sample**")
                    st.dataframe(compare_df.drop(columns=["prompt"], errors="ignore"), use_container_width=True)
                    st.bar_chart(compare_df.set_index("template")[["confidence_pct"]], use_container_width=True)

                    if result["expected_label"] != "(none)" and "is_correct" in compare_df.columns:
                        one_sample_acc = 100.0 * float(compare_df["is_correct"].mean())
                        st.metric("Template accuracy on this sample", f"{one_sample_acc:.1f}%")
            else:
                st.info("Run analysis to see prompt behavior on your sample.")


def main() -> None:
    st.set_page_config(page_title="Visual Entailment", page_icon="👁️", layout="wide")
    apply_minimal_styles()
    st.markdown(
        "<h1 class='ve-title'>Task 4 <span class='ve-title-emphasis'>Prompt Transfer</span> Studio</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p class='ve-subtitle'>Review your prompt-engineering analysis and test template behavior on your own image-text inputs.</p>",
        unsafe_allow_html=True,
    )

    checkpoints = _list_checkpoints()
    adapters = _list_adapter_weights()

    tab_report, tab_playground = st.tabs(["Prompt Analysis Report", "Custom Prompt Playground"])

    with tab_report:
        c1, c2, c3 = st.columns([1.2, 1.2, 1], gap="large")
        with c1:
            results_path = st.text_input("Results CSV", value=DEFAULT_RESULTS)
            diagnostics_path = st.text_input("Diagnostics CSV", value=DEFAULT_DIAGNOSTICS)
        with c2:
            summary_path = st.text_input("Run summary JSON", value=DEFAULT_SUMMARY)
            prompt_guide_path = st.text_input("Prompt guide MD", value=DEFAULT_PROMPT_GUIDE)
        with c3:
            failure_report_path = st.text_input("Failure report MD", value=DEFAULT_FAILURE_REPORT)

        render_analysis_report(
            results_path=results_path,
            summary_path=summary_path,
            diagnostics_path=diagnostics_path,
            prompt_guide_path=prompt_guide_path,
            failure_report_path=failure_report_path,
        )

    with tab_playground:
        render_custom_playground(checkpoints=checkpoints, adapters=adapters)


if __name__ == "__main__":
    main()
