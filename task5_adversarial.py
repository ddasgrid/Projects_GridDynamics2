from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer


LABEL2ID = {"entailment": 0, "neutral": 1, "contradiction": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

TEXT_ATTACK_MODES = (
    "negation",
    "color_swap",
    "object_substitution",
    "synonym_substitution",
    "paraphrase",
)

COLOR_SWAPS: Dict[str, List[str]] = {
    "red": ["blue", "green", "yellow"],
    "blue": ["red", "green", "black"],
    "green": ["red", "blue", "brown"],
    "yellow": ["blue", "black", "white"],
    "black": ["white", "yellow", "red"],
    "white": ["black", "red", "blue"],
    "brown": ["black", "white", "gray"],
    "gray": ["black", "white", "red"],
}

OBJECT_SWAPS: Dict[str, List[str]] = {
    "man": ["woman", "boy", "girl"],
    "woman": ["man", "boy", "girl"],
    "boy": ["girl", "man", "woman"],
    "girl": ["boy", "man", "woman"],
    "dog": ["cat", "horse", "person"],
    "cat": ["dog", "rabbit", "person"],
    "horse": ["dog", "bike", "car"],
    "bike": ["car", "horse", "bus"],
    "car": ["bike", "bus", "motorcycle"],
    "bus": ["car", "train", "truck"],
    "ball": ["book", "bag", "hat"],
    "table": ["chair", "bench", "floor"],
    "chair": ["table", "bench", "sofa"],
}

PARAPHRASE_RULES: Tuple[Tuple[str, str], ...] = (
    (r"\bA person\b", "Someone"),
    (r"\bA man\b", "A male person"),
    (r"\bA woman\b", "A female person"),
    (r"\bis standing\b", "stands"),
    (r"\bis sitting\b", "sits"),
    (r"\bis running\b", "runs"),
    (r"\bis walking\b", "walks"),
    (r"\bis riding\b", "rides"),
    (r"\bis holding\b", "holds"),
)

_WORDNET_SETUP_CACHE: Optional[Tuple[object, object]] = None
_WORDNET_SETUP_ATTEMPTED = False


@dataclass
class Task5Config:
    test_csv: str = "cleaned_snli_ve_test.csv"
    checkpoint_path: str = "final_sota_visual_entailment3.pth"
    vision_model_name: str = "google/vit-base-patch16-224"
    text_model_name: str = "bert-base-uncased"
    output_dir: str = "task5_artifacts"
    num_adversarial_pairs: int = 100
    batch_size: int = 8
    max_text_length: int = 64
    seed: int = 42
    clean_eval_limit: Optional[int] = None
    epsilons: List[float] = field(default_factory=lambda: [1 / 255, 4 / 255, 8 / 255])
    pgd_steps: int = 10
    pgd_step_scale: float = 0.25
    image_attack_methods: Tuple[str, ...] = ("fgsm", "pgd")
    strongest_attack_method: str = "pgd"
    save_adversarial_images: bool = False
    save_top_k_tokens: int = 25
    save_top_k_regions: int = 10


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
        fusion_type: str = "attention",
        freeze_mode: str = "full",
        num_layers_to_freeze: int = 12,
    ):
        super().__init__()
        self.fusion_type = fusion_type

        self.vit = AutoModel.from_pretrained(vision_model_name)
        self.bert = AutoModel.from_pretrained(text_model_name)
        self._apply_freezing(freeze_mode, num_layers_to_freeze)

        vit_hidden = self.vit.config.hidden_size
        bert_hidden = self.bert.config.hidden_size
        if vit_hidden != bert_hidden:
            raise ValueError("This model expects matching ViT and BERT hidden sizes for attention fusion.")

        self.cross_attention = nn.MultiheadAttention(embed_dim=vit_hidden, num_heads=8, batch_first=True)

        fusion_layers: List[nn.Module] = []
        in_features = vit_hidden
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
            for param in self.vit.parameters():
                param.requires_grad = False
            for param in self.bert.parameters():
                param.requires_grad = False
            return

        if mode == "partial":
            def freeze_n_layers(model: nn.Module, n: int) -> None:
                if hasattr(model, "embeddings"):
                    for param in model.embeddings.parameters():
                        param.requires_grad = False
                if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
                    layers = model.encoder.layer
                    n_layers = min(n, len(layers))
                    for i in range(n_layers):
                        for param in layers[i].parameters():
                            param.requires_grad = False

            freeze_n_layers(self.vit, num_layers)
            freeze_n_layers(self.bert, num_layers)

    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        vit_outputs = self.vit(pixel_values=pixel_values)
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        vit_cls = vit_outputs.last_hidden_state[:, 0, :]
        bert_cls = bert_outputs.last_hidden_state[:, 0, :]

        query = bert_cls.unsqueeze(1)
        key_value = vit_outputs.last_hidden_state
        attn_output, _ = self.cross_attention(query=query, key=key_value, value=key_value)
        base_fused = attn_output.squeeze(1)

        deep_fused_features = self.fusion_head(base_fused)
        logits = self.classifier_head(deep_fused_features)
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
    preds: List[torch.Tensor] = []
    confs: List[torch.Tensor] = []

    for start in range(0, len(pixel_values), batch_size):
        end = start + batch_size
        x = pixel_values[start:end].to(device)
        ids = input_ids[start:end].to(device)
        mask = attention_mask[start:end].to(device)

        logits = model(x, ids, mask)
        probs = torch.softmax(logits, dim=-1)
        preds.append(torch.argmax(probs, dim=-1).cpu())
        confs.append(torch.max(probs, dim=-1).values.cpu())

    return torch.cat(preds), torch.cat(confs)


def evaluate_dataframe_accuracy(
    model: nn.Module,
    df: pd.DataFrame,
    processor,
    tokenizer,
    device: torch.device,
    batch_size: int,
    max_text_length: int,
    limit: Optional[int] = None,
) -> Dict[str, float]:
    if limit is not None:
        df = df.head(limit).copy()

    labels = []
    preds = []
    valid_count = 0

    for start in range(0, len(df), batch_size):
        batch = df.iloc[start : start + batch_size].copy()
        batch = batch[batch["image_path"].map(lambda p: Path(p).exists())]
        if batch.empty:
            continue

        px = _prepare_pixels(batch["image_path"].tolist(), processor)
        toks = _prepare_text(batch["hypothesis"].astype(str).tolist(), tokenizer, max_length=max_text_length)
        batch_preds, _ = predict_from_tensors(
            model=model,
            pixel_values=px,
            input_ids=toks["input_ids"],
            attention_mask=toks["attention_mask"],
            device=device,
            batch_size=batch_size,
        )

        preds.extend(batch_preds.tolist())
        labels.extend(batch["label"].map(LABEL2ID).tolist())
        valid_count += len(batch)

    y_true = np.array(labels)
    y_pred = np.array(preds)
    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0

    return {
        "accuracy": acc,
        "evaluated_rows": int(valid_count),
    }


def _replace_first_word(text: str, source: str, replacement: str) -> Tuple[str, bool]:
    pattern = re.compile(rf"\b{re.escape(source)}\b", flags=re.IGNORECASE)
    if not pattern.search(text):
        return text, False
    return pattern.sub(replacement, text, count=1), True


def _apply_negation(text: str) -> Tuple[str, bool, str]:
    auxiliaries = ["is", "are", "was", "were", "has", "have", "can", "will", "does", "do", "did"]
    for aux in auxiliaries:
        pattern = re.compile(rf"\b{aux}\b(?!\s+not\b)", flags=re.IGNORECASE)
        if pattern.search(text):
            return pattern.sub(f"{aux} not", text, count=1), True, aux
    return f"It is not true that {text}", True, "prefix_negation"


def _apply_dict_swap(text: str, mapping: Dict[str, List[str]], rng: random.Random) -> Tuple[str, bool, str]:
    keys = list(mapping.keys())
    rng.shuffle(keys)
    for source in keys:
        candidates = mapping[source]
        replacement = rng.choice(candidates)
        out, changed = _replace_first_word(text, source, replacement)
        if changed:
            return out, True, source
    return text, False, ""


def _safe_wordnet_setup() -> Optional[Tuple[object, object]]:
    global _WORDNET_SETUP_CACHE, _WORDNET_SETUP_ATTEMPTED
    if _WORDNET_SETUP_ATTEMPTED:
        return _WORDNET_SETUP_CACHE

    _WORDNET_SETUP_ATTEMPTED = True
    try:
        import nltk
        from nltk.corpus import wordnet
        from nltk.corpus.reader.wordnet import NOUN, VERB, ADJ, ADV
    except Exception:
        return None

    try:
        resource_checks = [
            ("tokenizers/punkt", "punkt"),
            ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
            ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
            ("corpora/wordnet", "wordnet"),
            ("corpora/omw-1.4", "omw-1.4"),
        ]
        for resource_path, _pkg in resource_checks:
            try:
                nltk.data.find(resource_path)
            except LookupError:
                return None
    except Exception:
        return None

    pos_map = {"N": NOUN, "V": VERB, "J": ADJ, "R": ADV}
    _WORDNET_SETUP_CACHE = (wordnet, pos_map)
    return _WORDNET_SETUP_CACHE


def _apply_synonym_substitution(text: str, rng: random.Random) -> Tuple[str, bool, str]:
    setup = _safe_wordnet_setup()
    if setup is None:
        return text, False, ""
    wordnet, pos_map = setup

    try:
        import nltk
    except Exception:
        return text, False, ""

    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)

    for idx, (token, pos) in enumerate(tagged):
        wn_pos = pos_map.get(pos[:1].upper())
        if wn_pos is None:
            continue
        synsets = wordnet.synsets(token, pos=wn_pos)
        synonyms = set()
        for syn in synsets:
            for lemma in syn.lemmas():
                cand = lemma.name().replace("_", " ")
                if cand.lower() != token.lower() and cand.isalpha():
                    synonyms.add(cand)
        if synonyms:
            replacement = rng.choice(sorted(synonyms))
            tokens[idx] = replacement
            return " ".join(tokens), True, token
    return text, False, ""


def _apply_paraphrase(text: str, rng: random.Random) -> Tuple[str, bool, str]:
    rules = list(PARAPHRASE_RULES)
    rng.shuffle(rules)
    for pattern, repl in rules:
        out, n = re.subn(pattern, repl, text, count=1, flags=re.IGNORECASE)
        if n > 0:
            return out, True, pattern
    return f"In this image, {text[0].lower() + text[1:] if text else text}", True, "prefix"


def generate_text_adversary(text: str, mode: str, rng: random.Random) -> Tuple[str, str, str]:
    if mode == "negation":
        out, changed, anchor = _apply_negation(text)
    elif mode == "color_swap":
        out, changed, anchor = _apply_dict_swap(text, COLOR_SWAPS, rng)
    elif mode == "object_substitution":
        out, changed, anchor = _apply_dict_swap(text, OBJECT_SWAPS, rng)
    elif mode == "synonym_substitution":
        out, changed, anchor = _apply_synonym_substitution(text, rng)
    elif mode == "paraphrase":
        out, changed, anchor = _apply_paraphrase(text, rng)
    else:
        out, changed, anchor = text, False, ""

    if changed and out != text:
        return out, mode, anchor

    for fallback in TEXT_ATTACK_MODES:
        if fallback == mode:
            continue
        out, used_mode, anchor = generate_text_adversary(text, fallback, rng)
        if out != text:
            return out, used_mode, anchor

    return f"It is not true that {text}", "negation", "forced_prefix"


def create_text_adversarial_pairs(base_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for i, row in base_df.reset_index(drop=True).iterrows():
        preferred_mode = TEXT_ATTACK_MODES[i % len(TEXT_ATTACK_MODES)]
        adv_text, mode_used, anchor = generate_text_adversary(str(row["hypothesis"]), preferred_mode, rng)
        expected_label = "entailment" if mode_used in {"synonym_substitution", "paraphrase"} else "contradiction"
        rows.append(
            {
                "source_index": int(row["source_index"]),
                "image_path": row["image_path"],
                "clean_hypothesis": row["hypothesis"],
                "adversarial_hypothesis": adv_text,
                "attack_mode": mode_used,
                "anchor_token_or_rule": anchor,
                "expected_label": expected_label,
            }
        )
    return pd.DataFrame(rows)


def _norm_params(processor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(processor.image_mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor(processor.image_std, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    return mean, std


def _clamp_normalized(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    x_min = (0.0 - mean) / std
    x_max = (1.0 - mean) / std
    return torch.max(torch.min(x, x_max), x_min)


def fgsm_attack(
    model: nn.Module,
    x_orig: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    x = x_orig.detach().clone().requires_grad_(True)
    logits = model(x, input_ids, attention_mask)
    loss = F.cross_entropy(logits, labels)
    loss.backward()

    grad = x.grad.detach()
    eps = torch.tensor(epsilon, dtype=x.dtype, device=x.device).view(1, 1, 1, 1) / std
    x_adv = x + eps * grad.sign()
    delta = torch.clamp(x_adv - x_orig, min=-eps, max=eps)
    x_adv = _clamp_normalized(x_orig + delta, mean, std).detach()
    return x_adv, grad.abs().detach()


def pgd_attack(
    model: nn.Module,
    x_orig: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    alpha: float,
    steps: int,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    model.eval()
    eps = torch.tensor(epsilon, dtype=x_orig.dtype, device=x_orig.device).view(1, 1, 1, 1) / std
    alpha_t = torch.tensor(alpha, dtype=x_orig.dtype, device=x_orig.device).view(1, 1, 1, 1) / std

    x_adv = x_orig.detach() + torch.empty_like(x_orig).uniform_(-1.0, 1.0) * eps
    x_adv = _clamp_normalized(x_adv, mean, std)

    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        logits = model(x_adv, input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha_t * grad.sign()
        delta = torch.clamp(x_adv - x_orig, min=-eps, max=eps)
        x_adv = _clamp_normalized(x_orig + delta, mean, std)

    return x_adv.detach()


def _save_adv_batch_images(
    batch_adv_pixels: torch.Tensor,
    out_dir: Path,
    sample_ids: List[int],
    mean: torch.Tensor,
    std: torch.Tensor,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    mean_cpu = mean.detach().cpu()
    std_cpu = std.detach().cpu()

    for i in range(len(batch_adv_pixels)):
        x = batch_adv_pixels[i : i + 1].detach().cpu()
        img = x * std_cpu + mean_cpu
        img = img.clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0).numpy()
        pil = Image.fromarray((img * 255.0).astype(np.uint8))
        pil.save(out_dir / f"sample_{sample_ids[i]:03d}.png")


def select_base_pairs(
    model: nn.Module,
    test_df: pd.DataFrame,
    processor,
    tokenizer,
    device: torch.device,
    batch_size: int,
    max_text_length: int,
    num_samples: int,
    seed: int,
) -> pd.DataFrame:
    entail_df = test_df[test_df["label"] == "entailment"].copy()
    entail_df = entail_df[entail_df["image_path"].map(lambda p: Path(p).exists())]
    entail_df = entail_df.sample(frac=1.0, random_state=seed).reset_index(drop=False)
    entail_df = entail_df.rename(columns={"index": "source_index"})

    selected_rows = []
    for start in range(0, len(entail_df), batch_size):
        batch = entail_df.iloc[start : start + batch_size]
        if batch.empty:
            continue

        px = _prepare_pixels(batch["image_path"].tolist(), processor)
        toks = _prepare_text(batch["hypothesis"].astype(str).tolist(), tokenizer, max_length=max_text_length)
        preds, _ = predict_from_tensors(
            model=model,
            pixel_values=px,
            input_ids=toks["input_ids"],
            attention_mask=toks["attention_mask"],
            device=device,
            batch_size=batch_size,
        )

        keep = preds.numpy() == LABEL2ID["entailment"]
        if keep.any():
            selected_rows.append(batch.loc[keep, ["source_index", "image_path", "hypothesis", "label"]])
        if selected_rows and sum(len(x) for x in selected_rows) >= num_samples:
            break

    if not selected_rows:
        return pd.DataFrame(columns=["source_index", "image_path", "hypothesis", "label"])

    out = pd.concat(selected_rows, ignore_index=True).head(num_samples).copy()
    return out


def _token_gradients(
    model: nn.Module,
    pixel_values: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    captured: Dict[str, torch.Tensor] = {}

    def hook(_module, _inp, out):
        if not out.requires_grad:
            out = out.detach().requires_grad_(True)
        out.retain_grad()
        captured["emb"] = out
        return out

    handle = model.bert.embeddings.register_forward_hook(hook)
    word_emb = model.bert.embeddings.word_embeddings
    prev_requires_grad = bool(word_emb.weight.requires_grad)
    try:
        if not prev_requires_grad:
            word_emb.weight.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        logits = model(
            pixel_values.to(device),
            input_ids.to(device),
            attention_mask.to(device),
        )
        loss = F.cross_entropy(logits, labels.to(device))
        loss.backward()
        grads = captured["emb"].grad.detach().cpu()
    finally:
        handle.remove()
        if not prev_requires_grad:
            word_emb.weight.requires_grad_(False)
        model.zero_grad(set_to_none=True)
    return grads


def compute_token_vulnerability(
    model: nn.Module,
    pixel_values: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    tokenizer,
    device: torch.device,
    batch_size: int,
    top_k: int = 25,
) -> pd.DataFrame:
    special_ids = set(tokenizer.all_special_ids)
    token_stats: Dict[str, List[float]] = {}

    for start in range(0, len(pixel_values), batch_size):
        end = start + batch_size
        grads = _token_gradients(
            model=model,
            pixel_values=pixel_values[start:end],
            input_ids=input_ids[start:end],
            attention_mask=attention_mask[start:end],
            labels=labels[start:end],
            device=device,
        )
        norms = grads.norm(dim=-1)
        ids = input_ids[start:end]
        masks = attention_mask[start:end]

        for b in range(ids.size(0)):
            valid_len = int(masks[b].sum().item())
            for t in range(valid_len):
                token_id = int(ids[b, t].item())
                if token_id in special_ids:
                    continue
                token = tokenizer.convert_ids_to_tokens(token_id).lower().replace("##", "")
                if not token or not re.match(r"^[a-z]+$", token):
                    continue
                score = float(norms[b, t].item())
                token_stats.setdefault(token, []).append(score)

    rows = []
    for token, scores in token_stats.items():
        if len(scores) < 2:
            continue
        rows.append(
            {
                "token": token,
                "mean_grad_norm": float(np.mean(scores)),
                "median_grad_norm": float(np.median(scores)),
                "count": int(len(scores)),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["mean_grad_norm", "count"], ascending=[False, False]).head(top_k).reset_index(drop=True)
    return out


def summarize_saliency_regions(
    saliency_sum: torch.Tensor,
    count: int,
    top_k: int,
) -> pd.DataFrame:
    if count <= 0:
        return pd.DataFrame(columns=["patch_row", "patch_col", "mean_saliency"])

    mean_map = (saliency_sum / count).cpu()
    flat = mean_map.flatten()
    top_vals, top_idx = torch.topk(flat, k=min(top_k, flat.numel()))

    rows = []
    width = mean_map.size(1)
    for val, idx in zip(top_vals.tolist(), top_idx.tolist()):
        r = idx // width
        c = idx % width
        rows.append({"patch_row": int(r), "patch_col": int(c), "mean_saliency": float(val)})
    return pd.DataFrame(rows)


def run_task5(config: Optional[Task5Config] = None) -> Dict[str, object]:
    cfg = config or Task5Config()
    set_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(cfg.test_csv)
    df = df[df["label"].isin(LABEL2ID)].copy()

    device = pick_device()
    print(f"Using device: {device}")

    image_processor = _load_processor(cfg.vision_model_name)
    tokenizer = _load_tokenizer(cfg.text_model_name)

    model = VisualEntailmentModel(
        vision_model_name=cfg.vision_model_name,
        text_model_name=cfg.text_model_name,
        hidden_dim=512,
        dropout_rate=0.3,
        depth=1,
        fusion_type="attention",
        freeze_mode="full",
        num_layers_to_freeze=12,
    )
    ckpt_path = Path(cfg.checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    clean_metrics = evaluate_dataframe_accuracy(
        model=model,
        df=df,
        processor=image_processor,
        tokenizer=tokenizer,
        device=device,
        batch_size=cfg.batch_size,
        max_text_length=cfg.max_text_length,
        limit=cfg.clean_eval_limit,
    )
    print(
        "Clean test accuracy: "
        f"{clean_metrics['accuracy']:.4f} "
        f"on {clean_metrics['evaluated_rows']} rows"
    )

    base_df = select_base_pairs(
        model=model,
        test_df=df,
        processor=image_processor,
        tokenizer=tokenizer,
        device=device,
        batch_size=cfg.batch_size,
        max_text_length=cfg.max_text_length,
        num_samples=cfg.num_adversarial_pairs,
        seed=cfg.seed,
    )
    if len(base_df) < cfg.num_adversarial_pairs:
        print(f"Warning: only found {len(base_df)} clean entailment-correct pairs.")
    if base_df.empty:
        raise RuntimeError("Could not create base adversarial set.")

    base_df.to_csv(output_dir / "task5_base_pairs.csv", index=False)

    base_pixels = _prepare_pixels(base_df["image_path"].tolist(), image_processor)
    clean_toks = _prepare_text(base_df["hypothesis"].astype(str).tolist(), tokenizer, max_length=cfg.max_text_length)
    clean_labels = torch.tensor([LABEL2ID["entailment"]] * len(base_df), dtype=torch.long)

    clean_base_preds, clean_base_conf = predict_from_tensors(
        model=model,
        pixel_values=base_pixels,
        input_ids=clean_toks["input_ids"],
        attention_mask=clean_toks["attention_mask"],
        device=device,
        batch_size=cfg.batch_size,
    )
    clean_base_acc = float((clean_base_preds == clean_labels).float().mean().item())

    text_adv_df = create_text_adversarial_pairs(base_df, seed=cfg.seed)
    text_adv_df.to_csv(output_dir / "task5_text_adversarial_pairs.csv", index=False)

    text_toks = _prepare_text(
        text_adv_df["adversarial_hypothesis"].astype(str).tolist(),
        tokenizer,
        max_length=cfg.max_text_length,
    )
    text_expected = torch.tensor(text_adv_df["expected_label"].map(LABEL2ID).values, dtype=torch.long)

    text_preds, _ = predict_from_tensors(
        model=model,
        pixel_values=base_pixels,
        input_ids=text_toks["input_ids"],
        attention_mask=text_toks["attention_mask"],
        device=device,
        batch_size=cfg.batch_size,
    )
    text_adv_acc = float((text_preds == text_expected).float().mean().item())
    text_flip_rate = float((text_preds != clean_base_preds).float().mean().item())

    mean, std = _norm_params(image_processor, device=device)
    image_results: List[Dict[str, object]] = []
    strongest_eps = max(cfg.epsilons)
    strongest_adv_pixels: Optional[torch.Tensor] = None
    saliency_sum = torch.zeros(14, 14, dtype=torch.float32)
    saliency_batches = 0

    for method in cfg.image_attack_methods:
        for eps in cfg.epsilons:
            adv_chunks: List[torch.Tensor] = []
            pred_chunks: List[torch.Tensor] = []

            for start in range(0, len(base_pixels), cfg.batch_size):
                end = start + cfg.batch_size
                x = base_pixels[start:end].to(device)
                ids = clean_toks["input_ids"][start:end].to(device)
                mask = clean_toks["attention_mask"][start:end].to(device)
                y = clean_labels[start:end].to(device)

                if method == "fgsm":
                    adv_x, grad_abs = fgsm_attack(
                        model=model,
                        x_orig=x,
                        input_ids=ids,
                        attention_mask=mask,
                        labels=y,
                        epsilon=eps,
                        mean=mean,
                        std=std,
                    )
                    patch_saliency = F.adaptive_avg_pool2d(
                        grad_abs.mean(dim=1, keepdim=True),
                        (14, 14),
                    ).mean(dim=0).squeeze(0).cpu()
                    saliency_sum += patch_saliency
                    saliency_batches += 1
                elif method == "pgd":
                    alpha = eps * cfg.pgd_step_scale
                    adv_x = pgd_attack(
                        model=model,
                        x_orig=x,
                        input_ids=ids,
                        attention_mask=mask,
                        labels=y,
                        epsilon=eps,
                        alpha=alpha,
                        steps=cfg.pgd_steps,
                        mean=mean,
                        std=std,
                    )
                else:
                    raise ValueError(f"Unsupported image attack method: {method}")

                with torch.no_grad():
                    logits = model(adv_x, ids, mask)
                    preds = torch.argmax(logits, dim=-1)

                adv_chunks.append(adv_x.cpu())
                pred_chunks.append(preds.cpu())

                if cfg.save_adversarial_images:
                    method_dir = output_dir / "adversarial_images" / method / f"eps_{eps:.6f}"
                    sample_ids = list(range(start, min(end, len(base_pixels))))
                    _save_adv_batch_images(adv_x, method_dir, sample_ids, mean=mean, std=std)

            adv_pixels_all = torch.cat(adv_chunks, dim=0)
            preds_all = torch.cat(pred_chunks, dim=0)
            acc = float((preds_all == clean_labels).float().mean().item())
            flip = float((preds_all != clean_labels).float().mean().item())

            image_results.append(
                {
                    "attack_method": method,
                    "epsilon": float(eps),
                    "accuracy": acc,
                    "attack_success_rate": flip,
                }
            )
            print(
                f"{method.upper()} eps={eps:.6f} "
                f"| acc={acc:.4f} | asr={flip:.4f}"
            )

            if method == cfg.strongest_attack_method and abs(eps - strongest_eps) < 1e-12:
                strongest_adv_pixels = adv_pixels_all.clone()

    image_results_df = pd.DataFrame(image_results).sort_values(["attack_method", "epsilon"]).reset_index(drop=True)
    image_results_df.to_csv(output_dir / "task5_image_attack_results.csv", index=False)

    combined_adv_acc = None
    combined_flip_rate = None
    if strongest_adv_pixels is not None:
        combined_preds, _ = predict_from_tensors(
            model=model,
            pixel_values=strongest_adv_pixels,
            input_ids=text_toks["input_ids"],
            attention_mask=text_toks["attention_mask"],
            device=device,
            batch_size=cfg.batch_size,
        )
        combined_adv_acc = float((combined_preds == text_expected).float().mean().item())
        combined_flip_rate = float((combined_preds != clean_base_preds).float().mean().item())

    token_vuln_df = compute_token_vulnerability(
        model=model,
        pixel_values=base_pixels,
        input_ids=clean_toks["input_ids"],
        attention_mask=clean_toks["attention_mask"],
        labels=clean_labels,
        tokenizer=tokenizer,
        device=device,
        batch_size=cfg.batch_size,
        top_k=cfg.save_top_k_tokens,
    )
    token_vuln_df.to_csv(output_dir / "task5_token_vulnerability.csv", index=False)

    region_vuln_df = summarize_saliency_regions(
        saliency_sum=saliency_sum,
        count=saliency_batches,
        top_k=cfg.save_top_k_regions,
    )
    region_vuln_df.to_csv(output_dir / "task5_image_region_vulnerability.csv", index=False)

    robustness_summary = {
        "config": asdict(cfg),
        "clean_test_accuracy": clean_metrics["accuracy"],
        "clean_test_evaluated_rows": clean_metrics["evaluated_rows"],
        "clean_base_accuracy": clean_base_acc,
        "clean_base_mean_confidence": float(clean_base_conf.mean().item()),
        "text_adversarial_accuracy": text_adv_acc,
        "text_flip_rate": text_flip_rate,
        "combined_adversarial_accuracy": combined_adv_acc,
        "combined_flip_rate": combined_flip_rate,
        "image_attack_results": image_results,
        "artifacts": {
            "base_pairs_csv": str(output_dir / "task5_base_pairs.csv"),
            "text_pairs_csv": str(output_dir / "task5_text_adversarial_pairs.csv"),
            "image_results_csv": str(output_dir / "task5_image_attack_results.csv"),
            "token_vulnerability_csv": str(output_dir / "task5_token_vulnerability.csv"),
            "region_vulnerability_csv": str(output_dir / "task5_image_region_vulnerability.csv"),
        },
    }

    with (output_dir / "task5_robustness_summary.json").open("w", encoding="utf-8") as f:
        json.dump(robustness_summary, f, indent=2)

    print(f"Saved artifacts to: {output_dir}")
    return robustness_summary


if __name__ == "__main__":
    run_task5()
