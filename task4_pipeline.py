from __future__ import annotations
import torch.nn.functional as F
from tqdm import tqdm

import copy
import gc
import json
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer


LABEL2ID = {"entailment": 0, "neutral": 1, "contradiction": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

CHECKPOINT_CANDIDATES = [
    "final_sota_visual_entailment3.pth",
    "final_sota_visual_entailment2.pth",
    "final_sota_visual_entailment.pth",
    "sota_visual_entailment2.pth",
    "sota_visual_entailment.pth",
    "best_model_acc_73.7.pth",
]

PROMPT_TEMPLATES = {
    # 1. The Default Baseline
    "base": "Image shows {concept}. Statement: {text}. Entail, Contradict, or Neutral?",
    
    # 2. Low Specificity (The "Car" level)
    # Instructs the model to look for broad, general category matches.
    "low_specificity": "Look at the broad categories and general scene. Loosely speaking, does the image align with the general idea of: {text}. Output: Entail, Contradict, or Neutral.",
    
    # 3. High Specificity (The "Red Mercedes Maybach" level)
    # Forces the model to actively hunt for exact sub-categories, colors, and fine-grained attributes.
    "high_specificity": "Verify the exact specific sub-categories, precise attributes, and exact object types. Does the visual evidence strictly entail every specific detail in the statement: {text}? Output: Entail, Contradict, or Neutral.",
    
    # 4. Concept Framing (Structural anchoring)
    "concept_framing": "Task: Logical Verification. Visual Anchor: [{concept}]. Based on this anchor, evaluate the claim: {text}. Output: Entail, Contradict, Neutral.",
    
    # 5. Explicit Negation
    "negation_explicit": "Image shows {concept}. Statement to verify: {text}. Handle negation explicitly and output one of: Entail, Contradict, Neutral.",
}

CONCEPT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "over",
    "that",
    "the",
    "there",
    "this",
    "to",
    "under",
    "was",
    "were",
    "while",
    "with",
}

CONCEPT_LABELS = ("negation", "count", "spatial", "action", "object_or_attribute")
CONCEPT_PRIORITY = {
    "negation": 0,
    "count": 1,
    "spatial": 2,
    "action": 3,
    "object_or_attribute": 4,
}

NEGATION_TOKENS = {
    "not",
    "no",
    "never",
    "none",
    "neither",
    "nor",
    "without",
    "cannot",
    "can't",
    "dont",
    "don't",
    "doesnt",
    "doesn't",
    "didnt",
    "didn't",
    "isnt",
    "isn't",
    "arent",
    "aren't",
    "wasnt",
    "wasn't",
    "werent",
    "weren't",
    "wont",
    "won't",
}

SPATIAL_SINGLE = {
    "left",
    "right",
    "behind",
    "front",
    "near",
    "beside",
    "between",
    "under",
    "over",
    "above",
    "below",
    "inside",
    "outside",
    "around",
    "across",
}
SPATIAL_PHRASES = {
    "next to",
    "in front of",
    "on top of",
    "close to",
    "far from",
}

COUNT_WORDS = {
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "many",
    "several",
    "few",
    "multiple",
    "single",
    "double",
    "pair",
}

ACTION_LEMMAS = {
    "run",
    "jump",
    "walk",
    "stand",
    "sit",
    "ride",
    "play",
    "dance",
    "swim",
    "climb",
    "throw",
    "catch",
    "eat",
    "drink",
    "talk",
    "smile",
    "look",
    "hold",
}

ACTION_FORMS = {
    "running",
    "jumping",
    "walking",
    "standing",
    "sitting",
    "riding",
    "playing",
    "dancing",
    "swimming",
    "climbing",
    "throwing",
    "catching",
    "eating",
    "drinking",
    "talking",
    "smiling",
    "looking",
    "holding",
    "ran",
    "jumped",
    "walked",
    "stood",
    "sat",
}

ACTION_BLACKLIST_PHRASES = (
    "running out of",
    "standing desk",
    "standing table",
)

AUXILIARY_VERBS = {
    "is",
    "are",
    "am",
    "was",
    "were",
    "be",
    "been",
    "being",
    "has",
    "have",
    "had",
    "do",
    "does",
    "did",
    "can",
    "could",
    "will",
    "would",
    "should",
    "may",
    "might",
    "must",
}

SPATIAL_PREPS = {
    "in",
    "on",
    "under",
    "over",
    "behind",
    "beside",
    "near",
    "between",
    "around",
    "across",
    "left",
    "right",
}

_SPACY_NLP = None
_SPACY_MODEL: Optional[str] = None
_SPACY_INIT = False


@dataclass
class EncoderConfig:
    vision_model_name: str = "google/vit-base-patch16-224"
    text_model_name: str = "bert-base-uncased"
    max_text_len: int = 48


@dataclass
class ConceptLexicon:
    negation_tokens: Set[str] = field(default_factory=set)
    count_words: Set[str] = field(default_factory=set)
    action_lemmas: Set[str] = field(default_factory=set)
    action_forms: Set[str] = field(default_factory=set)
    top_nouns: List[str] = field(default_factory=list)
    top_adjectives: List[str] = field(default_factory=list)
    top_verbs: List[str] = field(default_factory=list)
    source: str = "seed"
    pos_tagger: str = "none"


def build_seed_concept_lexicon() -> ConceptLexicon:
    return ConceptLexicon(
        negation_tokens=set(NEGATION_TOKENS),
        count_words=set(COUNT_WORDS),
        action_lemmas=set(ACTION_LEMMAS),
        action_forms=set(ACTION_FORMS),
        source="seed",
        pos_tagger="none",
    )


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def workspace_history_summary(base_dir: Path) -> Dict[str, List[str]]:
    notebook_files = sorted(p.name for p in base_dir.glob("*.ipynb"))
    checkpoint_files = sorted(p.name for p in base_dir.glob("*.pth"))
    data_files = sorted(p.name for p in base_dir.glob("*") if p.is_file() and p.suffix in {".csv", ".jsonl"})
    return {
        "notebooks": notebook_files,
        "checkpoints": checkpoint_files,
        "datasets": data_files,
    }


def load_clean_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    keep = ["image_path", "premise", "hypothesis", "label"]
    df = df[keep].dropna().copy()
    df = df[df["label"].isin(LABEL2ID.keys())].reset_index(drop=True)
    return df


def sample_df(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    n = int(min(max(n, 1), len(df)))
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def negate_sentence(text: str) -> str:
    t = str(text).strip()
    if not t:
        return "This is not true."
    if t.lower().startswith("not "):
        return t
    if t.endswith("."):
        t = t[:-1]
    return f"It is not true that {t}."


def build_synthetic_sets(base_df: pd.DataFrame, n_each: int, seed: int) -> Dict[str, pd.DataFrame]:
    src = sample_df(base_df, min(n_each, len(base_df)), seed=seed)

    synth_identity = src.copy()
    synth_identity["hypothesis"] = synth_identity["premise"]
    synth_identity["label"] = "entailment"

    synth_negation = src.copy()
    synth_negation["hypothesis"] = synth_negation["premise"].map(negate_sentence)
    synth_negation["label"] = "contradiction"

    shuffled = src["premise"].sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)
    synth_neutral = src.copy()
    synth_neutral["hypothesis"] = shuffled
    synth_neutral["label"] = "neutral"

    same_mask = synth_neutral["hypothesis"].values == synth_neutral["premise"].values
    if same_mask.any():
        rolled = np.roll(synth_neutral.loc[same_mask, "hypothesis"].values, 1)
        synth_neutral.loc[same_mask, "hypothesis"] = rolled

    return {
        "synth_identity_entail": synth_identity.reset_index(drop=True),
        "synth_negation_contradict": synth_negation.reset_index(drop=True),
        "synth_mismatch_neutral": synth_neutral.reset_index(drop=True),
    }


def build_balanced_concept_stress_set(
    base_df: pd.DataFrame,
    n_per_concept: int,
    seed: int,
    concept_parser_enabled: bool,
    concept_min_confidence: float,
    lexicon: ConceptLexicon,
) -> pd.DataFrame:
    df = base_df.copy()
    concept_records = [
        analyze_concept(
            t,
            concept_parser_enabled=concept_parser_enabled,
            concept_min_confidence=concept_min_confidence,
            lexicon=lexicon,
        )
        for t in df["hypothesis"].tolist()
    ]
    df["concept_primary"] = [str(r["primary_concept"]) for r in concept_records]
    rng = np.random.default_rng(seed)
    parts: List[pd.DataFrame] = []

    for concept in CONCEPT_LABELS:
        cblock = df[df["concept_primary"] == concept]
        if cblock.empty:
            continue
        per_label = max(1, n_per_concept // 3)
        sampled: List[pd.DataFrame] = []
        for lbl in LABEL2ID.keys():
            lbl_block = cblock[cblock["label"] == lbl]
            if lbl_block.empty:
                continue
            take = int(min(per_label, len(lbl_block)))
            idx = rng.choice(lbl_block.index.to_numpy(), size=take, replace=False)
            sampled.append(cblock.loc[idx])
        if sampled:
            concept_df = pd.concat(sampled, ignore_index=False)
            if len(concept_df) > n_per_concept:
                concept_df = concept_df.sample(n=n_per_concept, random_state=seed).copy()
            parts.append(concept_df)
        else:
            take = int(min(n_per_concept, len(cblock)))
            parts.append(cblock.sample(n=take, random_state=seed).copy())

    if not parts:
        return df.head(0).copy()

    out = pd.concat(parts, ignore_index=False).drop_duplicates().reset_index(drop=True)
    return out[df.columns].reset_index(drop=True)


def extract_concept(text: str, max_tokens: int = 3) -> str:
    words = re.findall(r"[a-zA-Z]+", str(text).lower())
    filtered = [w for w in words if w not in CONCEPT_STOPWORDS and len(w) > 2]
    if not filtered:
        return "scene"
    return " ".join(filtered[:max_tokens])


def build_prompt(template: str, hypothesis: str) -> str:
    concept = extract_concept(hypothesis)
    return template.format(concept=concept, text=str(hypothesis).strip())


def _init_spacy_pipeline(enabled: bool = True) -> Tuple[Optional[Any], Optional[str], bool]:
    global _SPACY_NLP, _SPACY_MODEL, _SPACY_INIT
    if not enabled:
        return None, None, False
    if _SPACY_INIT:
        return _SPACY_NLP, _SPACY_MODEL, _SPACY_NLP is not None

    _SPACY_INIT = True
    try:
        import spacy  # type: ignore
    except Exception:
        return None, None, False

    for model_name in ("en_core_web_sm",):
        try:
            _SPACY_NLP = spacy.load(model_name, disable=["ner"])
            _SPACY_MODEL = model_name
            return _SPACY_NLP, _SPACY_MODEL, True
        except Exception:
            continue
    return None, None, False


def _init_score_dict() -> Dict[str, float]:
    return {k: 0.0 for k in CONCEPT_LABELS}


def _tokenize_words(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9']+", str(text).lower())


def _normalize_action_lemma(token: str) -> str:
    w = str(token).lower()
    if len(w) <= 3:
        return w
    if w.endswith("ing") and len(w) > 5:
        base = w[:-3]
        if len(base) >= 2 and base[-1] == base[-2]:
            base = base[:-1]
        return base
    if w.endswith("ied") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("ed") and len(w) > 4:
        base = w[:-2]
        if len(base) >= 2 and base[-1] == base[-2]:
            base = base[:-1]
        return base
    if w.endswith("es") and len(w) > 4:
        return w[:-2]
    if w.endswith("s") and len(w) > 4:
        return w[:-1]
    return w


def build_dataset_concept_lexicon(
    train_df: pd.DataFrame,
    min_freq: int = 8,
    top_k: int = 120,
) -> ConceptLexicon:
    lexicon = build_seed_concept_lexicon()
    texts = train_df["hypothesis"].dropna().astype(str).tolist()
    token_counter: Counter[str] = Counter()
    tokenized_texts: List[List[str]] = []
    for text in texts:
        toks = _tokenize_words(text)
        tokenized_texts.append(toks)
        token_counter.update(toks)

    if not token_counter:
        return lexicon
    min_effective = min_freq if len(texts) >= 1000 else 2

    verb_counter: Counter[str] = Counter()
    noun_counter: Counter[str] = Counter()
    adj_counter: Counter[str] = Counter()
    cd_counter: Counter[str] = Counter()
    pos_tagger = "fallback"

    try:
        import nltk  # type: ignore

        for toks in tokenized_texts:
            if not toks:
                continue
            for tok, tag in nltk.pos_tag(toks):
                tok_l = tok.lower()
                if token_counter[tok_l] < min_effective:
                    continue
                if tag.startswith("VB"):
                    verb_counter[tok_l] += 1
                elif tag.startswith("NN"):
                    noun_counter[tok_l] += 1
                elif tag.startswith("JJ"):
                    adj_counter[tok_l] += 1
                elif tag == "CD":
                    cd_counter[tok_l] += 1
        pos_tagger = "nltk_pos_tag"
    except Exception:
        # Lightweight fallback if POS tagging is unavailable.
        for tok, cnt in token_counter.items():
            if cnt < min_effective:
                continue
            if tok.endswith("ing") or tok.endswith("ed"):
                verb_counter[tok] += cnt
            if tok.endswith(("y", "ful", "ous", "ive", "al")):
                adj_counter[tok] += cnt
            if tok not in CONCEPT_STOPWORDS and len(tok) > 2:
                noun_counter[tok] += cnt
            if tok.isdigit() or tok in COUNT_WORDS:
                cd_counter[tok] += cnt

    # Dataset-driven updates; seed lexicon remains as a safety net.
    content_verb_counter = Counter({w: c for w, c in verb_counter.items() if w not in AUXILIARY_VERBS})

    derived_neg = {
        w for w, c in token_counter.items() if c >= min_effective and (w in NEGATION_TOKENS or w.endswith("n't"))
    }
    derived_count = {w for w, c in cd_counter.items() if c >= max(2, min_effective // 2)}
    derived_count.update({w for w, c in token_counter.items() if c >= min_effective and re.fullmatch(r"\d+", w)})
    derived_actions = {
        w
        for w, c in content_verb_counter.items()
        if c >= max(2, min_effective // 2)
        and (
            w in ACTION_FORMS
            or w in ACTION_LEMMAS
            or w.endswith("ing")
            or w.endswith("ed")
        )
    }
    derived_action_lemmas = {_normalize_action_lemma(w) for w in derived_actions}
    derived_action_lemmas = {w for w in derived_action_lemmas if len(w) >= 3}

    lexicon.negation_tokens.update(derived_neg)
    lexicon.count_words.update(derived_count)
    lexicon.action_forms.update(derived_actions)
    lexicon.action_lemmas.update(derived_action_lemmas)

    lexicon.top_nouns = [w for w, _ in noun_counter.most_common(top_k)]
    lexicon.top_adjectives = [w for w, _ in adj_counter.most_common(top_k)]
    if content_verb_counter:
        lexicon.top_verbs = [w for w, _ in content_verb_counter.most_common(top_k)]
    else:
        seed_action_hits = Counter(
            {w: token_counter[w] for w in lexicon.action_forms if w not in AUXILIARY_VERBS and token_counter[w] > 0}
        )
        lexicon.top_verbs = [w for w, _ in seed_action_hits.most_common(top_k)]
    lexicon.source = "dataset_mined"
    lexicon.pos_tagger = pos_tagger
    return lexicon


def concept_lexicon_to_dict(lexicon: ConceptLexicon) -> Dict[str, Any]:
    return {
        "source": lexicon.source,
        "pos_tagger": lexicon.pos_tagger,
        "negation_tokens": sorted(lexicon.negation_tokens),
        "count_words": sorted(lexicon.count_words),
        "action_lemmas": sorted(lexicon.action_lemmas),
        "action_forms": sorted(lexicon.action_forms),
        "top_nouns": lexicon.top_nouns,
        "top_adjectives": lexicon.top_adjectives,
        "top_verbs": lexicon.top_verbs,
    }


def _rule_concept_scores(text: str, lexicon: ConceptLexicon) -> Tuple[Dict[str, float], List[str]]:
    lowered = str(text).lower()
    words = _tokenize_words(lowered)
    word_set = set(words)
    bigrams = {" ".join(words[i : i + 2]) for i in range(max(len(words) - 1, 0))}
    trigrams = {" ".join(words[i : i + 3]) for i in range(max(len(words) - 2, 0))}
    joined_ngrams = bigrams.union(trigrams)

    scores = _init_score_dict()
    cues: List[str] = []

    neg_hits = [w for w in words if w in lexicon.negation_tokens or w.endswith("n't")]
    if neg_hits:
        scores["negation"] += min(2.0, float(len(set(neg_hits))))
        cues.extend([f"rule:negation:{w}" for w in sorted(set(neg_hits))[:3]])

    phrase_hits = [p for p in SPATIAL_PHRASES if p in lowered]
    single_hits = [w for w in SPATIAL_SINGLE if w in word_set]
    if phrase_hits or single_hits:
        scores["spatial"] += min(2.0, 0.8 * len(phrase_hits) + 0.5 * len(single_hits))
        cues.extend([f"rule:spatial:{p}" for p in sorted(phrase_hits)[:3]])
        cues.extend([f"rule:spatial:{w}" for w in sorted(single_hits)[:3]])

    count_hits = [w for w in words if w in lexicon.count_words]
    has_numeric = bool(re.search(r"\b\d+\b", lowered))
    if count_hits or has_numeric:
        scores["count"] += min(2.0, float(has_numeric) + 0.5 * len(set(count_hits)))
        if has_numeric:
            cues.append("rule:count:digit")
        cues.extend([f"rule:count:{w}" for w in sorted(set(count_hits))[:3]])

    action_hits = []
    for w in words:
        lw = _normalize_action_lemma(w)
        if w in lexicon.action_forms or lw in lexicon.action_lemmas:
            action_hits.append(w)
    if any(phrase in lowered for phrase in ACTION_BLACKLIST_PHRASES):
        action_hits = []
    if "standing" in word_set and ("desk" in word_set or "table" in word_set):
        action_hits = [w for w in action_hits if w != "standing"]
    if "running out of" in lowered:
        action_hits = [w for w in action_hits if w not in {"run", "running"}]
    if "look" in action_hits and "at" in word_set:
        action_hits = [w for w in action_hits if w != "look"]
    if action_hits:
        # Avoid treating "in front of" as action evidence when words overlap.
        overlap_penalty = 0.3 if "in front of" in joined_ngrams else 0.0
        scores["action"] += max(0.0, min(2.0, 0.6 * len(set(action_hits)) - overlap_penalty))
        cues.extend([f"rule:action:{w}" for w in sorted(set(action_hits))[:4]])

    if max(scores.values()) <= 0:
        scores["object_or_attribute"] = 1.0
        cues.append("rule:fallback:object_or_attribute")
    return scores, cues


def _dependency_concept_scores(doc: Any, lexicon: ConceptLexicon) -> Tuple[Dict[str, float], List[str]]:
    scores = _init_score_dict()
    cues: List[str] = []

    for tok in doc:
        lemma = tok.lemma_.lower()
        text = tok.text.lower()
        dep = tok.dep_.lower()
        pos = tok.pos_.upper()

        if dep == "neg" or text in lexicon.negation_tokens:
            scores["negation"] += 1.8
            cues.append(f"dep:negation:{tok.text}")

        if dep == "nummod" or (getattr(tok, "like_num", False) and tok.head.pos_ in {"NOUN", "PROPN"}):
            scores["count"] += 1.8
            cues.append(f"dep:count:{tok.text}")

        if dep in {"prep", "advmod"} and lemma in SPATIAL_PREPS.union(SPATIAL_SINGLE):
            scores["spatial"] += 1.1
            cues.append(f"dep:spatial:{tok.text}")
        if dep == "pobj" and tok.head.dep_.lower() == "prep" and tok.head.lemma_.lower() in SPATIAL_PREPS:
            scores["spatial"] += 0.9
            cues.append(f"dep:spatial:{tok.head.text}_{tok.text}")

        is_action_verb = (pos in {"VERB", "AUX"}) and (lemma in lexicon.action_lemmas or text in lexicon.action_forms)
        if is_action_verb:
            if lemma == "run" and any(child.lemma_.lower() == "out" for child in tok.children):
                continue
            if dep in {"amod", "compound"}:
                continue
            scores["action"] += 1.1
            cues.append(f"dep:action:{tok.text}")

    if max(scores.values()) <= 0:
        scores["object_or_attribute"] = 1.0
        cues.append("dep:fallback:object_or_attribute")
    return scores, cues


def _pick_primary_concept(scores: Dict[str, float]) -> str:
    return sorted(scores.keys(), key=lambda k: (-scores[k], CONCEPT_PRIORITY[k]))[0]


def _merge_concept_scores(
    rule_scores: Dict[str, float],
    dep_scores: Optional[Dict[str, float]],
    min_confidence: float,
) -> Tuple[Dict[str, float], str, float]:
    dep_weight = 1.4
    merged = _init_score_dict()
    for label in CONCEPT_LABELS:
        merged[label] = rule_scores.get(label, 0.0)
        if dep_scores is not None:
            merged[label] += dep_weight * dep_scores.get(label, 0.0)

    total = sum(v for v in merged.values() if v > 0)
    if total <= 0:
        merged["object_or_attribute"] = 1.0
        return merged, "object_or_attribute", 1.0

    primary = _pick_primary_concept(merged)
    confidence = float(merged[primary] / (total + 1e-9))
    if confidence < min_confidence:
        primary = "object_or_attribute"
    return merged, primary, confidence


def analyze_concept(
    text: str,
    concept_parser_enabled: bool = True,
    concept_min_confidence: float = 0.35,
    lexicon: Optional[ConceptLexicon] = None,
) -> Dict[str, Any]:
    if lexicon is None:
        lexicon = build_seed_concept_lexicon()
    nlp, _, parser_available = _init_spacy_pipeline(enabled=concept_parser_enabled)
    rule_scores, rule_cues = _rule_concept_scores(text, lexicon=lexicon)

    dep_scores: Optional[Dict[str, float]] = None
    dep_cues: List[str] = []
    if parser_available and nlp is not None:
        dep_scores, dep_cues = _dependency_concept_scores(nlp(str(text)), lexicon=lexicon)

    merged, primary, confidence = _merge_concept_scores(
        rule_scores=rule_scores,
        dep_scores=dep_scores,
        min_confidence=concept_min_confidence,
    )
    positive = [(k, v) for k, v in merged.items() if v > 0]
    all_concepts = [k for k, _ in sorted(positive, key=lambda it: (-it[1], CONCEPT_PRIORITY[it[0]]))]
    if not all_concepts:
        all_concepts = ["object_or_attribute"]
    if primary not in all_concepts:
        all_concepts.insert(0, primary)

    rule_total = sum(rule_scores.values())
    dep_total = sum(dep_scores.values()) if dep_scores is not None else 0.0
    if dep_scores is None and rule_total > 0:
        source = "rule"
    elif dep_total > 0 and rule_total <= 0:
        source = "dependency"
    elif dep_total > 0 and rule_total > 0:
        source = "hybrid_fallback"
    else:
        source = "hybrid_fallback"

    if primary == "object_or_attribute" and confidence < concept_min_confidence:
        source = "hybrid_fallback"

    return {
        "primary_concept": primary,
        "all_concepts": all_concepts,
        "concept_confidence": float(confidence),
        "evidence_source": source,
        "matched_cues": {"rule": rule_cues, "dependency": dep_cues},
    }


def concept_type(text: str) -> str:
    return str(analyze_concept(text)["primary_concept"])


class FallbackImageProcessor:
    def __init__(self, size: int = 224):
        self.size = size
        self.mean = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)

    def __call__(self, images, return_tensors: str = "pt"):
        arrs = []
        for img in images:
            x = img.convert("RGB").resize((self.size, self.size))
            x = np.asarray(x, dtype=np.float32) / 255.0
            x = x.transpose(2, 0, 1)
            x = (x - self.mean) / self.std
            arrs.append(x)
        pixel_values = torch.tensor(np.stack(arrs), dtype=torch.float32)
        return {"pixel_values": pixel_values} if return_tensors == "pt" else pixel_values


class FrozenViTTextEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.vit = AutoModel.from_pretrained(cfg.vision_model_name)
        self.bert = AutoModel.from_pretrained(cfg.text_model_name)
        self.image_processor = self._build_image_processor()
        self.tokenizer = self._build_tokenizer()

        for p in self.vit.parameters():
            p.requires_grad = False
        for p in self.bert.parameters():
            p.requires_grad = False

        self.out_dim = self.vit.config.hidden_size + self.bert.config.hidden_size

    def _build_image_processor(self):
        try:
            return AutoImageProcessor.from_pretrained(self.cfg.vision_model_name, local_files_only=True)
        except Exception:
            return FallbackImageProcessor(size=224)

    def _build_tokenizer(self):
        try:
            return AutoTokenizer.from_pretrained(self.cfg.text_model_name, local_files_only=True)
        except Exception:
            return AutoTokenizer.from_pretrained(self.cfg.text_model_name)

    def load_encoder_weights_from_checkpoint(self, ckpt_path: Path, device: torch.device) -> Dict[str, int]:
        if not ckpt_path.exists():
            return {"vit_loaded": 0, "bert_loaded": 0}

        state = torch.load(ckpt_path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if not isinstance(state, dict):
            return {"vit_loaded": 0, "bert_loaded": 0}

        vit_state: Dict[str, torch.Tensor] = {}
        bert_state: Dict[str, torch.Tensor] = {}
        for k, v in state.items():
            if k.startswith("vit."):
                vit_state[k.replace("vit.", "", 1)] = v
            elif k.startswith("bert."):
                bert_state[k.replace("bert.", "", 1)] = v

        vit_loaded = 0
        bert_loaded = 0

        if vit_state:
            _, unexpected = self.vit.load_state_dict(vit_state, strict=False)
            vit_loaded = len(vit_state) - len(unexpected)

        if bert_state:
            _, unexpected = self.bert.load_state_dict(bert_state, strict=False)
            bert_loaded = len(bert_state) - len(unexpected)

        return {"vit_loaded": vit_loaded, "bert_loaded": bert_loaded}


class BottleneckAdapter(nn.Module):
    def __init__(self, in_dim: int, bottleneck_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.down = nn.Linear(in_dim, bottleneck_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck_dim, in_dim)
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.down(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.up(x)
        x = residual + self.scale * x
        return self.norm(x)


# ==========================================
# 🚨 SURGICALLY GRAFTED SWIGLU ARCHITECTURE 🚨
# ==========================================

class SwiGLU_MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout_rate=0.1):
        super().__init__()
        self.w12 = nn.Linear(in_features, hidden_features * 2)
        self.w3 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2 # The "Gate" that filters prompt noise!
        hidden = self.dropout(hidden)
        return self.w3(hidden)

class AdapterClassifier(nn.Module):
    def __init__(
        self,
        in_dim: int,
        bottleneck_dim: int = 32,
        dropout: float = 0.1,
        num_classes: int = 3,
    ):
        super().__init__()
        # 1. The fast, lightweight bottleneck adapter
        self.adapter = BottleneckAdapter(in_dim, bottleneck_dim=bottleneck_dim, dropout=dropout)
        
        # 2. Your advanced SwiGLU classifier replaces the vanilla Linear layer!
        self.classifier = SwiGLU_MLP(
            in_features=in_dim,
            hidden_features=in_dim // 2, 
            out_features=num_classes,
            dropout_rate=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Data flows from the cached features -> Adapter -> SwiGLU -> Logits
        x = self.adapter(x)
        return self.classifier(x)


def trainable_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class FeatureBuilder:
    def __init__(self, encoder: FrozenViTTextEncoder, device: torch.device):
        self.encoder = encoder.to(device).eval()
        self.device = device
        self.image_cache: Dict[str, torch.Tensor] = {}

    def _encode_images(self, image_paths: List[str], batch_size: int) -> torch.Tensor:
        missing = [p for p in image_paths if p not in self.image_cache]

        amp_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        amp_enabled = self.device.type == "cuda"

        for i in range(0, len(missing), batch_size):
            batch_paths = missing[i : i + batch_size]
            images = []
            for p in batch_paths:
                with Image.open(p) as im:
                    images.append(im.convert("RGB"))
            with torch.autocast(device_type=self.device.type, enabled=amp_enabled, dtype=amp_dtype):
                pixel_values = self.encoder.image_processor(images=images, return_tensors="pt")["pixel_values"].to(self.device)
                vit_out = self.encoder.vit(pixel_values=pixel_values).last_hidden_state[:, 0, :]
            vit_out = vit_out.detach().to("cpu")
            for path, feat in zip(batch_paths, vit_out):
                self.image_cache[path] = feat

        return torch.stack([self.image_cache[p] for p in image_paths], dim=0)

    def _encode_texts(self, texts: List[str], batch_size: int) -> torch.Tensor:
        out = []
        amp_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        amp_enabled = self.device.type == "cuda"

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            toks = self.encoder.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.encoder.cfg.max_text_len,
                return_tensors="pt",
            )
            toks = {k: v.to(self.device) for k, v in toks.items()}
            with torch.autocast(device_type=self.device.type, enabled=amp_enabled, dtype=amp_dtype):
                bert_out = self.encoder.bert(**toks).last_hidden_state[:, 0, :]
            out.append(bert_out.detach().to("cpu"))

        return torch.cat(out, dim=0)

    def build_features(self, df: pd.DataFrame, template: str, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        prompts = [build_prompt(template, h) for h in df["hypothesis"].tolist()]
        image_paths = df["image_path"].tolist()

        vit_cls = self._encode_images(image_paths=image_paths, batch_size=max(8, batch_size // 2))
        bert_cls = self._encode_texts(prompts, batch_size=batch_size)

        feats = torch.cat([vit_cls, bert_cls], dim=1)
        feats = nn.functional.normalize(feats.float(), p=2, dim=1)

        labels = torch.tensor(df["label"].map(LABEL2ID).values, dtype=torch.long)
        return feats, labels


def macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 3) -> float:
    f1s: List[float] = []
    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        f1s.append(f1)
    return float(np.mean(f1s))


def evaluate_head(
    head: AdapterClassifier,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
) -> Dict[str, float]:
    head.eval().to(device)
    preds = []
    for i in range(0, len(x), batch_size):
        xb = x[i : i + batch_size].to(device, non_blocking=True)
        logits = head(xb)
        preds.append(torch.argmax(logits, dim=1).to("cpu"))

    pred = torch.cat(preds)
    acc = ((pred == y).float().mean().item())
    macro_f1 = macro_f1_score(y.numpy(), pred.numpy(), num_classes=3)
    return {"acc": acc, "macro_f1": macro_f1}


def train_adapter(
    head: AdapterClassifier,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    valid_x: torch.Tensor,
    valid_y: torch.Tensor,
    device: torch.device,
    epochs: int = 8,
    batch_size: int = 256,
    lr: float = 2e-3,
    weight_decay: float = 1e-3,
    verbose: bool = False,
    patience: int = 3, # The early stopping counter limit!
) -> Dict[str, float]:
    head = head.to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_state = copy.deepcopy(head.state_dict())
    best_val_loss = float('inf')
    best_acc = -1.0
    no_improve = 0

    train_ds = torch.utils.data.TensorDataset(train_x, train_y)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    valid_ds = torch.utils.data.TensorDataset(valid_x, valid_y)
    valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

    for epoch in range(1, epochs + 1):
        if verbose:
            print(f"\n--- Epoch {epoch}/{epochs} ---")
            
        # ---------------- TRAINING ----------------
        head.train()
        train_loss, correct_train, total_train = 0.0, 0, 0
        
        # We only show the progress bar if verbose=True so it doesn't spam your 1-shot loops!
        train_pbar = tqdm(train_loader, desc="Training  ", disable=not verbose, 
                          bar_format="{l_bar}{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")
        
        for xb, yb in train_pbar:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            logits = head(xb)
            loss = criterion(logits, yb)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * xb.size(0)
            preds = torch.argmax(logits, dim=1)
            correct_train += (preds == yb).sum().item()
            total_train += yb.size(0)
            
        avg_train_loss = train_loss / total_train
        train_acc = (correct_train / total_train) * 100

        # --------------- VALIDATING ---------------
        head.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        
        val_pbar = tqdm(valid_loader, desc="Validating", disable=not verbose, 
                        bar_format="{l_bar}{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")
        
        with torch.no_grad():
            for xb, yb in val_pbar:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                logits = head(xb)
                loss = criterion(logits, yb)

                val_loss += loss.item() * xb.size(0)
                preds = torch.argmax(logits, dim=1)
                correct_val += (preds == yb).sum().item()
                total_val += yb.size(0)

        avg_val_loss = val_loss / total_val
        val_acc = (correct_val / total_val) * 100

        # --------------- METRICS & SAVING ---------------
        if verbose:
            print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_acc = val_acc
            best_state = copy.deepcopy(head.state_dict())
            no_improve = 0
            if verbose:
                print(f"   🌟 New best model saved! (Val Loss: {avg_val_loss:.4f})")
        else:
            no_improve += 1
            if verbose:
                print(f"   ⚠️ No improvement. Early stopping counter: {no_improve}/{patience}")
            
            if no_improve >= patience:
                break

    # Load the best weights back before returning
    head.load_state_dict(best_state)
    
    # We return the accuracies divided by 100 so the rest of your script mathematically aligns!
    return {"best_valid_acc": best_acc / 100.0, "final_valid_acc": val_acc / 100.0}


def pick_one_shot_indices_from_labels(labels: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray, int]:
    rng = np.random.default_rng(seed)
    support_idx: List[int] = []
    label_coverage = 0
    for class_id in (0, 1, 2):
        idx = np.where(labels == class_id)[0]
        if len(idx) == 0:
            continue
        label_coverage += 1
        support_idx.append(int(rng.choice(idx)))

    support_mask = np.zeros(len(labels), dtype=bool)
    if support_idx:
        support_mask[np.array(support_idx, dtype=int)] = True

    return np.where(support_mask)[0], np.where(~support_mask)[0], int(label_coverage)


def run_one_shot_eval(
    base_head: AdapterClassifier,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    seed: int,
) -> Tuple[Dict[str, float], str, int]:
    y_np = y.numpy()
    support_idx, query_idx, coverage = pick_one_shot_indices_from_labels(y_np, seed=seed)

    if coverage < 3:
        return {"acc": float("nan"), "macro_f1": float("nan")}, "insufficient_class_coverage", coverage
    if len(query_idx) == 0:
        return {"acc": float("nan"), "macro_f1": float("nan")}, "insufficient_query_examples", coverage

    sx = x[support_idx]
    sy = y[support_idx]
    qx = x[query_idx]
    qy = y[query_idx]

    one_shot_head = copy.deepcopy(base_head)
    _ = train_adapter(
        head=one_shot_head,
        train_x=sx,
        train_y=sy,
        valid_x=sx,
        valid_y=sy,
        device=device,
        epochs=20,
        batch_size=3,
        lr=1e-3,
        weight_decay=0.0,
        verbose=False,
    )
    one = evaluate_head(one_shot_head, qx, qy, device=device)
    return one, "ok", coverage


def summarize_concept_tags(tag_lists: List[List[str]], top_k: int = 3) -> str:
    flat: List[str] = []
    for tags in tag_lists:
        for tag in tags:
            flat.append(str(tag))
    if not flat:
        return "object_or_attribute"
    counts = Counter(flat)
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], CONCEPT_PRIORITY.get(kv[0], 99), kv[0]))
    return "|".join([name for name, _ in ordered[:top_k]])


def write_prompt_guide(results: pd.DataFrame, out_path: Path) -> None:
    lines: List[str] = []
    lines.append("# Prompt Engineering Guide (Task 4)\n")
    lines.append("## Global Winners\n")

    g = (
        results.groupby(["template"])["zero_shot_acc"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    for _, row in g.iterrows():
        lines.append(f"- `{row['template']}`: mean zero-shot acc = {row['zero_shot_acc']:.4f}")

    lines.append("\n## By Concept Type\n")
    for ctype, block in results.groupby("concept_type"):
        rank = block.groupby("template")["zero_shot_acc"].mean().sort_values(ascending=False)
        best_t = rank.index[0]
        best_v = rank.iloc[0]
        if "bucket_size" in block.columns:
            support = int(block["bucket_size"].median())
            lines.append(f"- `{ctype}` best template: `{best_t}` ({best_v:.4f}), median bucket size={support}")
        else:
            lines.append(f"- `{ctype}` best template: `{best_t}` ({best_v:.4f})")

    if "bucket_size" in results.columns:
        lines.append("\n## Reliability Notes\n")
        for ctype, block in results.groupby("concept_type"):
            med = int(block["bucket_size"].median())
            if med < 40:
                lines.append(f"- `{ctype}` has low support (median bucket size={med}); interpret rankings cautiously.")

    lines.append("\n## Practical Recommendations\n")
    lines.append("- Use `high_specificity` for complex hypotheses requiring exact taxonomic object matching (e.g., specific car models).")
    lines.append("- Use `low_specificity` to allow the model to generalize broad categories.")
    lines.append("- Use `negation_explicit` for hypotheses that contain negation or polarity cues.")
    lines.append("- Use `concept_framing` when the hypothesis is short and concept extraction is clean.")
    lines.append("- Keep `base` as the fastest default baseline for broad transfer.")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_failure_report(diagnostics: pd.DataFrame, out_path: Path) -> None:
    lines: List[str] = []
    lines.append("# What Till We Do Wrong (Task 4)\n")
    if diagnostics.empty:
        lines.append("No diagnostics rows were generated.")
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return

    total = len(diagnostics)
    wrong = diagnostics[diagnostics["pred_zero_id"] != diagnostics["true_label_id"]].copy()
    err_rate = len(wrong) / max(total, 1)
    lines.append(f"- Total evaluated rows: {total}")
    lines.append(f"- Zero-shot error rate: {err_rate:.4f}")

    lines.append("\n## Worst Concept/Template Buckets\n")
    if wrong.empty:
        lines.append("- No zero-shot errors found.")
    else:
        grouped = (
            diagnostics.groupby(["concept_type", "template"])
            .agg(total=("true_label_id", "size"), errors=("is_error", "sum"))
            .reset_index()
        )
        grouped["error_rate"] = grouped["errors"] / grouped["total"].clip(lower=1)
        grouped = grouped.sort_values(["error_rate", "errors"], ascending=[False, False]).head(10)
        for _, row in grouped.iterrows():
            lines.append(
                f"- `{row['concept_type']}` + `{row['template']}`: "
                f"error_rate={row['error_rate']:.4f} ({int(row['errors'])}/{int(row['total'])})"
            )

    lines.append("\n## Frequent Confusions\n")
    if wrong.empty:
        lines.append("- No confusion pairs to report.")
    else:
        conf = (
            wrong.groupby(["true_label", "pred_zero_label"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(10)
        )
        for _, row in conf.iterrows():
            lines.append(f"- true `{row['true_label']}` -> predicted `{row['pred_zero_label']}`: {int(row['count'])}")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def run_task4(
    base_dir: str,
    train_sample: int = 12000,
    dev_sample: int = 3000,
    eval_sample: int = 1500,
    batch_size: int = 64,
    seed: int = 42,
    concept_parser_enabled: bool = True,
    concept_min_confidence: float = 0.35,
    verbose: bool = True,
):
    seed_everything(seed)
    base = Path(base_dir).resolve()
    device = get_device()
    if verbose:
        print(f"[task4] device={device}")

    history = workspace_history_summary(base)
    if verbose:
        print("[task4] loading datasets...")

    _, spacy_model, spacy_available = _init_spacy_pipeline(enabled=concept_parser_enabled)
    if concept_parser_enabled and spacy_available:
        concept_mode_used = "hybrid_dependency"
    elif concept_parser_enabled:
        concept_mode_used = "rule_only_fallback"
    else:
        concept_mode_used = "parser_disabled_rule_only"
    if verbose:
        print(f"[task4] concept mode={concept_mode_used} model={spacy_model}")
        if concept_parser_enabled and not spacy_available:
            print("[task4] spaCy parser unavailable. Install `spacy` + `en_core_web_sm` for full hybrid parsing.")

    train_df = sample_df(load_clean_df(base / "cleaned_snli_ve_train.csv"), train_sample, seed=seed)
    dev_df = sample_df(load_clean_df(base / "cleaned_snli_ve_dev.csv"), dev_sample, seed=seed)
    test_df = sample_df(load_clean_df(base / "cleaned_snli_ve_test.csv"), eval_sample, seed=seed)
    heldout_dev = sample_df(load_clean_df(base / "cleaned_snli_ve_dev.csv"), eval_sample, seed=seed + 1)
    concept_lexicon = build_dataset_concept_lexicon(train_df)
    concept_lexicon_path = base / "task4_concept_lexicons.json"
    concept_lexicon_path.write_text(
        json.dumps(concept_lexicon_to_dict(concept_lexicon), indent=2),
        encoding="utf-8",
    )
    if verbose:
        print(
            "[task4] concept lexicon mined from train sample "
            f"(source={concept_lexicon.source}, tagger={concept_lexicon.pos_tagger})"
        )
        print(f"[task4] top verbs (sample): {concept_lexicon.top_verbs[:12]}")
        print(f"[task4] top nouns (sample): {concept_lexicon.top_nouns[:12]}")
        print(f"[task4] top adjectives (sample): {concept_lexicon.top_adjectives[:12]}")

    heldout_sets: Dict[str, pd.DataFrame] = {
        "snli_dev_holdout": heldout_dev,
        "snli_test_holdout": test_df,
    }
    heldout_sets.update(build_synthetic_sets(test_df, n_each=min(eval_sample, len(test_df)), seed=seed))
    stress_per_concept = max(30, min(180, eval_sample // 6))
    stress_df = build_balanced_concept_stress_set(
        test_df,
        n_per_concept=stress_per_concept,
        seed=seed + 7,
        concept_parser_enabled=concept_parser_enabled,
        concept_min_confidence=concept_min_confidence,
        lexicon=concept_lexicon,
    )
    if len(stress_df) > 0:
        heldout_sets["stress_balanced_concepts"] = stress_df
    if verbose:
        print(f"[task4] eval datasets: {list(heldout_sets.keys())}")

    encoder = FrozenViTTextEncoder(EncoderConfig())
    loaded = {"vit_loaded": 0, "bert_loaded": 0}
    used_ckpt: Optional[str] = None
    for ckpt_name in CHECKPOINT_CANDIDATES:
        ckpt_path = base / ckpt_name
        if not ckpt_path.exists():
            continue
        used_ckpt = ckpt_name
        loaded = encoder.load_encoder_weights_from_checkpoint(ckpt_path, device=device)
        break
    if verbose:
        print(f"[task4] checkpoint used: {used_ckpt}")
        print(f"[task4] encoder weights loaded: {loaded}")

    feature_builder = FeatureBuilder(encoder=encoder, device=device)
    default_template = PROMPT_TEMPLATES["base"]

    train_x, train_y = feature_builder.build_features(train_df, template=default_template, batch_size=batch_size)
    dev_x, dev_y = feature_builder.build_features(dev_df, template=default_template, batch_size=batch_size)
    if verbose:
        print("[task4] feature extraction complete for train/dev")

    head = AdapterClassifier(
        in_dim=encoder.out_dim,
        bottleneck_dim=32,
        dropout=0.1,
        num_classes=3,
    )
    adapter_stats = train_adapter(
        head=head,
        train_x=train_x,
        train_y=train_y,
        valid_x=dev_x,
        valid_y=dev_y,
        device=device,
        epochs=8,
        batch_size=256,
        verbose=verbose,
    )
    if verbose:
        print(f"[task4] adapter trained. stats={adapter_stats}")

    rows: List[Dict[str, object]] = []
    diagnostics_rows: List[Dict[str, object]] = []

    for ds_i, (ds_name, ds_df) in enumerate(heldout_sets.items()):
        ds_df = ds_df.copy()
        concept_records = [
            analyze_concept(
                t,
                concept_parser_enabled=concept_parser_enabled,
                concept_min_confidence=concept_min_confidence,
                lexicon=concept_lexicon,
            )
            for t in ds_df["hypothesis"].tolist()
        ]
        ds_df["concept_primary"] = [str(r["primary_concept"]) for r in concept_records]
        ds_df["all_concepts"] = [list(r["all_concepts"]) for r in concept_records]
        ds_df["concept_confidence"] = [float(r["concept_confidence"]) for r in concept_records]
        ds_df["concept_source"] = [str(r["evidence_source"]) for r in concept_records]
        ds_df["matched_cues"] = [json.dumps(r["matched_cues"], sort_keys=True) for r in concept_records]
        ds_df["concept_type"] = ds_df["concept_primary"]  # Backward-compatible alias.
        if verbose:
            print(f"[task4] evaluating dataset={ds_name} rows={len(ds_df)}")

        for t_i, (template_name, template) in enumerate(PROMPT_TEMPLATES.items()):
            x, y = feature_builder.build_features(ds_df, template=template, batch_size=batch_size)
            zero = evaluate_head(head, x, y, device=device)
            with torch.no_grad():
                head.eval().to(device)
                pred_zero = torch.argmax(head(x.to(device, non_blocking=True)), dim=1).cpu()

            global_one, global_status, global_coverage = run_one_shot_eval(
                base_head=head,
                x=x,
                y=y,
                device=device,
                seed=seed + (ds_i * 101) + t_i,
            )
            if verbose:
                global_one_acc = "nan" if np.isnan(global_one["acc"]) else f"{global_one['acc']:.4f}"
                print(
                    f"[task4]  template={template_name} zero_acc={zero['acc']:.4f} "
                    f"zero_f1={zero['macro_f1']:.4f} "
                    f"global_one_shot={global_one_acc}"
                )

            y_np = y.numpy()
            pred_zero_np = pred_zero.numpy()
            for row_idx, (y_true_id, y_pred_id) in enumerate(zip(y_np.tolist(), pred_zero_np.tolist())):
                row = ds_df.iloc[row_idx]
                concept_tags = "|".join(row["all_concepts"]) if isinstance(row["all_concepts"], list) else str(row["all_concepts"])
                diagnostics_rows.append(
                    {
                        "dataset": ds_name,
                        "template": template_name,
                        "row_index": int(row_idx),
                        "hypothesis": str(row["hypothesis"]),
                        "true_label": str(row["label"]),
                        "true_label_id": int(y_true_id),
                        "pred_zero_label": ID2LABEL[int(y_pred_id)],
                        "pred_zero_id": int(y_pred_id),
                        "is_error": int(y_true_id != y_pred_id),
                        "concept_type": str(row["concept_type"]),
                        "concept_tags": concept_tags,
                        "concept_confidence": float(row["concept_confidence"]),
                        "concept_source": str(row["concept_source"]),
                        "matched_cues": str(row["matched_cues"]),
                    }
                )

            concept_groups = sorted(
                ds_df.groupby("concept_type"),
                key=lambda it: CONCEPT_PRIORITY.get(str(it[0]), 99),
            )
            for c_i, (ctype, block) in enumerate(concept_groups):
                idx = block.index.to_numpy()
                y_true_c = y[idx].numpy()
                y_pred_c = pred_zero[idx].numpy()
                z_acc = float((pred_zero[idx] == y[idx]).float().mean().item())
                z_f1 = macro_f1_score(y_true_c, y_pred_c, num_classes=3)
                bucket_x = x[idx]
                bucket_y = y[idx]
                one, one_status, label_coverage = run_one_shot_eval(
                    base_head=head,
                    x=bucket_x,
                    y=bucket_y,
                    device=device,
                    seed=seed + (ds_i * 10007) + (t_i * 113) + c_i,
                )
                source_mode = (
                    block["concept_source"].mode().iloc[0]
                    if not block["concept_source"].mode().empty
                    else "hybrid_fallback"
                )
                concept_tags = summarize_concept_tags(block["all_concepts"].tolist())
                rows.append(
                    {
                        "dataset": ds_name,
                        "template": template_name,
                        "concept_type": ctype,
                        "concept_primary": ctype,
                        "concept_tags": concept_tags,
                        "concept_confidence_mean": float(block["concept_confidence"].mean()),
                        "concept_source_mode": str(source_mode),
                        "bucket_size": int(len(block)),
                        "label_coverage": int(label_coverage),
                        "one_shot_status": one_status,
                        "global_one_shot_acc": global_one["acc"],
                        "global_one_shot_macro_f1": global_one["macro_f1"],
                        "global_one_shot_status": global_status,
                        "global_label_coverage": int(global_coverage),
                        "zero_shot_acc": z_acc,
                        "zero_shot_macro_f1": z_f1,
                        "one_shot_acc": one["acc"],
                        "one_shot_macro_f1": one["macro_f1"],
                    }
                )

    results = pd.DataFrame(rows)
    results_path = base / "task4_results.csv"
    results.to_csv(results_path, index=False)
    diagnostics = pd.DataFrame(diagnostics_rows)
    diagnostics_path = base / "task4_concept_diagnostics.csv"
    diagnostics.to_csv(diagnostics_path, index=False)

    write_prompt_guide(results, out_path=base / "prompt_engineering_guide_task4.md")
    failure_report_path = base / "what_till_we_do_wrong.md"
    write_failure_report(diagnostics, out_path=failure_report_path)
    # ==========================================
    # 💾 SAVE THE TRAINED ADAPTER WEIGHTS
    # ==========================================
    adapter_save_path = base / "task4_swiglu_adapter_best.pth"
    torch.save(head.state_dict(), adapter_save_path)
    if verbose:
        print(f"[task4] 💾 Successfully saved trained adapter weights to {adapter_save_path}")

    summary = {
        "device": str(device),
        "used_checkpoint": used_ckpt,
        "loaded_encoder_keys": loaded,
        "adapter_trainable_params": trainable_params(head),
        "adapter_stats": adapter_stats,
        "results_path": str(results_path),
        "history_counts": {
            "notebooks": len(history["notebooks"]),
            "checkpoints": len(history["checkpoints"]),
            "datasets": len(history["datasets"]),
        },
        "spacy_available": bool(spacy_available),
        "spacy_model": spacy_model,
        "concept_mode_used": concept_mode_used,
        "concept_parser_enabled": concept_parser_enabled,
        "concept_min_confidence": concept_min_confidence,
        "concept_lexicon_source": concept_lexicon.source,
        "concept_lexicon_pos_tagger": concept_lexicon.pos_tagger,
        "concept_lexicon_path": str(concept_lexicon_path),
        "top_mined_verbs": concept_lexicon.top_verbs[:25],
        "top_mined_nouns": concept_lexicon.top_nouns[:25],
        "top_mined_adjectives": concept_lexicon.top_adjectives[:25],
        "concept_diagnostics_path": str(diagnostics_path),
        "failure_report_path": str(failure_report_path),
        "eval_datasets": list(heldout_sets.keys()),
        "templates": list(PROMPT_TEMPLATES.keys()),
        "adapter_weights_path": str(adapter_save_path), # <--- Add this line!
        "adapter_trainable_params": trainable_params(head),
    }

    (base / "task4_run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    del train_x, train_y, dev_x, dev_y
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"summary": summary, "results": results}
