from __future__ import annotations

from pathlib import Path


SUPPORTED_BACKBONE_CHECKPOINTS = (
    "final_sota_visual_entailment.pth",
    "final_sota_visual_entailment2.pth",
    "final_sota_visual_entailment3.pth",
    "sota_visual_entailment.pth",
    "sota_visual_entailment2.pth",
    "saved_model_acc_58.0.pth",
    "best_model_acc_70.7.pth",
    "best_model_acc_73.7.pth",
)


def list_backbone_checkpoints() -> list[str]:
    return [name for name in SUPPORTED_BACKBONE_CHECKPOINTS if Path(name).exists()]
