"""Minimal classifier adapter for a caller-supplied DINOv3-compatible model."""

from __future__ import annotations

import torch
from torch import nn


class DinoV3BinaryClassifier(nn.Module):
    """Concatenate CLS and mean patch features, then predict two classes.

    The caller supplies the backbone and handles preprocessing, model loading,
    checkpoint terms, and any task-specific labels. No model is downloaded here.
    """

    def __init__(self, backbone: nn.Module, hidden_size: int, num_register_tokens: int = 0,
                 dropout: float = 0.1) -> None:
        super().__init__()
        if hidden_size < 1 or num_register_tokens < 0 or not 0 <= dropout < 1:
            raise ValueError("invalid hidden size, register-token count, or dropout")
        self.backbone = backbone
        self.num_register_tokens = num_register_tokens
        self.classifier = nn.Sequential(nn.LayerNorm(hidden_size * 2), nn.Dropout(dropout),
                                        nn.Linear(hidden_size * 2, 2))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        output = self.backbone(pixel_values=pixel_values)
        tokens = output.last_hidden_state
        first_patch = 1 + self.num_register_tokens
        if tokens.ndim != 3 or tokens.shape[1] <= first_patch:
            raise ValueError("backbone output must contain CLS, optional register tokens, and patch tokens")
        cls_token = tokens[:, 0]
        patch_mean = tokens[:, first_patch:].mean(dim=1)
        return self.classifier(torch.cat((cls_token, patch_mean), dim=-1))


def apply_lora(model: nn.Module, target_modules: list[str], *, rank: int = 8,
               alpha: int = 16, dropout: float = 0.05) -> nn.Module:
    """Freeze base parameters and apply PEFT LoRA to caller-selected modules."""
    if not target_modules or rank < 1 or alpha < 1 or not 0 <= dropout < 1:
        raise ValueError("provide target modules and valid LoRA parameters")
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise RuntimeError("Install the optional fine-tuning dependencies: pip install torch peft") from exc
    for parameter in model.parameters():
        parameter.requires_grad = False
    config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout,
                        target_modules=target_modules, modules_to_save=["classifier"], bias="none")
    return get_peft_model(model, config)
