"""Shared helpers used by all three pipeline stages."""

import os
import yaml
import torch
from transformers.trainer_utils import get_last_checkpoint

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def find_last_checkpoint(output_dir: str) -> str | None:
    """Return the path to the latest `checkpoint-*` under output_dir, or None.

    Lets a script resume after an interruption (e.g. a Colab disconnect)
    instead of silently retraining from the base checkpoint. Only meaningful
    for Trainer subclasses that support `resume_from_checkpoint` -- TRL's
    experimental PPOTrainer does not, so 03_ppo_train.py can't use this.
    """
    if not os.path.isdir(output_dir):
        return None
    return get_last_checkpoint(output_dir)


def precision_kwargs() -> dict:
    """Pick bf16 vs fp16 based on actual hardware support.

    `bf16=torch.cuda.is_available()` is wrong on its own: T4 (free Colab)
    has CUDA but no bf16 support, and requesting bf16 there produces
    bf16 gradient tensors that crash inside GradScaler (which only
    handles fp16) with "not implemented for 'BFloat16'". Checking
    `torch.cuda.is_bf16_supported()` picks fp16 on T4 and bf16 on
    L4/A10G/A100 automatically.
    """
    if not torch.cuda.is_available():
        return {"bf16": False, "fp16": False}
    bf16_ok = torch.cuda.is_bf16_supported()
    return {"bf16": bf16_ok, "fp16": not bf16_ok}


def load_config(config_path: str | None = None) -> dict:
    """Load configs/config.yaml (or a custom path) and resolve relative paths
    against the project root so scripts work regardless of the caller's cwd.
    """
    if config_path is None:
        config_path = os.path.join(PROJECT_ROOT, "configs", "config.yaml")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    for key in ("sft_data_path", "preference_data_path", "ppo_prompts_path",
                "sft_output_dir", "reward_output_dir", "ppo_output_dir",
                "rm_eval_path", "gen_eval_path", "dpo_output_dir"):
        if key in cfg and not os.path.isabs(cfg[key]):
            cfg[key] = os.path.join(PROJECT_ROOT, cfg[key])

    return cfg


def build_lora_config(cfg: dict):
    """Return a peft LoraConfig from the config file, or None if LoRA is off."""
    if not cfg.get("use_lora"):
        return None
    from peft import LoraConfig, TaskType

    lora_cfg = cfg["lora"]
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
    )
