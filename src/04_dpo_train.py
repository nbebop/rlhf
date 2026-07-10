"""
Stage 4 (alternative to Stages 2+3): Direct Preference Optimization baseline.

DPO trains directly on (prompt, chosen, rejected) preference pairs, starting
from the SFT checkpoint. It skips the separate reward model and the PPO RL
loop entirely -- there is no reward model to train and no rollout/generation
step. Instead, trl's DPOTrainer builds a frozen reference copy of the SFT
model internally and optimizes the policy so that chosen responses become
more likely than rejected responses relative to that reference, via a closed-
form loss on log-probability ratios.

Run:
    python src/04_dpo_train.py --config configs/ckan_config.yaml
"""

import argparse

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from common import build_lora_config, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)

    # Start DPO from the SFT checkpoint, same as the reward model / PPO stages.
    base_ckpt = cfg["sft_output_dir"]

    tokenizer = AutoTokenizer.from_pretrained(base_ckpt)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_ckpt)

    # Same preference data the reward model trains on. DPOTrainer natively
    # accepts the {"prompt", "chosen", "rejected"} explicit-prompt format, so
    # no remapping is needed here (unlike the reward model's format_pairs).
    dataset = load_dataset("json", data_files=cfg["preference_data_path"], split="train")

    dpo_cfg = cfg["dpo"]
    training_args = DPOConfig(
        output_dir=cfg["dpo_output_dir"],
        num_train_epochs=dpo_cfg["num_train_epochs"],
        per_device_train_batch_size=dpo_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=dpo_cfg["gradient_accumulation_steps"],
        learning_rate=dpo_cfg["learning_rate"],
        beta=dpo_cfg["beta"],
        max_length=dpo_cfg["max_length"],
        logging_steps=1,
        save_strategy="epoch",
        report_to="none",
        bf16=torch.cuda.is_available(),  # transformers defaults bf16=True, which fails validation on CPU-only machines
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # DPOTrainer builds its own frozen reference copy; must stay None when peft is active
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=build_lora_config(cfg),
    )

    trainer.train()
    trainer.save_model(cfg["dpo_output_dir"])
    tokenizer.save_pretrained(cfg["dpo_output_dir"])
    print(f"DPO model saved to {cfg['dpo_output_dir']}")


if __name__ == "__main__":
    main()
