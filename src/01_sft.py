"""
Stage 1: Supervised Fine-Tuning (SFT).

Fine-tunes the base model on (prompt, response) demonstrations so it has a
sane starting policy before reward modeling / PPO. Uses trl's SFTTrainer,
which natively understands the "prompt" + "completion" dataset format.

Run:
    python src/01_sft.py
    python src/01_sft.py --config configs/config.yaml
"""

import argparse

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from common import build_lora_config, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"])

    # dataset must have "prompt" + "response" columns; SFTTrainer expects
    # "completion" so we rename.
    dataset = load_dataset("json", data_files=cfg["sft_data_path"], split="train")
    dataset = dataset.rename_column("response", "completion")

    sft_cfg = cfg["sft"]
    training_args = SFTConfig(
        output_dir=cfg["sft_output_dir"],
        num_train_epochs=sft_cfg["num_train_epochs"],
        per_device_train_batch_size=sft_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=sft_cfg["gradient_accumulation_steps"],
        learning_rate=sft_cfg["learning_rate"],
        max_length=sft_cfg["max_seq_length"],
        logging_steps=1,
        save_strategy="epoch",
        report_to="none",
        bf16=torch.cuda.is_available(),  # transformers defaults bf16=True, which fails validation on CPU-only machines
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=build_lora_config(cfg),
    )

    trainer.train()
    trainer.save_model(cfg["sft_output_dir"])
    tokenizer.save_pretrained(cfg["sft_output_dir"])
    print(f"SFT model saved to {cfg['sft_output_dir']}")


if __name__ == "__main__":
    main()
