"""
Stage 2: Reward Model training.

Trains a scalar reward model on (prompt, chosen, rejected) preference pairs,
starting from the SFT checkpoint. Uses trl's RewardTrainer, which expects a
dataset with "chosen" and "rejected" text columns and trains an
AutoModelForSequenceClassification(num_labels=1) head to score full
(prompt + response) sequences.

Run:
    python src/02_reward_model.py
"""

import argparse
import os

import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardConfig, RewardTrainer

from common import build_lora_config, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)

    # Start the reward model from the SFT checkpoint so it shares the
    # policy's tokenizer/vocab and has some task understanding already.
    base_ckpt = cfg["sft_output_dir"]

    tokenizer = AutoTokenizer.from_pretrained(base_ckpt)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(base_ckpt, num_labels=1)
    model.config.pad_token_id = tokenizer.pad_token_id

    dataset = load_dataset("json", data_files=cfg["preference_data_path"], split="train")

    # RewardTrainer scores full sequences; fold prompt into chosen/rejected.
    def format_pairs(example):
        return {
            "chosen": example["prompt"] + "\n" + example["chosen"],
            "rejected": example["prompt"] + "\n" + example["rejected"],
        }

    dataset = dataset.map(format_pairs, remove_columns=["prompt"])

    eval_path = cfg.get("rm_eval_path")
    has_eval = bool(eval_path) and os.path.exists(eval_path)
    eval_dataset = None
    if has_eval:
        eval_dataset = load_dataset("json", data_files=eval_path, split="train")
        eval_dataset = eval_dataset.map(format_pairs, remove_columns=["prompt"])

    reward_cfg = cfg["reward"]
    training_args = RewardConfig(
        output_dir=cfg["reward_output_dir"],
        num_train_epochs=reward_cfg["num_train_epochs"],
        per_device_train_batch_size=reward_cfg["per_device_train_batch_size"],
        learning_rate=reward_cfg["learning_rate"],
        max_length=reward_cfg["max_length"],
        logging_steps=1,
        save_strategy="epoch",
        report_to="none",
        bf16=torch.cuda.is_available(),  # transformers defaults bf16=True, which fails validation on CPU-only machines
        **({"eval_strategy": "epoch", "per_device_eval_batch_size": reward_cfg["per_device_train_batch_size"]} if has_eval else {}),
    )

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=build_lora_config(cfg),
    )

    trainer.train()
    trainer.save_model(cfg["reward_output_dir"])
    tokenizer.save_pretrained(cfg["reward_output_dir"])
    print(f"Reward model saved to {cfg['reward_output_dir']}")

    if has_eval:
        metrics = trainer.evaluate()
        print(f"Eval metrics: {metrics}")
        print(f"eval_accuracy: {metrics.get('eval_accuracy')}")


if __name__ == "__main__":
    main()
