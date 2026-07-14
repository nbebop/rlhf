"""
Stage 3: PPO fine-tuning (the "RL" in RLHF).

Uses the SFT checkpoint as the starting policy, the trained reward model to
score generations, and PPO to nudge the policy towards higher-reward
responses while a KL penalty keeps it close to the SFT reference model.

NOTE: as of current TRL releases, PPOTrainer/PPOConfig live under
`trl.experimental.ppo` (they were moved out of the stable `trl` namespace).
This API is explicitly marked experimental upstream and may change between
TRL versions -- pin your `trl` version in requirements.txt for reproducibility,
and re-check https://huggingface.co/docs/trl/main/en/ppo_trainer before
upgrading.

NOTE: unlike the other three stages, PPOTrainer.train() takes no
resume_from_checkpoint argument and unconditionally resets its step counter,
so an interrupted PPO run cannot be resumed -- a rerun starts over from
sft_output_dir. Save often (ppo.total_episodes) and expect to redo this stage
in one sitting on a free-tier GPU.

Run:
    python src/03_ppo_train.py
"""

import argparse

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from trl.experimental.ppo import PPOConfig, PPOTrainer

from common import build_lora_config, load_config, precision_kwargs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)

    sft_ckpt = cfg["sft_output_dir"]
    reward_ckpt = cfg["reward_output_dir"]

    tokenizer = AutoTokenizer.from_pretrained(sft_ckpt)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for generation during rollouts

    # Policy being optimized.
    policy = AutoModelForCausalLM.from_pretrained(sft_ckpt)
    # Frozen reference policy for the KL penalty (None -> PPOTrainer clones `policy`).
    ref_policy = AutoModelForCausalLM.from_pretrained(sft_ckpt)
    # Frozen reward model trained in stage 2.
    reward_model = AutoModelForSequenceClassification.from_pretrained(reward_ckpt, num_labels=1)
    # Trainable value head, initialized from the SFT checkpoint so it shares
    # the policy's representations at the start of PPO.
    value_model = AutoModelForSequenceClassification.from_pretrained(sft_ckpt, num_labels=1)

    for m in (reward_model, value_model, policy, ref_policy):
        m.config.pad_token_id = tokenizer.pad_token_id

    # PPO rollouts only need the prompt, tokenized to input_ids.
    dataset = load_dataset("json", data_files=cfg["ppo_prompts_path"], split="train")

    def tokenize(example):
        return tokenizer(example["prompt"], truncation=True, max_length=256)

    dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

    ppo_cfg = cfg["ppo"]
    training_args = PPOConfig(
        output_dir=cfg["ppo_output_dir"],
        learning_rate=ppo_cfg["learning_rate"],
        num_ppo_epochs=ppo_cfg["num_ppo_epochs"],
        per_device_train_batch_size=ppo_cfg["per_device_train_batch_size"],
        mini_batch_size=ppo_cfg["mini_batch_size"],
        total_episodes=ppo_cfg["total_episodes"],
        response_length=ppo_cfg["response_length"],
        kl_coef=ppo_cfg["kl_coef"],
        missing_eos_penalty=ppo_cfg["missing_eos_penalty"],
        stop_token="eos",
        report_to="none",
        **precision_kwargs(),
    )

    trainer = PPOTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=dataset,
        peft_config=build_lora_config(cfg),
    )

    trainer.train()

    # trainer.save_model() saves only the LoRA adapter when peft is active,
    # which 05_eval_compare.py can't load with AutoModelForCausalLM.from_pretrained.
    # Merge it into the base weights so the checkpoint is a plain model either way.
    policy = trainer.model.policy if hasattr(trainer.model, "policy") else trainer.model
    if cfg.get("use_lora"):
        policy = policy.merge_and_unload()
    policy.save_pretrained(cfg["ppo_output_dir"])
    tokenizer.save_pretrained(cfg["ppo_output_dir"])
    print(f"PPO-tuned model saved to {cfg['ppo_output_dir']}")


if __name__ == "__main__":
    main()
