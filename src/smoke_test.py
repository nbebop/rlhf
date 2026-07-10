"""
Fast end-to-end sanity check for the pipeline, using a tiny model and 1 step
per stage. Run this after `pip install -r requirements.txt` and before
kicking off a real (slow, GPU) run, to catch wiring bugs cheaply.

Run:
    python src/smoke_test.py
"""

import argparse
import copy
import shutil
import sys

from common import load_config

TINY_MODEL = "hf-internal-testing/tiny-random-gpt2"  # a few MB, CPU-friendly


def make_smoke_config(cfg):
    smoke = copy.deepcopy(cfg)
    smoke["base_model"] = TINY_MODEL
    smoke["sft_output_dir"] = cfg["sft_output_dir"] + "_smoke"
    smoke["reward_output_dir"] = cfg["reward_output_dir"] + "_smoke"
    smoke["ppo_output_dir"] = cfg["ppo_output_dir"] + "_smoke"
    smoke["sft"]["num_train_epochs"] = 1
    smoke["reward"]["num_train_epochs"] = 1
    smoke["ppo"]["total_episodes"] = 8
    smoke["ppo"]["per_device_train_batch_size"] = 2
    smoke["ppo"]["mini_batch_size"] = 1
    smoke["ppo"]["response_length"] = 8
    return smoke


def run(config_path=None):
    cfg = load_config(config_path)
    smoke_cfg = make_smoke_config(cfg)

    print("=== Stage 1/3: SFT (tiny model, 1 epoch) ===")
    import importlib
    sft_mod = importlib.import_module("01_sft")
    _run_stage(sft_mod, smoke_cfg)

    print("=== Stage 2/3: Reward model (tiny model, 1 epoch) ===")
    reward_mod = importlib.import_module("02_reward_model")
    _run_stage(reward_mod, smoke_cfg)

    print("=== Stage 3/3: PPO (8 episodes) ===")
    ppo_mod = importlib.import_module("03_ppo_train")
    _run_stage(ppo_mod, smoke_cfg)

    print("\nSmoke test passed: all three stages ran without errors.")
    for d in (smoke_cfg["sft_output_dir"], smoke_cfg["reward_output_dir"], smoke_cfg["ppo_output_dir"]):
        shutil.rmtree(d, ignore_errors=True)


def _run_stage(module, cfg):
    # Each stage script does `from common import load_config`, which binds
    # its own local name -- patch that local binding, not common's, or the
    # patch is invisible to the already-imported module.
    original = module.load_config
    module.load_config = lambda *_a, **_k: cfg
    try:
        module.main()
    finally:
        module.load_config = original


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    sys.path.insert(0, "")
    run(args.config)
