#!/bin/bash
# GRPO Training Launch Script
# Reference: RL2/examples/orz_ppo.sh adapted for GRPO configuration

torchrun \
    --nproc_per_node=4 \
    -m train.trainer.grpo \
    data.train_data_path=null \
    data.test_data_path=null \
    data.prompts_per_rollout=128 \
    data.responses_per_prompt=64 \
    actor.model_name=null \
    actor.sp_size=2 \
    actor.max_length_per_device=8192 \
    actor.freeze_steps=4 \
    actor.kl.reward_estimator=k3 \
    rollout.train_sampling_params.max_new_tokens=8192 \
    rollout.env_path=null \
    adv.estimator=reinforce \
    adv.norm_var=true \
    trainer.project=GRPO \
    trainer.experiment_name=grpo-minimal \
    trainer.test_freq=8 \
    trainer.save_freq=32

# GRPO-specific configuration:
# - adv.estimator=reinforce (Dr. GRPO default)
# - adv.norm_var=true (variance normalization)
# - actor.kl.reward_estimator=k3 (GRPO KL estimator)
#
# Fill in null values before running:
# - data.train_data_path: Path to your JSON/JSONL training data
# - data.test_data_path: Path to your JSON/JSONL test data  
# - actor.model_name: Hugging Face model name (e.g., Qwen/Qwen2.5-7B)
# - rollout.env_path: Path to your environment file if needed