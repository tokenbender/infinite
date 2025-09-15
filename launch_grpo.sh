#!/bin/bash
# GRPO Training Launch Script
# Reference: RL2/examples/orz_ppo.sh adapted for GRPO configuration

# Setup Hugging Face environment to handle rate limiting
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME}"

# Load environment variables if .env file exists
if [ -f "env/.env" ]; then
    echo "Loading environment variables from env/.env"
    source env/.env
fi

# Check for HF token
if [ -z "$HF_TOKEN" ] && [ -z "$HUGGING_FACE_HUB_TOKEN" ]; then
    echo "WARNING: No Hugging Face token found!"
    echo "To avoid rate limiting, please set HF_TOKEN in env/.env or export it"
    echo "Get your token from: https://huggingface.co/settings/tokens"
    echo ""
    echo "Continuing without authentication (may hit rate limits)..."
    sleep 3
else
    echo "Using Hugging Face token (${HF_TOKEN:0:8}...)"
fi

# Optional: Enable offline mode if models are already cached
# export TRANSFORMERS_OFFLINE=1
# export HF_DATASETS_OFFLINE=1

echo "Starting GRPO training with retry logic for model downloads..."
echo ""

## Example: full run (commented out; replace with real data/model before use)
# torchrun \
#     --nproc_per_node=8 \
#     -m train.trainer.grpo \
#     data.train_data_path=Chenmien/OpenReasonerZero \
#     data.test_data_path=Chenmien/OlympiadBench \
#     data.prompts_per_rollout=128 \
#     data.responses_per_prompt=32 \
#     actor.model_name=Qwen/Qwen2.5-7B \
#     actor.sp_size=4 \
#     actor.max_length_per_device=8192 \
#     actor.freeze_steps=4 \
#     actor.kl.reward_estimator=k3 \
#     rollout.train_sampling_params.max_new_tokens=4096 \
#     rollout.env_path=env/orz.py \
#     rollout.apply_chat_template=false \
#     adv.estimator=reinforce \
#     adv.norm_var=true \
#     trainer.project=GRPO \
#     trainer.experiment_name=grpo-minimal \
#     trainer.test_freq=8 \
#     trainer.save_freq=32

torchrun \
    --nproc_per_node=1 \
    -m train.trainer.grpo \
    data.train_data_path=stub/data/math/train.jsonl \
    data.test_data_path=stub/data/math/test.jsonl \
    data.prompts_per_rollout=4 \
    data.responses_per_prompt=2 \
    actor.model_name=Qwen/Qwen2.5-1.5B-Instruct \
    actor.max_length_per_device=512 \
    rollout.train_sampling_params.max_new_tokens=128 \
    rollout.env_path=env/eq.py \
    adv.estimator=reinforce \
    adv.norm_var=true \
    trainer.project=GRPO \
    trainer.experiment_name=grpo-math-stub \
    trainer.test_freq=4 \
    trainer.save_freq=8

# GRPO-specific configuration:
# - adv.estimator=reinforce (Dr. GRPO default)
# - adv.norm_var=true (variance normalization)
# - actor.kl.reward_estimator=k3 (GRPO KL estimator)
#
# Replace placeholder values before running:
# - data.train_data_path: Replace with path to your JSON/JSONL training data
# - data.test_data_path: Replace with path to your JSON/JSONL test data  
# - actor.model_name: Replace with Hugging Face model name (e.g., Qwen/Qwen2.5-7B)
# - rollout.env_path: Replace with path to your environment file if needed
