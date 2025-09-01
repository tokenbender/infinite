# infinite
a rubric driven prioritized replay rl algo to maximise continual learning

Article: infinite — a rubric driven prioritized replay to maximise continual learning
https://tokenbender.com/post.html?id=infinite-a-rubric-driven-prioritized-replay-to-maximise-continual-learning

this project implements a novel reinforcement learning approach that leverages prioritized replay mechanisms guided by learning rubrics to optimize continual learning performance. the algorithm focuses on:

- **rubric-driven prioritization**: using structured evaluation criteria to determine experience replay priority
- **continual learning optimization**: maximizing knowledge retention and transfer across learning tasks
- **adaptive replay strategies**: dynamically adjusting replay priorities based on learning progress and performance metrics
- **grpo training**: currently has a grpo implementation

## features

### core implementation
  - distributed training across multiple gpus
  - fsdp (fully sharded data parallel) for model parallelism
  - tensor parallelism and sequence parallelism support
  - zigzag ring attention for efficient long-context training
  
### infrastructure  
- **worker architecture**: modular design with separate actor, critic, and rollout workers
- **multi-turn dataset handling for RL rollouts**
- **checkpointing**
- **environment rewards**: currently added few basic custom reward functions for various rl tasks:
  - local search optimization
  - orz problem solving
  - searchr1 retrieval tasks
- **hydra configuration**: yaml-based configuration management for hyperparameters

## installation

```bash
# clone the repository
git clone https://github.com/tokenbender/infinite.git
cd infinite

uv venv --python 3.11
uv pip install packaging wheel
uv pip install torch
uv pip install "flash-attn==2.7.4.post1" --no-build-isolation -v
pip install -r requirements.txt
```

## quick start

### training with grpo

```bash
# single gpu training
python -m train.trainer.grpo --config-name grpo

# multi-gpu training (8 gpus)
torchrun --nproc_per_node=8 -m train.trainer.grpo --config-name grpo

# example launch script with custom grpo config for testing
./launch_grpo.sh
```

### configuration

edit `config/grpo.yaml` to customize training parameters:
- model architecture settings
- parallelism configurations (ddp, tp, sp sizes)
- learning rate and optimizer settings
- dataset paths and batch sizes
- reward function specifications

## project structure

```
infinite/
├── config/          # hydra configuration files
├── env/            # custom reward functions
├── train/          # training implementation
│   ├── datasets/   # dataset handlers
│   ├── trainer/    # training algorithms
│   ├── utils/      # utility functions
│   └── workers/    # distributed workers
├── rubric/         # rubric evaluation system (to be planned for infinite)
└── planner/        # planning and scheduler implementation for prioritized experience replay
```

## technical details

### grpo implementation
the grpo trainer uses reinforcement learning with group-based policy optimization for improved sample efficiency. key features:
- advantage normalization with variance reduction
- kl divergence regularization with k3 reward estimator
- support for multi-turn conversations and function calling

### parallelism support
- **fsdp**: fully sharded data parallel for large model training
- **tensor parallelism**: split model layers across devices
- **sequence parallelism**: distribute sequence computation
- **zigzag ring attention**: efficient attention for long sequences

### data format
training data should be in json format:
```json
[
    {
        "messages": [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "user query"}
        ],
        "answer": "expected response"
    }
]
```

## roadmap and research plan

- High-level plan (live checklist in repo):
  https://github.com/tokenbender/infinite/blob/main/high_level_plan.md


## contributing
you can contribute in various tasks here as listed in the roadmap, or just hop in for brainstorming and discussion - [e/Xperiments discord server](https://discord.gg/YaYfPu4ZT4)
contributions are welcome! please feel free to submit pull requests or open issues for bugs and feature requests.

## license
this is a fully apache 2.0 license oss work.
