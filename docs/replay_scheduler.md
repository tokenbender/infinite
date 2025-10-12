# Initial Replay Scheduler Design

Context: tracks the planning deliverable for [[Planning] Plan of initial replay scheduler](https://github.com/tokenbender/infinite/issues/10). The goal is to land a minimal prioritized replay loop for GRPO that can prioritize prompts or domains using a single control signal, while keeping the path open for richer strategies later.

## Objectives
- Provide a block-diagram view of the scheduler so we agree on the key components before implementation.
- Leave the existing GRPO trainer largely untouched; the scheduler integrates through narrow seams (dataloader hooks, rollout telemetry, simple configuration flags).
- Keep the first iteration simple: one scorer signal (e.g., pass-rate EMA) that influences which prompts get replayed in the next batch.
- Persist enough scheduler state to resume runs and compare prioritized vs uniform sampling without losing history.

## Existing Signals To Reuse
- `Rollout.__call__` already returns per-sample rewards and logs metrics (`train/workers/rollout.py`).
- GRPO stores rollouts per prompt across `responses_per_prompt`; dynamic filtering (same file) already demonstrates prompt-level grouping.
- Checkpointing utilities (`train/utils/checkpointing.py`) are able to persist arbitrary trainer state dictionaries.
- WandB logging is centralised, so scheduler stats can be reported alongside existing metrics.

## System Overview

```
Prompts (dataset JSONL / verifiers stream)
    ↓
Prompt Scorer (pluggable strategy -> priority scores)
    ↓
Prompts With Scores (priority queue or ranked list)
    ↓ top_k / sample
Selected Prompts
    ↓
Inference (existing rollout worker)
    ↓
Reasoning Traces (messages, rewards, metadata)
    ↓
Reward Model / Signal Collector (reuse env rewards or extra heuristics)
    ↓
Reasoning Traces With Rewards
    ├─→ Scheduler State Update (EMA, counters, cooldowns) ──┐
    └─→ Update Model Weights (current GRPO path)           │
                                                           │
Reflection / Library (future work: storing traces, prompt generation)
                                                           │
                          └────────────────────────────────┘
```

Black arrows mark the minimal loop for v0; red arrows in the original sketch correspond to future extensions (reflection libraries, learned memory, prompt generation).

## Core Components

### 1. Prompt Inventory
- Source: existing datasets (`verifiers:` URIs, JSONL files). No format changes.
- Representation: `PromptRecord` with `id`, `messages`, `answer`, optional metadata (`domain`, `source`, `include=True`).
- Storage: in-memory list for now; future versions can stream from disk if needed.

### 2. Scheduler Registry
- Location: new module under `train/scheduler/`.
- Interface: `Scheduler.step(observations: list[PromptOutcome]) -> SchedulerBatch`.
  - `PromptOutcome`: prompt id, reward, pass bool, timestamp, any auxiliary metrics.
  - `SchedulerBatch`: list of prompt ids selected for the next training batch, plus debug info.
- Strategy selection handled via Hydra config (`scheduler.strategy: pass_rate_ema`, `scheduler.enabled: true|false`).

### 3. Prompt Scorer
- Minimal scorer uses pass-rate EMA per prompt or per domain.
- API: `score(prompt_id) -> float`.
- Updates EMA with `score = beta * outcome + (1 - beta) * prev_score`.
- Sampling rule: lower EMA -> higher priority (`softmax(tau * (1 - ema))` with epsilon floor).

### 4. Sampler / Replay Buffer
- Maintains priority queue keyed by prompt id.
- For the first version, no deep buffer: the dataset itself is the buffer.
- When `SchedulerBatch` requests `top_k`, select prompts with highest priority that are not currently “cooling down”.
- `cooldown` flag prevents immediate requeue if the run already produced high reward (e.g., all responses correct).

### 5. Signal Collector
- Sits between rollout and scheduler.
- Collates per-prompt observations: reward mean, std, pass flag, dynamic filtering ratio.
- Hooks:
  - Extend `Rollout.__call__` to emit prompt-level summaries.
  - Extend `Trainer.train` to pass summaries into `Scheduler.step`.

### 6. State Persistence
- Checkpoint contents: EMA table, cooldown timers, random seed, config hash.
- Use `save_ckpt` to stash scheduler state as an extra entry (`scheduler_state.pt`).

### 7. Telemetry
- WandB keys: `scheduler/ema/<prompt_id or domain>`, `scheduler/priority/<prompt_id>`, `scheduler/pass_rate`.
- Local logs via `tqdm.write` for debugging when scheduler is enabled.

## Integration Points

1. **Configuration**
   - Add `scheduler` section to `config/grpo.yaml` (enabled flag, strategy name, hyperparams `beta`, `tau`, `epsilon`, `cooldown_steps`).
2. **Dataset Loader**
   - Wrap `RLDataset.collate_fn` so that, when scheduler is on, it asks the scheduler for prompt ids to load instead of purely random selection.
3. **Trainer Loop**
   - After each rollout batch, aggregate outcomes and call `scheduler.update`.
   - Before each batch, fetch prompt ids via `scheduler.sample(batch_size)`.
4. **Checkpointing**
   - Inject scheduler state into `save_ckpt` and restore during `load_ckpt`.

## Minimal Implementation Plan

1. **Scaffolding**
   - Create `train/scheduler/__init__.py`, `pass_rate.py`, `state.py`.
   - Define data classes: `PromptOutcome`, `SchedulerState`, `SamplerConfig`.
2. **Configuration Wiring**
   - Update `config/grpo.yaml` with scheduler section.
   - Update Hydra config to instantiate scheduler in `GRPOTrainer.__init__`.
3. **Signal Collection**
   - Modify `Rollout.__call__` to group metrics by prompt id and return a `List[PromptOutcome]` when `train=True`.
   - Adjust `Train.train` to receive both tensor dicts and outcomes.
4. **Sampling**
   - Add scheduler call before dataloader fetch; fallback to uniform when disabled.
   - Temporary approach: wrap existing dataloader so it reads prompts by id.
5. **State Persistence**
   - Register scheduler state with `save_ckpt` / `load_ckpt`.
6. **Telemetry**
   - Log EMA and priority stats at rank 0.
   - Add command-line flag to dump scheduler summary every N steps.
7. **Validation**
   - Run `launch_grpo.sh` once with scheduler off (baseline) and once on (with deterministic random seed) to ensure identical behavior when EMA starts uniform.
   - Add unit test for scheduler strategy (if test harness exists).

## Future Directions (from diagram notes)
- **Reflections:** store successful reasoning traces in a library; feed back into prompts as extra context or generative seed. Out of scope for v0.
- **Adaptive Prompt Generation:** use stored reflections to synthesize new prompts or adjust datasets (requires separate pipeline).
- **Reward-Aware Rescheduling:** integrate more complex heuristics (worst-of-N, variance thresholds) once EMA pipeline proves stable.
- **Learned Memory Components:** experiment with LoRA or cartridge modules informed by reflections.

## Open Questions
- Should EMA track per prompt, per domain, or both? Proposal: v0 uses per-domain EMA to avoid sparse updates.
- How to handle new prompts arriving mid-run (dynamic datasets)? For now, assume static prompt list; dynamic ingestion could treat new prompts as cold-start with default EMA.
- What is the right fallback when scheduler cannot fill a batch (e.g., too many prompts cooling down)? Option: revert to uniform sampling for the remainder of the batch.

## Next Actions
1. Review and approve this design outline.
2. Create `train/scheduler` scaffolding and config wiring.
3. Instrument rollout and trainer to emit prompt-level outcomes.
4. Implement the EMA-based scorer and integrate with dataloader sampling.
5. Validate via smoke tests, then iterate on strategy plugins.

The above steps deliver the minimal replay scheduler described in Issue #10 while keeping the door open to the richer strategy and reflection loops illustrated in the diagram.
