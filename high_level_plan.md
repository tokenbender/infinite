## Infinite — Bitter‑Lesson Single‑Signal Prioritized Replay (Checklist)

Goal: Maximize continual learning via a minimal replay mechanism driven by per‑domain pass‑rate EMA, keeping GRPO core unchanged.

### Issue Tasklist
- [ ] #2 Models, domains, datasets, evals for first iteration (P0)
- [ ] #5 Domain registry + multi-domain dataset/env routing
- [ ] #8 Stubs and randomized rewards to unblock E2E runs
- [ ] #6 Single-signal prioritized replay scheduler (acc_ema)
- [ ] #7 Contamination detector (train/eval overlap audit)

### P0 — Prerequisites ([#2](https://github.com/tokenbender/infinite/issues/2))
- [x] Define domains and coverage: math, code, tools, language (knowledge deferred)
- [x] Create dataset layout per domain (stubs allowed under `stub/data/<domain>/{train,test}.jsonl` with Hydra overrides)
- [x] Add eval adapters under `evals/<domain>/` (stubs in place):
  - [ ] math500
  - [ ] aiderbench
  - [ ] terminalbench
  - [ ] tau2bench
  - [ ] vrcli
  - [ ] eq bench
  - [ ] simpleqa
- [x] Ensure env/verifiers exist or stubbed in `env/` (math/code/language → `env/eq.py`; tools → `env/searchr1.py`)
- [ ] Implement contamination gate CLI and config (Issue #7)
- [x] Choose baseline model (Qwen/Qwen2.5-1.5B-Instruct; target Qwen/Qwen2.5-7B-Instruct) and record sampling params

#### Baseline (iteration 1)
- Model: `Qwen/Qwen2.5-1.5B-Instruct` (smoke); target `Qwen/Qwen2.5-7B-Instruct`
- Sampling: train `temperature=1.0`, test `temperature=0.0`; `max_new_tokens=128` (tools: 256); `responses_per_prompt=8`; `prompts_per_rollout=64`
- GRPO: `adv.estimator=reinforce`, `adv.norm_var=true`, `actor.kl.reward_estimator=k3`

### P0 — Domain Registry & Routing ([#5](https://github.com/tokenbender/infinite/issues/5))
- [ ] Add config `data.domains: [{id, train_path, test_path, env_path, rubric?}]`
- [ ] Implement `MultiDomainRLDataset` yielding `{domain_id, messages, answer}` with per‑domain iterators
- [ ] Add env registry in rollout; select env by `domain_id`
- [ ] W&B: log `domain_id` distribution

### P0 — Stubs & Randomized Rewards ([#8](https://github.com/tokenbender/infinite/issues/8))
- [ ] Add `rollout.mode: real|stub`; bypass LLM in stub mode with placeholder responses
- [ ] Implement `DomainStubEnv` (Bernoulli `p_domain` rewards) and `RubricPassEvaluatorStub` (1–4 categorical; pass if ≥3)
- [ ] Config `stubs: { enable, seed, domains: [{id, p_base}], rubric: {p_scores: [...]}}`
- [ ] W&B: log `stub_mode`, per‑domain `p_base`, realized pass‑rate

### P1 — Single‑Signal Scheduler ([#6](https://github.com/tokenbender/infinite/issues/6))
- [ ] `acc_ema` per domain; init with `init_acc_ema` or baseline pass‑rate
- [ ] Sampling `p_i = softmax(τ · (1 − acc_ema_i))` with ε floor (applied at sampling time)
- [ ] Update rule `acc ← (1−β)·acc + β·pass` (pass=env bool; later rubric ≥3)
- [ ] Persist/restore `acc_ema` via checkpointing
- [ ] Toggle `trainer.prioritized_replay: true|false`
- [ ] W&B: log `acc_ema/<domain>`, `p/<domain>`, `pass_rate/<domain>`

### P1→P2 — Toy Run
- [ ] Prepare small multi‑domain slices; run GRPO with scheduler enabled
- [ ] Compare prioritized vs uniform sampling baselines

### P2 — Upgrade Mode
- [ ] Add a new domain while preserving prior skills; fix KL to anchor behavior (≥0.1)
- [ ] Safety gates: warn/increase KL/pause on prior‑domain drops (θ1/θ2/θ3)

### Evaluation & Success Criteria
- [ ] Compute BWT, FWT, ACC, AURC, and per‑domain `acc_ema` curves
- [ ] Achieve ≥25% AURC gain vs uniform under equal compute
- [ ] Upgrade runs: new‑domain +5 pts; prior‑domain ≤1 pt drop
- [ ] Contamination detection ≥95% on seeded duplicates

### Hyperparameters (initial)
- [ ] β=0.05, τ=3.0, ε=1e‑3, `init_acc_ema`=0.5 (or baseline), `kl_coefficient`=0.1 (upgrade mode)

### Unblocking Order & Ownership
- [ ] #2 — P0: models + domains/datasets/evals selection — Assignee: @me — https://github.com/tokenbender/infinite/issues/2
- [ ] #5 — P0 (Blocks #6): domain registry + routing — Assignee: @me — https://github.com/tokenbender/infinite/issues/5
- [ ] #8 — P0 (optional dep on #5): stubs + randomized rewards — Assignee: @me — https://github.com/tokenbender/infinite/issues/8
- [ ] #6 — P1 (Blocked by #5): scheduler + `acc_ema` persistence — https://github.com/tokenbender/infinite/issues/6
- [ ] #7 — P0 (Independent): contamination detector gate — https://github.com/tokenbender/infinite/issues/7
