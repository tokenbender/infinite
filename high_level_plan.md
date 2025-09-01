## Infinite — Bitter‑Lesson Single‑Signal Prioritized Replay (Checklist)

Goal: Maximize continual learning via a minimal replay mechanism driven by per‑domain pass‑rate EMA, keeping GRPO core unchanged.

### P0 — Prerequisites
- [ ] Define domains and coverage: math, code, tools, language, knowledge
- [ ] Create dataset layout per domain under `data/<domain>/{train,test}.jsonl`
- [ ] Add eval adapters under `evals/<domain>/` (stubs allowed):
  - [ ] math500
  - [ ] aiderbench
  - [ ] terminalbench
  - [ ] tau2bench
  - [ ] vrcli
  - [ ] eq bench
  - [ ] simpleqa
- [ ] Ensure env/verifiers exist or stubbed in `env/` (use `env/orz.py`, `env/searchr1.py`, extend as needed)
- [ ] Implement contamination gate CLI and config (Issue #7)
- [ ] Choose baseline model (qwen3‑4b or gemma3‑270m) and run baseline eval sweep (optional seeding for `init_acc_ema`)

### P0 — Domain Registry & Routing (Issue #5)
- [ ] Add config `data.domains: [{id, train_path, test_path, env_path, rubric?}]`
- [ ] Implement `MultiDomainRLDataset` yielding `{domain_id, messages, answer}` with per‑domain iterators
- [ ] Add env registry in rollout; select env by `domain_id`
- [ ] W&B: log `domain_id` distribution

### P0 — Stubs & Randomized Rewards (Issue #8)
- [ ] Add `rollout.mode: real|stub`; bypass LLM in stub mode with placeholder responses
- [ ] Implement `DomainStubEnv` (Bernoulli `p_domain` rewards) and `RubricPassEvaluatorStub` (1–4 categorical; pass if ≥3)
- [ ] Config `stubs: { enable, seed, domains: [{id, p_base}], rubric: {p_scores: [...]}}`
- [ ] W&B: log `stub_mode`, per‑domain `p_base`, realized pass‑rate

### P1 — Single‑Signal Scheduler (Issue #6)
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
- [ ] P0 — Issue #5 (Blocks #6): domain registry + routing — Assignee: @me — https://github.com/tokenbender/infinite/issues/5
- [ ] P0 — Issue #8 (optional dep on #5): stubs + randomized rewards — Assignee: @me — https://github.com/tokenbender/infinite/issues/8
- [ ] P1 — Issue #6 (Blocked by #5): scheduler + `acc_ema` persistence — https://github.com/tokenbender/infinite/issues/6
- [ ] P0 — Issue #7 (Independent): contamination detector gate — https://github.com/tokenbender/infinite/issues/7

