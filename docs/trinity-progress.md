# Trinity alignment — live progress

Plan: [trinity-alignment.md](./trinity-alignment.md) · Paper: [arXiv:2512.04695](https://arxiv.org/abs/2512.04695)

**Goal:** val pass@1 lift ≥ **+8pp** over `max(single, single_reflect)` with fair 5-turn budget.

**Last updated:** 2026-06-24 — SLM training stopped (gen 3 plateau)

---

## Status

| # | Step | Status | Notes |
|---|------|--------|-------|
| 1 | Complementary active pool (drop duplicate locals) | done | qwen + deepseek + mimo Venice |
| 2 | Union ceiling on full val (23 tasks) | done | **91.3%** union vs **87.0%** best single |
| 3 | Transcript → coordinator `pick()` | done | live loop + SLM + trained features |
| 4 | Benchmark/train progress logging | done | per-task / per-eval prints |
| 5 | Trinity-fair compare (5 turns, `max_turns` enforced) | done | `trinity_loop_config` |
| 6 | `per_question_best` / `gap_to_union` metrics | done | compare + union modes |
| 7 | sep-CMA-ES + larger train defaults | done | `mat-train-live` |
| 8 | Train linear coordinator (10 train tasks) | done | 90% train pass@1; weights updated |
| 9 | Full val compare (trained checkpoint) | done | **0pp** vs DeepSeek single; **8.7pp** below union |

---

| 10 | SLM coordinator train (Qwen3-0.6B head, 20 tasks) | **stopped** | gens 1–3 flat at **0.6952** (~70% train); ~16h for 3 gens |
| 11 | SLM val compare | skipped | killed before checkpoint useful |

Checkpoint (partial): `~/.config/mat/coordinator/latest_slm.json` — do not use for val; training plateaued.

---

## Active run

None.

```bash
# Union ceiling only (fastest diagnostic)
mat-benchmark --split val --mode union --out traces/val_union.json 2>&1 | tee traces/val_union.log

# Full honest compare (after mat-train-live; refuses untrained checkpoint)
mat-benchmark --split val --mode compare \
  --connector deepseek-v4-flash@venice \
  --checkpoint ~/.config/mat/coordinator/latest.json \
  --out traces/val_compare_trained.json 2>&1 | tee traces/val_compare_trained.log

# Coordinator training (train split only)
mat-train-live --tasks 10 --generations 8 --population 8 \
  --checkpoint ~/.config/mat/coordinator/latest.json

# Trinity SLM head (pip install mat[slm]; transcript hidden states)
mat-train-live-slm --tasks 20 --generations 10 --parallel 4 \
  --checkpoint ~/.config/mat/coordinator/latest_slm.json

# Overnight train + compare
bash scripts/overnight_slm.sh
```

Watch progress: `tail -f traces/val_qwen_single.log` (or whichever log is running).

Merge partial connector reports:

```bash
python3.11 scripts/merge_union.py traces/val_qwen_single.json traces/val_deepseek_single.json traces/val_mimo_single.json --out traces/val_union.json
```

---

## Results log

| Run | per_question_best | best single | orchestrated | delta | gap_to_union |
|-----|-------------------|-------------|--------------|-------|--------------|
| Trained compare (val 23, DeepSeek baseline) | 95.7%* | **87.0%** (single) | **87.0%** | **0.0pp** | **8.7pp** |
| 3-connector union (val 23) | — | **87.0%** best single | **91.3%** union | +4.3pp vs single | — |
| SLM train (20 tasks, gen 3) | — | — | **~70%** train pass@1 | — | plateau |

\*Compare `per_question_best` uses single+reflect baselines for this connector only (95.7%), not the 3-connector union (91.3%).

## Lessons (2026-06-24)

1. **Pool has real headroom** — union 91.3% vs best single 87% (+4.3pp). Routing *should* help.
2. **Neither coordinator learned to capture it** — linear 0pp val lift; SLM stuck at ~70% train after 24 evals.
3. **Live CMA-ES is expensive** — ~5h/gen at `parallel=2`; need fast probes before long runs.
4. **Next bets** — joint (model×role) actions, replicates, singular-value SLM tuning, or pool swap before more head training.
