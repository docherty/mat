# Trinity alignment — live progress

Plan: [trinity-alignment.md](./trinity-alignment.md) · Paper: [arXiv:2512.04695](https://arxiv.org/abs/2512.04695)

**Goal:** val pass@1 lift ≥ **+8pp** over `max(single, single_reflect)` with fair 5-turn budget.

**Last updated:** 2026-06-23 — val compare with trained checkpoint **running** (`traces/val_compare_trained.log`)

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
| 9 | Full val compare (trained checkpoint) | **running** | DeepSeek baseline + orchestrated, 23 val tasks |

---

## Active run

```bash
tail -f traces/val_compare_trained.log
```

Output: `traces/val_compare_trained.json` (when complete).

```bash
# Union ceiling only (fastest diagnostic)
mat-benchmark --split val --mode union --out traces/val_union.json 2>&1 | tee traces/val_union.log

# Full honest compare (after mat-train-live; refuses untrained checkpoint)
mat-benchmark --split val --mode compare \
  --connector deepseek-v4-flash@venice \
  --checkpoint ~/.config/mat/coordinator/latest.json \
  --out traces/val_compare_trained.json 2>&1 | tee traces/val_compare_trained.log

# Coordinator training (hours; prints each eval + generation)
mat-train-live --tasks 10 --generations 8 --population 8 \
  --checkpoint ~/.config/mat/coordinator/latest.json \
  2>&1 | tee traces/train_live.log
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
| Union (3 connectors, val 23) | **91.3%** | **87.0%** (DeepSeek) | — | — | — |
| Qwen / DeepSeek / MiMo singles | — | 82.6% / 87.0% / 78.3% | — | — | — |

Fill this table as benchmarks complete.
