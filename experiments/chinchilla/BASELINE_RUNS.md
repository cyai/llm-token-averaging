# OLM Baseline & Ablation Runs — Summary

All models are OLM transformers trained from scratch on FineWeb (`sample-10BT`),
tokenizer `EleutherAI/pythia-70m` (vocab 50304). `eval_loss` and `train_loss`
are cross-entropy in **nats/token**; the reported value is the final logged row.

**Key setting — token averaging (`k`):** `k=1` is the dense baseline (predict
every token). For `k>1` we mean-pool every `k` consecutive token embeddings
before the transformer, so the transformer sees `seq_len / k` positions and the
model predicts every `k`-th token. Unless a run is in the "context-scaling"
table below, **`seq_len = 1024` (raw, pre-averaging)**, so the transformer
length is `1024 / k`.

> ⚠️ **Comparability caveat:** `k=1` and `k>1` losses are *not* directly
> comparable — a `k=2` model only scores half the token positions. For
> apples-to-apples numbers use the offset-ensemble full-position eval
> (`eval_full_positions.py`, results in `results/eval_full_positions.json`), not
> the raw `eval_loss` below.

---

## 1. Dense (k=1) baselines

| Run | Params | d / L / H | Raw ctx | Tokens | Final train | Final eval | Source CSV |
|-----|-------:|:---------:|:-------:|:------:|:-----------:|:----------:|------------|
| `model1_50m`  | 50.9M  | 512 / 8 / 8   | 1024 | 1.0B  | 4.432 | **4.437** | `model1_50m/loss_log.csv` |
| `model1_125m` | 123.5M | 768 / 12 / 12 | 1024 | 2.5B  | 4.204 | **4.204** | `model1_125m/loss_log.csv` |
| `model1_250m` | 252.8M | 1024 / 16 / 16| 1024 | 4.18B\* | 3.505 | **3.458** | `model1_250m/loss_log.csv` |

\* `model1_250m` target was 5B tokens; latest log stops at 4.18B (still training / stopped early).

---

## 2. Token-averaging runs (the method) — iso-parameter to their baseline

Each averaging run has the **same architecture** as its `k=1` baseline above and
is trained on `k ×` the token budget (so it takes the same number of transformer
forward passes / roughly iso-FLOPs at the endpoint).

| Run | Params | d / L / H | k | Transformer len | Tokens | Final train | Final eval | Source CSV |
|-----|-------:|:---------:|:-:|:---------------:|:------:|:-----------:|:----------:|------------|
| `avg_50m_k2`  | 50.9M  | 512 / 8 / 8   | 2 | 512 | 2.0B   | 4.427 | **4.433** | `avg_50m_k2/loss_log.csv` (= `_untied`) |
| `avg_50m_k4`  | 50.9M  | 512 / 8 / 8   | 4 | 256 | 4.07B  | 4.717 | **4.734** | `avg_50m_k4/loss_log.csv` (= `_untied`) |
| `avg_125m_k2` | 123.5M | 768 / 12 / 12 | 2 | 512 | 5.0B   | 4.178 | **4.171** | `avg_125m_k2/loss_log.csv` |
| `avg_250m_k2` | 252.8M | 1024 / 16 / 16| 2 | 512 | 8.36B\* | 3.460 | **3.438** | `avg_250m_k2/loss_log.csv` |

\* `avg_250m_k2` target was 10B tokens; latest log stops at 8.36B.

---

## 3. Context-scaling variants (50M, `transformer len` held at 1024)

These probe "double the raw context for free" by raising `seq_len` with `k` so
the transformer length stays 1024. Here raw context = `1024 × k` (or 2048 for
the true-2048-ctx dense control). All at 50.9M params, 512 / 8 / 8.

| Run | k | seq_len (raw) | Transformer len | Tokens | Final eval | Source CSV |
|-----|:-:|:-------------:|:---------------:|:------:|:----------:|------------|
| `model1_50m` (1× ctx)      | 1 | 1024 | 1024 | 1.0B | **4.437** | `model1_50m/loss_log_1x_ctx.csv` |
| `avg_50m_k2` (2× ctx)      | 2 | 2048 | 1024 | 2.0B | **4.650** | `avg_50m_k2/loss_log_2x_ctx.csv` |
| `avg_50m_k4` (4× ctx)      | 4 | 4096 | 1024 | 4.07B| **5.174** | `avg_50m_k4/loss_log_4x_ctx.csv` |
| `model2_50m_ctx2n` (dense) | 1 | 2048 | 2048 | 1.0B | **4.650** | `model2_50m_ctx2n/loss_log.csv` |

Takeaway: on FineWeb, extending raw context (whether by averaging or by a true
2048-length dense model) *hurts* loss at this scale — so the gains in Table 2
come from cheaper tokens, not from the extra context.

---

## 4. 50M pooling ablations (all `seq_len = 1024`, Config A)

One thing changed vs plain mean-pooling (`avg_50m_k2` / `avg_50m_k4`); everything
else identical. Baselines to compare against: **k=2 → 4.433**, **k=4 → 4.734**.

| Run | k | Pooling change | Tokens | Final train | Final eval | Δ vs mean | Source CSV |
|-----|:-:|----------------|:------:|:-----------:|:----------:|:---------:|------------|
| `avg_50m_k2_learnable` | 2 | learned combiner       | 2.0B  | 4.407 | **4.383** | −0.050 | `avg_50m_k2_learnable/loss_log.csv` |
| `avg_50m_k2_ov4s2`     | 2 | overlapping (w=4, s=2) | 2.0B  | 4.522 | **4.492** | +0.059 | `avg_50m_k2_ov4s2/loss_log.csv` |
| `avg_50m_k2_word`      | 2 | word-boundary windows (~2.1× realized) | 2.0B | 4.612 | **4.577** | +0.144 | `avg_50m_k2_word/loss_log.csv` |
| `avg_50m_k2_wexp`      | 2 | exponential weights    | 2.0B  | 4.622 | **4.617** | +0.184 | `avg_50m_k2_wexp/loss_log.csv` |
| `avg_50m_k4_learnable` | 4 | learned combiner       | 4.07B | 4.436 | **4.445** | −0.289 | `avg_50m_k4_learnable/loss_log.csv` |
| `avg_50m_k4_wexp`      | 4 | exponential weights    | 4.07B | 4.254 | **4.265** | −0.469 | `avg_50m_k4_wexp/loss_log.csv` |

Takeaway: **learned pooling** helps at both k (most at k=4). At k=2, unequal
fixed weights and word/overlap windows all *hurt* vs plain mean. At k=4,
exponential weighting is the strongest variant.

---

## Notes & caveats

- **Canonical CSV selection.** Several directories contain buggy/superseded logs
  (`_rope_bug`, `_same_ctx`, `_tied`, `_v2`, `loss_log_avg_50m_k2`, etc.). This
  doc uses only the runs above. For the 50M base/averaging trio, `loss_log.csv`
  is byte-identical to the `_untied` variant.
- **Embedding tying.** The 50M base/averaging trio (Tables 1–2) canonical logs
  are **untied**; the 125M/250M configs are **tied**. The pooling ablations
  (Table 4) were run with the current configs (tied by default), so their
  nominal k=2/k=4 baselines carry a minor tie-vs-untie confound — re-run the mean
  baseline tied if you need an exact ablation control.
- **Batch/steps.** The Table 4 ablations used a smaller effective batch (≈122k
  steps for 2B tokens vs ≈15k for `avg_50m_k2`), i.e. more optimizer steps for
  the same token budget. Flag this if step count is a concern for the comparison.
- **250M runs are in progress** (did not reach their target token budgets).
