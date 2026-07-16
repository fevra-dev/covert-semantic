# SIEVE · COVERT_SEMANTIC — Phase-2 Rigor Pass: Distributional Detector + Fusion Ensemble at n=150

**Design Document (spec)**
**Date:** 2026-06-28 · **Author:** Mac Calarco (fevra-dev)
**Status:** Design — approved (brainstorm 2026-06-28), pending implementation plan-3
**Parent design:** `2026-06-26-covert-semantic-detection-design.md` (RQ2, §4 Phase 2)
**Supersedes for RQ2 evaluation:** the in-sample protocol used in the n=20 live pilot (`experiments/phase2_live_n20.json`)

> **One-line thesis of this pass.** Turn the n=20 directional pilot into an RQ2 *verdict* by (1) replacing in-sample calibration with out-of-sample calibration + leave-one-construction-out generalization, (2) adding a 4th construction and a fusion ensemble, (3) adding a reference-model size ablation, and (4) applying pre-registered multiple-comparison-corrected statistics — all on a corpus generated **once** and analyzed **offline and re-runnably**.

---

## 0. Why this pass exists (honest reading of the n=20 pilot)

Re-reading the pilot *code* (not just its JSON) surfaced three facts that bound what the pilot can claim:

1. **The cross-perplexity baseline was degenerate.** No 2nd GGUF was supplied, so `ref_b = ref_a` → the surprisal ratio is identically 1.0 → z ≡ 0 → the detector predicts "no stego" for every item (J = 0.00, tp = fp = 0 in every cell). The pilot's cross-perplexity column is a **non-result**, not a finding. A real Binoculars-lineage baseline *requires* a second, distinct reference model.

2. **Every in-distribution number is in-sample.** `PerplexityAnomalyDetector.calibrate(neg)` then scores those same `neg` as negatives (in-sample FPR). Worse, `DistributionalDetector.fit(all_neg, all_pos)` trains on the pooled corpus and is then evaluated on subsets of *that same corpus* — so the headline distributional wins (acrostic J = 0.55, synonym J = 0.79) are **training accuracy**, not generalization.

3. **The only honest signal in the pilot is the held-out block, and it is weak.** Leave-one-construction-out: narrative J = 0.35, **acrostic J = 0.05**, synonym J = 0.58. The acrostic collapse is the central open question this pass must adjudicate.

**Consequence for the headline.** We cannot yet claim the distributional detector — or an ensemble — "dominates." We can only claim it *might*, once compared against honestly-calibrated baselines on held-out constructions. This pass builds exactly that comparison.

---

## 1. Research question (restated for the verdict)

**RQ2.** At n = 150, with out-of-sample calibration: does the model-conditional **distributional** detector, and/or a **fusion ensemble**, achieve higher detection (AUC / Youden J) at no worse FPR than the **perplexity-anomaly** and **cross-perplexity** baselines — **and does any advantage survive leave-one-construction-out generalization** — across **four** constructions, after Holm-Bonferroni correction?

**RQ2-ablation (sub-finding).** Does a stronger reference model P_ref lift the held-out collapse (a "weak-reference-model" explanation — itself a publishable finding) or does it persist across model sizes (a genuine construction-probe limitation, design §8 demotion)?

---

## 2. Pre-registered hypotheses & verdicts

Registered **before** the live run; the analysis stage reports against these verbatim.

- **H1 (in-distribution).** Distributional J > best-baseline J at ≤ its FPR, per construction, on out-of-fold predictions.
- **H2 (generalization).** Distributional J > best-baseline J at ≤ its FPR, per held-out construction (the load-bearing test).
- **H3 (fusion).** Ensemble-max J ≥ max(perplexity, distributional) J on held-out constructions (does fusing complementary detectors beat either alone on *novel* constructions?).
- **H4 (learned fusion).** Ensemble-stacked J ≥ Ensemble-max J (does a learned combiner add anything over the fixed rule?).
- **H5 (ablation).** Held-out J on the structural constructions (acrostic, sentence-length) is monotonically non-decreasing in P_ref size.

**Verdict table (committed in advance):**

| Outcome | Reading |
|---|---|
| H1 ∧ H2 hold post-Holm across all 4 constructions | **Strong RQ2 success** — distributional/ensemble is a general detector |
| H1 holds, H2 fails (held-out collapse) | **Honest-partial** — demote to "construction probe" (parent design §8) |
| H2 holds only after H5 (collapse lifts with larger P_ref) | **Reframe** — detectability is reference-model-capacity-bound; report the curve |
| H3 holds where H1/H2 are mixed per-construction | **Complementarity confirmed** — fusion is the deliverable, singles are components |

No outcome is a failure of the *study*; each maps to an honest claim. This is the point of pre-registration.

---

## 3. Architecture — generate once, analyze offline

The only irreversible cost is the frontier API. The design splits the pipeline so that cost is paid exactly once and every analysis variant is free and reproducible.

### 3.1 Stage A — `generate` (live, once, resumable)

`ApiSender` (OpenRouter, `anthropic/claude-opus-4-8`) produces the full corpus:

```
4 constructions × 150 × {stego, cover} = 1,200 generations
```

Written to a **versioned, append-only, resumable** JSONL at `experiments/corpus/phase2_n150_<YYYYMMDD>.jsonl`. One record per generation:

```json
{"id": "...", "construction": "acrostic", "label": "stego|cover",
 "payload_code": 1234, "payload_fields": {...}, "prompt": [...],
 "response": "...", "model": "anthropic/claude-opus-4-8",
 "max_tokens": 256, "temperature": 0.7, "ts": "..."}
```

**Resumability is load-bearing** at 1,200 calls: each record is flushed as produced; a re-run skips `id`s already present. A mid-run crash or rate-limit never loses spend (the n=20 run hit two live-only bugs — this makes recovery free).

**Decodability is recorded, not assumed.** For each stego record we attempt `decode_text` and store `decoded_ok: bool` + recovered fields. Compliance/decodability per construction is itself a reported observable (especially for sentence-length, the hardest to comply with).

### 3.2 Stage B — `analyze` (offline, re-runnable, deterministic)

Loads the cached corpus and sweeps **detectors × reference models**, emitting the full results matrix to `experiments/phase2_rigor_<YYYYMMDD>.json`. Fixed seeds throughout. No network. Re-runnable as the GGUF set grows (e.g., adding the deferred Q4 quantization axis later costs zero API).

---

## 4. Constructions (4)

Same `TUPLE3_SCHEMA` payload across all four — **fairness requires identical payload** so detection difficulty reflects the *construction*, not the bit budget. Cardinality 6×100×6 = 3600 ≈ **11.81 bits**.

| Construction | Family | Encoding | Footprint |
|---|---|---|---|
| `narrative_positional` | statistical | color/number/shape across 3 sentences | 3 sentences |
| `acrostic` | structural | 3-letter base-26 code = sentence initials | 3 sentences |
| `synonym` | lexical | 3 base-26 synonym-table words | 2–3 sentences |
| `sentence_length` **(new)** | structural | **1 bit/sentence via word-count parity** | **12 sentences** |

**Sentence-length sizing (refinement A).** 11.81 bits at 1 bit/sentence ⇒ 12 sentences (2¹² = 4096 ≥ 3600). The integer `code ∈ [0, 3600)` is written big-endian as 12 parity bits (even word-count = 0, odd = 1). Decode reads the word-count parity of the first 12 sentences → 12-bit int; if ≥ 3600 the record is `decoded_ok=false`. `max_tokens` is raised to **384** for this construction only (12 short sentences ≈ 70–100 tokens, but give headroom for the harder instruction-following task). This makes sentence-length the longest/most-awkward construction by design — and its lower expected compliance rate is a finding, not a bug.

---

## 5. Detectors (5 arms)

All implement the existing `score(text) -> Verdict` contract; `Verdict.z` carries the **continuous** score (refinement B — required for ROC/AUC/threshold selection).

1. **`perplexity_anomaly`** — baseline / reproduction of prior art (`2603.21567`). Score = mean-surprisal z vs cover baseline. ROC ranks by signed z (one-sided "stilted" direction).
2. **`cross_perplexity`** — Binoculars-lineage baseline, **now real** (pair = 1.5B × 3B). Score = surprisal-ratio z. ROC ranks by |z| (two-sided).
3. **`distributional`** — logistic discriminator over hand-engineered features (`csd/features.py`: shape, spectral, ACF, rank). Score = p(stego).
4. **`ensemble_max` (primary)** — fuse perplexity + distributional by **max of calibrated-normalized scores**, single fused threshold chosen out-of-sample. The honest "union of complementary detectors" claim with no extra trainable parameters.
5. **`ensemble_stacked` (secondary)** — logistic over `[ppl_z, xppl_z, dist_p]`, trained on **cross-fitted out-of-fold** base scores (no leakage). Reported only as "does a learned combiner beat the fixed rule (H4)?"

**Score normalization for fusion.** Each base score is mapped to [0,1] using train-fold cover calibration: perplexity z → Φ(z) (cover-fit mean/std); distributional p is already [0,1]. `ensemble_max` score = max(Φ(ppl_z), dist_p). One ROC, one OOS-chosen threshold → a proper ROC rather than FPR-inflating logical OR.

---

## 6. Evaluation protocol (the #1 honesty fix)

The new `csd/evaluation.py` collects continuous scores + status per item and does all calibration/threshold-selection **out-of-sample**. The existing `harness.run_detection` (binary, fixed-threshold) is untouched and still serves Phase 1.

**In-distribution (per construction).** Stratified **5-fold CV** within each construction. Per fold:
- baselines: estimate cover mean/std **and** select the operating threshold (max-J) on train folds;
- distributional / stacker: fit on train folds (stacker on cross-fitted base scores);
- score the held-out fold only.
- J / AUC / FPR computed on pooled out-of-fold predictions.

**Generalization (leave-one-construction-out).** Train on 3 constructions, evaluate on the unseen 4th. Baselines calibrated on the 3 training constructions' covers; threshold chosen on training constructions; applied to the held-out construction. **This is the crux metric (H2/H3).**

**Operating point.** Threshold = max-Youden-J on the training partition, never hardcoded (replaces the pilot's fixed `z = 1.5`).

---

## 7. Statistics (pre-registered, multiple-comparison-corrected)

Additions to `csd/stats.py` (alongside existing `youden_j_ci`, `mcnemar`, `bonferroni_threshold`):

1. **`auc(scores, labels)`** — primary threshold-free comparison metric; removes threshold-choice noise.
2. **`youden_j`** at the OOS threshold — reported alongside AUC for comparability to GLOSSOPETRAE's J = 0.58.
3. **`paired_bootstrap_delta(scores_a, scores_b, labels, metric)`** — paired (same items) bootstrap of ΔJ and ΔAUC → **BCa CIs**. DeLong as an AUC cross-check.
4. **`holm_bonferroni(pvalues, alpha=0.05)`** — step-down FWER control (uniformly more powerful than plain Bonferroni) over the **pre-registered family**:
   - {distributional, ensemble_max} vs best-baseline × 4 constructions × {in-dist, held-out} = **16 comparisons**.
   - **`best-baseline` is defined per cell as the higher-AUC of {perplexity, cross_perplexity} selected on the *training* partition only** (never on the test fold/held-out construction) — comparing against the stronger baseline is the conservative choice, and training-partition selection prevents post-hoc cherry-picking. H4 (stacked vs max) and H5 (ablation monotonicity) are **secondary/exploratory** and sit outside this confirmatory family.
5. **`mcnemar`** (existing) at the operating point — secondary paired-binary agreement.

Single-detector J keeps Clopper-Pearson propagation (matches parent reporting); *differences* use the paired bootstrap (the CP propagation is neither paired nor difference-aware).

---

## 8. Reference-model size ablation

Qwen2.5 family (same arch/tokenizer ⇒ clean ablation), Q8: **{0.5B, 1.5B (have), 3B}**.

- **Primary P_ref = 1.5B** (the established baseline) for the headline comparisons.
- **0.5B / 3B** sweep the P_ref axis for perplexity + distributional + ensembles.
- **Cross-perplexity pair fixed at (1.5B, 3B)** — reported once, not swept (it is intrinsically a pair).
- Adjudicates **H5** and the acrostic-collapse question.
- Quantization axis (Q4) **deferred** — re-scores the cached corpus later at zero API cost.

**Risk noted:** `LlamaCppReferenceModel` uses `logits_all=True`; the 3B-Q8 over short (~256-tok) texts is fine on the Mac, but flagged for memory watch.

---

## 9. Build plan (TDD, single-thread per ADR-0009)

Logic-critical (detectors/stats/eval) ⇒ single-threaded in one context, not fanned out. All new code: TDD (mock-first), `ruff` + `bandit` clean, matching the existing 55-test-green discipline. Detailed task breakdown → implementation **plan-3**.

1. `SENTENCE_LENGTH` construction (enum + `encode_text`/`decode_text` parity codec) + `build_stego_messages` branch + per-construction `max_tokens` + tests.
2. `csd/corpus.py` — versioned/resumable JSONL load/save, dedup-by-id, decodability recording + tests.
3. `csd/evaluation.py` — continuous-score collection, 5-fold CV, OOS threshold selection, leave-one-construction-out, cross-fitted stacker scores + tests.
4. `csd/detectors/ensemble.py` — `MaxFusionEnsemble` + `StackedEnsemble` + tests.
5. `csd/stats.py` extensions — `auc`, `paired_bootstrap_delta` (BCa), `holm_bonferroni` + tests.
6. `experiments/phase2_rigor.py` — two-stage `generate` / `analyze` driver; emits the full matrix + the pre-registered verdict report.
7. Download Qwen2.5-0.5B-Q8 + Qwen2.5-3B-Q8.

## 10. Sequencing & gates (before any spend)

1. Build + **mock E2E** the entire pipeline (no API; ADR-0010 dogfood).
2. Download the two GGUFs.
3. **Tiny live smoke** — n = 2 (~16 calls) — to catch live-only bugs cheaply (the n=20 run hit two).
4. **Confirm exact Opus pricing + explicit operator go-ahead.**
5. Full n = 150 `generate` (cached, resumable).
6. Offline `analyze` sweep → results matrix → pre-registered verdict → honest write-up + PICKUP update.

## 11. Out of scope (this pass)

Cross-model auditor (Phase 1.5), the Phase-1 out-of-sample recalibration of the RQ1 perplexity cell, Plan-3/Phase-3 (adaptive attacker + frontier + capacity bound), and the Q4 quantization axis. All remain queued in PICKUP.

---

## Appendix — assumptions verified against source (2026-06-28 double-check)

- All three detectors expose continuous `Verdict.z` → AUC/ROC feasible. ✓ (`csd/detectors/*`)
- `LlamaCppReferenceModel(path)` re-instantiation = size ablation mechanism. ✓ (`csd/surprisal.py`)
- `ApiSender._chat` already guards null content → "" (refusal/filter). ✓ (`csd/senders.py:122`)
- `TUPLE3_SCHEMA` cardinality 3600 ⇒ 11.81 bits ⇒ 12 parity sentences. ✓ (`csd/payloads.py`)
- `stats.py` lacks AUC / bootstrap-Δ / Holm — additive, no collision. ✓ (`csd/stats.py`)
