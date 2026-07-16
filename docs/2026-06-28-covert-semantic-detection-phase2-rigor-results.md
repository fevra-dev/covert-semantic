# SIEVE · COVERT_SEMANTIC — Phase-2 Rigor Pass: Results (n=150)

**Results memo** · **Date:** 2026-06-28 · **Author:** Mac Calarco (fevra-dev)
**Spec:** `2026-06-28-covert-semantic-detection-phase2-rigor-design.md` · **Plan:** `…-implementation-plan-3.md`
**Corpus:** `covert-semantic/experiments/corpus/phase2_n150_20260628.jsonl` (1,200 Opus-4.8 generations)
**Result matrix:** `covert-semantic/experiments/phase2_rigor_20260628.json`

> **Headline (honest).** The distributional detector is the **best single detector in-distribution** — out-of-fold AUC 0.94–1.00, Holm-significant over the strongest baseline on **all four** constructions. But that advantage **does not uniformly survive leave-one-construction-out generalization**, so the universal claim ("a model-conditional distributional detector detects semantic stego *across constructions*") is **demoted to a per-construction / construction-family result** (design §8). The fusion ensemble's value is as a **robustness floor**, not a universal peak. **Pre-registered verdict cell: HONEST-PARTIAL.**

This is the outcome the pre-registration named as honest-partial, and it is reported as such rather than dressed up.

---

## 0. What changed from the n=20 pilot

The n=20 pilot's in-distribution numbers were in-*sample* (the distributional detector was trained and tested on the same texts). This pass fixes that: **all in-distribution numbers below are 5-fold out-of-fold**, and generalization is measured by **leave-one-construction-out (LOCO)**. The contrast between the two is the real result.

## 1. Corpus & decodability (a finding in itself)

n=150 per construction × {stego, cover} = 1,200 Opus-4.8 generations, 0 missing, 3 synonym refusals (→ empty, handled). **Decodability — whether `decode_text` recovers the planted payload — varies enormously by construction:**

| construction | decode rate | reading |
|---|---|---|
| acrostic | **1.00** | Opus reliably executes this channel |
| synonym | 0.51 | partial compliance |
| narrative_positional | **0.00** | "subtly evoke" → never literal (the GLOSSOPETRAE regime) |
| sentence_length | **0.05** | Opus rarely nails 12 exact word-count parities |

A 0%→100% spread across four channels is a standalone observation about *which covert constructions a frontier model can actually run* — independent of detectability.

## 2. In-distribution (5-fold OOS) — distributional wins, uniformly

AUC / Youden-J per detector, P_ref = Qwen2.5-1.5B-Q8:

| construction | perplexity | cross-ppl | **distributional** | ensemble-max | ensemble-stacked |
|---|---|---|---|---|---|
| narrative | 0.96 / 0.83 | 0.70 / 0.34 | **0.99 / 0.91** | 0.97 / 0.84 | 0.98 / 0.91 |
| acrostic | 0.68 / 0.26 | 0.69 / 0.37 | **0.94 / 0.74** | 0.88 / 0.62 | 0.94 / 0.76 |
| synonym | 0.92 / 0.70 | 0.79 / 0.47 | **0.98 / 0.84** | 0.95 / 0.78 | 0.97 / 0.83 |
| sentence_length | 0.85 / 0.68 | 0.74 / 0.50 | **1.00 / 0.98** | 0.95 / 0.81 | 1.00 / 0.98 |

Distributional beats the strongest baseline on **all 4** constructions, Holm-significant (H1 ✓). Perplexity reproduces its prior-art strength on the *statistical* channel (narrative 0.96) and the *lexical* channel (synonym 0.92) but is weak on the *structural* channel (acrostic 0.68). Cross-perplexity (a real Binoculars (1.5B,3B) pair this time, not the degenerate n=20 column) is the weakest baseline throughout.

## 3. Held-out (leave-one-construction-out) — the advantage is construction-dependent

Train on 3 constructions, evaluate on the unseen 4th:

| held-out construction | perplexity | cross-ppl | distributional | ensemble-max | reading |
|---|---|---|---|---|---|
| narrative | 0.97 / 0.77 | 0.70 / 0.37 | 0.97 / 0.81 | 0.97 / 0.79 | **tie** — perplexity needs no per-construction training |
| acrostic | 0.68 / 0.28 | 0.69 / 0.35 | **0.78 / 0.20** | 0.70 / 0.16 | ranking generalizes (AUC↑, Holm-sig vs cross), **operating-point does not** (J=0.20) |
| synonym | 0.92 / 0.71 | 0.79 / 0.49 | 0.97 / 0.79 | 0.97 / 0.76 | distributional ahead but **just misses Holm** (+0.03, p=0.048) |
| sentence_length | 0.86 / 0.68 | 0.74 / 0.50 | **0.68 / 0.02** | 0.86 / 0.67 | distributional **collapses** (−0.18 vs perplexity); ensemble held by perplexity |

**Only one held-out cell is Holm-significant** (acrostic distributional vs cross-perplexity, +0.09, p=0.0055). So the pre-registered H2 ("distributional beats both baselines, all 4 held-out, Holm") is **not met** → MIXED/NULL verdict, correctly.

The structure of the failure is the finding: the distributional detector's discriminative features (surprisal shape, spectral, ACF, rank) **transfer across lexical/statistical channels but not to a structurally novel channel it never trained on** (sentence_length). Its acrostic AUC=0.78 (vs J=0.05 at n=20) shows the earlier "total collapse" was *partly* n=20 noise — there is real cross-construction ranking signal — but the operating-point threshold chosen on other constructions does not transfer (J=0.20).

## 4. The ensemble is a floor, not a peak (honest)

Max-fusion of perplexity + distributional, held-out:

- **sentence_length:** distributional collapses to 0.68; **ensemble holds at 0.86 / J=0.67** (perplexity carries it). This is the complementarity payoff — fusion prevents the worst collapse.
- **acrostic:** distributional 0.78 alone **beats** the ensemble 0.70 — fusing the weak perplexity signal (0.68) *hurt* the ranking.

So H3 ("ensemble ≥ best component, held-out") holds 3/4 (fails acrostic). Honest statement: **the ensemble trades peak accuracy for a higher worst-case floor** — it never suffers the sentence_length-style collapse, at the cost of not always matching the best single detector. That is a defensible, real, modest contribution — not "the ensemble dominates."

## 5. Size ablation — leans construction-probe, not weak-P_ref

Best held-out AUC (max of distributional, ensemble-max) by reference-model size:

| construction | 0.5B | 1.5B | 3B |
|---|---|---|---|
| narrative | 0.94 | 0.97 | 0.97 |
| acrostic | 0.70 | **0.78** | 0.75 |
| synonym | 0.94 | **0.97** | 0.91 |
| sentence_length | 0.83 | **0.86** | 0.82 |

**1.5B is the sweet spot.** A stronger P_ref (3B) does **not** lift the structural held-out cells — acrostic, synonym, and sentence_length all peak at 1.5B and slightly regress at 3B (H5 not supported). Per the pre-registered verdict table, the acrostic held-out limitation is therefore read as a **genuine construction-probe limitation, not a weak-reference-model artifact** (design §8). (Total Holm-significant cells rise with size — 0.5B 9/16, 1.5B 8/16, 3B 12/16 — i.e. a larger P_ref sharpens *in-distribution* discrimination, but that does not translate into *held-out structural* generalization.)

## 6. Caveats (do not over-read)

1. **sentence_length is confounded.** Its cover is 3 everyday sentences while its stego is 12 short sentences, so part of its in-distribution AUC=1.00 is **surface structure (length), not covert signature**. Combined with 5% decodability, "detecting sentence_length stego" largely means "detecting a conspicuous, mostly-failed encoding attempt." *Fix for next pass: length-matched covers per construction.*
2. **Decodability lens on narrative.** narrative decodes at 0% — its "stego" is Opus subtly theming text; high perplexity AUC there is detecting "stilted-on-purpose" prose, consistent with prior art, not a recoverable payload.
3. **Single corpus / single seed.** n=150 per construction, one Opus sampling pass, one fold seed. CIs are bootstrap BCa on this corpus; a second corpus would tighten the held-out borderline cells (synonym p=0.048).
4. **Not adaptive.** Static detectors vs a static sender. The Carlini adaptive-attacker frontier (Phase 3) is the next claim, not this one.

## 7. Disposition (keep / demote)

- **Keep (honest, defensible):** (a) distributional detector is the strongest *in-distribution* semantic-stego detector across 4 constructions, on honest out-of-fold evaluation; (b) **cross-construction generalization is family-dependent** — it transfers across lexical/statistical channels and fails on a structurally novel one; (c) the **fusion ensemble raises the worst-case floor**; (d) **decodability ≠ detectability** — a 0–100% compliance spread across channels; (e) cross-perplexity/Binoculars is dominated by both perplexity and distributional here.
- **Demote (per §8):** any universal "detects semantic stego across constructions" claim. Hold it to per-construction + the ensemble-floor framing.
- **Perplexity stays a reproduction** of prior art (`2603.21567`), now confirmed at n=150 with out-of-sample calibration.

## 8. Next

1. **Length-matched covers** per construction (kills the sentence_length confound) → re-run analyze (offline, free) on a regenerated corpus.
2. **2nd corpus / multi-seed** to resolve the borderline synonym held-out cell.
3. **Phase 1.5 cross-model auditor** and **Phase 3 adaptive frontier + capacity bound** (Carlini-standard adaptive evaluation) remain the larger novel thrusts.
