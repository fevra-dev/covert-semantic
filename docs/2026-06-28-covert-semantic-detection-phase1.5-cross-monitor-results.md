# SIEVE · COVERT_SEMANTIC — Phase 1.5: Cross-Model Naive Auditor — Results

**Results memo** · **Date:** 2026-06-28 · **Author:** Mac Calarco (fevra-dev)
**Spec:** `2026-06-28-covert-semantic-detection-phase1.5-cross-monitor-design.md`
**Corpus:** `experiments/corpus/phase2_n150_20260628.jsonl` (reused, 1,200 Opus gens; 3 empty stego skipped → 1,197)
**Judgments:** `experiments/monitor_cache_20260628.jsonl` (4,788, cached) · **Matrix:** `experiments/phase1_5_20260628.json`

> **Headline (honest, updated for the 5-model panel).** Opus self-monitoring reproduces the GLOSSOPETRAE null at n=150 across all four constructions (**J≈0**, including the 100%-decodable acrostic). Detection **clusters in the OpenAI lineage**: **both** `gpt-5.5` (pooled J=0.53) **and** `gpt-5.4` (J=0.19) materially detect at **0% FPR**, with a steep within-lineage version gradient (5.5 ≈ 2.8× the J of 5.4); meanwhile the generator's own family (`opus` 0.00 self, `sonnet` 0.03) **and** an independent cross-vendor model (`deepseek-v3.2` 0.00) are all **blind**. This **refutes self-attribution** (DeepSeek didn't generate the corpus yet is blind) and **"any cross-vendor helps"** (DeepSeek detects nothing), and reveals a **lineage/capability-correlated** skill — with the honest caveat that a *single* detecting lineage cannot separate "OpenAI lineage" from "OpenAI models are simply more capable at this task."

> **Update note (panel widened to 6, 2026-06-28).** Claim tightening with data: (1) 4 models, one OpenAI → "model-specific"; (2) +`gpt-5.4` (both OpenAI detect) → "lineage-clustered"; (3) +`x-ai/grok-4.20` (a *capable* independent lineage) → **blind**, which weakens the lineage-vs-capability confound: a comparable-tier non-OpenAI model fails while a *weaker* OpenAI model (5.4) succeeds, so detection tracks **lineage, not raw capability**. Kept as a record of the tightening.

---

## 1. Design (recap)

Same **naive YES/NO monitor** as the GLOSSOPETRAE null condition (`MONITOR_SYSTEM`, text-only, no key, no construction hint), run over the Opus-generated corpus with a 4-model panel. Reasoning was **disabled** uniformly (`reasoning:{enabled:false}` for non-Anthropic; Anthropic naive by default) so every monitor makes a *snap* judgment — the fair naive condition. `gemini-3.1-pro` was dropped (forces reasoning). Metrics: J + Clopper-Pearson CI, FPR, paired McNemar vs Opus-self, Holm over 15 comparisons.

## 2. The matrix (Youden J = TPR_stego − FPR_cover; FPR was 0.00 for every cell)

| monitor | lineage | narrative | acrostic | synonym | sent-len | **pooled J** |
|---|---|---|---|---|---|---|
| `claude-opus-4.8` | **self** (Anthropic) | 0.00 | 0.00 | 0.00 | 0.01 | **0.00** |
| `claude-sonnet-4.6` | same-vendor (Anthropic) | 0.04 | 0.02 | 0.00 | 0.04 | **0.03** |
| **`openai/gpt-5.5`** | OpenAI | **0.79** | **0.48** | **0.55** | **0.32** | **0.53** |
| **`openai/gpt-5.4`** | OpenAI | **0.25** | **0.11** | **0.13** | **0.26** | **0.19** |
| `deepseek/deepseek-v3.2` | DeepSeek | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** |
| `x-ai/grok-4.20` | xAI (capable) | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** |

Detection is confined to the OpenAI rows (both Holm-significant vs self on all 4 constructions + pooled). **Four models across three other lineages** — including capability-comparable ones (`opus`, `grok-4.20`) — are all at chance. Within OpenAI, J scales steeply with version (5.5 = 0.53 vs 5.4 = 0.19); **across** lineages capability does *not* predict — the weaker OpenAI (5.4, J=0.19) detects while the strong xAI flagship (grok-4.20, J=0.00) is blind.

**Every monitor has FPR = 0.00** — none false-positives on cover. So J here is driven entirely by true-positive rate: GPT-5.5's 0.53 is *genuine detection*, not trigger-happiness.

## 3. Findings

**F1 — the self-null replicates at scale.** Opus flags essentially 0% of its own stego across all constructions, **including acrostic** — a *literally recoverable* payload (100% decodable) that the Phase-2 surprisal detector caught at AUC 0.94. Pure self-monitoring blind spot, n=150×4, 0 empties on 3 of 4 constructions.

**F2 — detection clusters in the OpenAI lineage, and scales with version.** Both OpenAI models are Holm-significant over self on **all 4 constructions + pooled** (gpt-5.5 pooled McNemar: 316 caught / 1 reverse, p≈1e-69; gpt-5.4: 111 / 2, p≈3e-24). But strength differs sharply by version: **gpt-5.5 pooled J=0.53 vs gpt-5.4 J=0.19** (≈2.8×). Within a fixed lineage, detection is a capability that scales steeply with model generation — the cleanest, least-confounded signal in the study.

**F3 — same-vendor is practically null, not "self-attribution".** Sonnet's pooled J=0.03 is Holm-significant vs self (15 discordant / 1,197) but **practically at chance**. The auto-verdict initially over-read this as "same-vendor beats self → self-attribution"; corrected to require a *material* J (≥0.15), Sonnet is classed blind. (Honesty fix committed.)

**F4 — cross-vendor is no guarantee, at any capability.** `deepseek-v3.2` (independent lab) *and* `grok-4.20` (xAI flagship, a capable independent lineage) are **both exactly as blind as Opus-self** (J=0.00, caught 0). So "use a cross-vendor auditor" is false as stated — the two non-OpenAI lineages fail across the capability range (weak DeepSeek, strong Grok).

**⇒ What survives: detection is OpenAI-lineage-associated, and capability does not explain it across lineages.** Both OpenAI models detect (version-scaled); four models across three other lineages — spanning weak (DeepSeek) to strong (opus, grok-4.20) capability — are all blind. The **cross-lineage dissociation** is the key evidence: a *weaker* OpenAI model (gpt-5.4, J=0.19) detects while a *comparable-or-stronger* non-OpenAI model (grok-4.20, J=0.00) does not. So it is **not** self-vs-other, **not** "any cross-vendor," and **not** raw capability — it tracks the OpenAI lineage. *Residual question:* why OpenAI (post-training exposure vs a genuine task skill)? — unresolved; needs a **second** detecting lineage or mechanistic analysis, not another blind model.

## 4. Interpretation through the decodability lens (Phase-2 carry-over)

- **acrostic (100% decode):** GPT catches 48% of a fully-recoverable hidden payload that self catches 0% — the cleanest cross-model result.
- **narrative (0% decode):** GPT J=0.79, but narrative barely encodes (Opus "subtly evokes"); GPT is likely detecting the *stilted "trying-to-hide" surface* (the same signal perplexity uses), not a payload. Still, self catches 0%.
- **sentence_length (5% decode, length confound):** GPT's *lowest* J (0.32) despite the 12-vs-3-sentence structural giveaway — notable.

## 5. Honest defensive takeaway

1. **Self-monitoring is reliably blind** to semantic stego — reconfirmed at scale, across constructions, including recoverable payloads. Necessary-but-not-sufficiently-bad.
2. **A cross-vendor auditor is *not* a sufficient fix** — a different-vendor frontier model (DeepSeek) was equally blind.
3. **Actionable rule:** monitor capability for semantic stego is **empirically per-model, clusters by lineage, and scales with version.** You cannot infer it from "not self" (DeepSeek is blind) — but you *can* prefer the **strongest model from a lineage empirically shown to detect** (both OpenAI models here), and you **must re-validate on version bumps** (5.4→5.5 nearly tripled J).
4. **Roger self-attribution hook, honestly engaged:** the pure self-attribution story is *complicated* — Opus is blind to its own output, but so is DeepSeek to Opus's, while both OpenAI models catch it. The defensible result is lineage-clustered, capability-graded detection, not self-attribution.

## 6. Limitations

- **Lineage-vs-capability: substantially addressed, not fully closed.** Adding a *capable* independent lineage (grok-4.20, blind) refutes the "non-OpenAI models are just weaker" alternative — a comparable-tier non-OpenAI fails while a weaker OpenAI succeeds. What remains open is *why* OpenAI: a genuine per-task skill vs post-training exposure to steganography/security data. Only ONE lineage detects, so "OpenAI-lineage-associated" is the honest ceiling; a **second detecting lineage** (or mechanistic analysis) is needed to call it capability-general or to explain the mechanism.
- **Naive / reasoning-off only.** The informed-monitor and reasoning-*on* factorials are deferred — both free follow-ups via the judgment cache. The OpenAI edge is a *snap-judgment* capability (reasoning was off), which is the interesting part.
- **Decodability confounds** carry over (narrative 0%, sentence_length length mismatch) — GPT's narrative 0.79 is partly stilted-ness detection, not payload.
- **Single corpus / seed.**

## 7. Next

1. **Done — capable non-OpenAI (`grok-4.20`) added → blind** ⇒ lineage-over-capability. Open: *why* OpenAI.
2. **A second detecting lineage?** Try another OpenAI-adjacent or reasoning-*on* setting to see if any non-OpenAI model can be coaxed to detect — this is what would separate "genuine task skill" from "OpenAI post-training artifact."
3. **Informed + reasoning-on factorials** (replicate GLOSSOPETRAE's key-possession finding cross-model) — free via cache. Note: reasoning was *off* for all monitors here, so the OpenAI edge is a **snap-judgment** capability; whether reasoning-on lets others catch up is the sharp follow-up.
4. Practical takeaway: *self-monitoring fails (known); cross-vendor is not a heuristic fix (Grok, DeepSeek blind); detection is OpenAI-lineage-associated and version-scaled — monitor selection needs per-model empirical validation and re-validation on upgrades.*

---

# PART 2 — Informed-monitor factorial (task-spec condition, 2026-07-11)

Same corpus, same 6-model panel, **same snap YES/NO**, but the system prompt now *names the payload type and the four constructions* (task-spec) **without the key** — GLOSSOPETRAE's "informed monitor" condition. (Forced-snap wording keeps it a one-word verdict, not analysis; separate cache `monitor_cache_informed_20260628.jsonl`.)

> **Headline.** Task-spec **largely closes the naive lineage gap and lifts self-monitoring out of the null.** Every model improves (Δ +0.15 to +0.72); the naively-blind models recover strongly and *cleanly* (**DeepSeek 0.00→0.72, Opus 0.00→0.44, Sonnet 0.03→0.49, all at ~0% FPR**). So the naive OpenAI-lineage advantage was **mostly "OpenAI models check *spontaneously*; the others just needed to be told."** This **contradicts GLOSSOPETRAE's "task-spec barely helps (95% is key-possession)"** — here task-spec is the difference between blind and strong. **Two honest caveats keep it from being a clean universal:** (a) **Grok's gain is trigger-happiness (FPR 0.39)**, not detection; (b) **structural constructions stay hard even when informed**, except model-specific competencies.

## Naive → Informed (pooled J)

| monitor | lineage | naive J | **informed J** | Δ | informed FPR | unparsed |
|---|---|---|---|---|---|---|
| opus | self | 0.00 | 0.44 | +0.44 | 0.00 | 51 |
| sonnet | Anthropic | 0.03 | 0.49 | +0.47 | 0.00 | 28 |
| gpt-5.5 | OpenAI | 0.53 | **0.77** | +0.24 | 0.01 | 0 |
| gpt-5.4 | OpenAI | 0.19 | 0.34 | +0.15 | 0.00 | 0 |
| deepseek | DeepSeek | 0.00 | **0.72** | +0.72 | 0.06 | 0 |
| grok | xAI | 0.00 | 0.54* | +0.54 | **0.39** | 0 |

\* Grok's informed J is inflated by false alarms — see G3.

## Informed J per construction

| monitor | narrative | acrostic | synonym | sent-len |
|---|---|---|---|---|
| opus | 0.93 | 0.07 | 0.61 | 0.03 |
| sonnet | 0.90 | 0.18 | 0.79 | 0.02 |
| gpt-5.5 | 0.99 | **0.75** | 0.95 | 0.40 |
| gpt-5.4 | 0.92 | 0.05 | 0.35 | 0.02 |
| deepseek | 0.91 | 0.45 | 0.60 | **0.91** |
| grok | 0.59 | 0.57 | 0.65 | 0.37 |

## Findings

**G1 — Task-spec closes the lineage gap.** The naive story (only OpenAI detects) does *not* survive informing: DeepSeek informed (0.72) is second only to gpt-5.5, above both OpenAI-4.x and every Anthropic model. So the naive gap was a *prompting/behavioral* difference (spontaneous checking), not a capability ceiling. Holm: 19/25 contrasts significant.

**G2 — Self-monitoring is *not* fundamentally blind.** Opus self-monitoring goes 0.00→0.44 (narrative 0.93, synonym 0.61) at **0% FPR** — told what to look for, a model catches its *own* stego. The naive J≈0 self-null is a naive-prompting artifact, not an inability. (The verdict engine flags this automatically: `NON-NULL SELF` in the informed run.)

**G3 — Not clean-universal: Grok became trigger-happy (FPR 0.39).** Informing pushed Grok to say YES to ~everything (TPR≈0.98, but 39% of *covers* flagged too). Its 0.54 is alarm, not detection. The false-alarm rate is **uniform across all four constructions** (FPR 0.33–0.43), i.e. *global* paranoia, not a construction-specific artifact — so it can't be repaired by a per-construction fix, only by asking Grok for a graded confidence instead of a flat YES/NO. Lesson: informing can *miscalibrate* a monitor rather than sharpen it — calibration (FPR) must be checked, not just TPR/J.

**G4 — Structural constructions resist task-spec, except model-specific skill.** acrostic and sentence_length stay ~0 for most models even when told exactly what to look for — mechanically checking first-letters or word-count parity is beyond a snap judgment. The exceptions are striking and model-specific: **gpt-5.5 nails acrostic (0.75)** and **deepseek nails sentence_length (0.91)** — genuine structural competencies the others lack. gpt-5.5 stays strongest overall (0.77) and improves *with* informing, so it is not *only* spontaneous checking.

**G5 — Contradicts the anchor paper's factorial.** GLOSSOPETRAE: task-spec buys ~5%, key-possession is 95%. Cross-model here, task-spec buys **0.44–0.72 J** on the semantic constructions (narrative, synonym) — an order of magnitude more than "barely helps." The divergence is worth flagging in the write-up (different models, different constructions, snap-judgment condition).

## Caveats

- **Anthropic unparsed is confined to sentence_length** (Opus 48/51, Sonnet 26/28 of all unparsed are the 12-sentence sentence_length texts — the informed prompt tips them into analysis that truncates at max_tokens=16). So only those two cells (0.03, 0.02) are on reduced parsed sets — treat as unreliable; narrative/acrostic/synonym are clean (1–2 unparsed each). A higher token budget would let them finish, but that turns the snap judgment into analysis (the reasoning-on condition we rejected) — so the honest fix is to leave it flagged, not to spend calls chasing a cleaner number on the weakest construction.
- **Grok FPR (G3)** — its informed row is not comparable to the others' clean detection.
- Single corpus / snap condition / one model per non-OpenAI lineage.

## Disposition

The Phase-1.5 story is now two-condition and honestly reframed: **naive** → self-monitoring blind, detection OpenAI-lineage-*associated*; **informed** → the gap is mostly *spontaneous checking* (task-spec recovers detection across lineages, self included, at low FPR), **except** (a) it can miscalibrate (Grok) and (b) structural constructions need model-specific skill (gpt-5.5 acrostic, deepseek sentence_length). Defensive takeaway sharpens: *a naive monitor is the wrong instrument; an **informed** monitor recovers most semantic-stego detection — but you must validate its FPR per model, and structural channels still need a specifically-capable model.*
