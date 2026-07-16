# SIEVE · COVERT_SEMANTIC — The Black-Box Detectability Frontier of LLM Semantic Steganography

**Design Document**
**Date:** 2026-06-26 · **Author:** Mac Calarco (fevra-dev)
**Status:** Design — approved for implementation planning
**Framework:** SIEVE (renames the defense framework previously specced as *AISec-Guard*) · **Benchmark:** AgentSec-Bench · **Attack engine:** PromptBreaker
**Output of:** brainstorming session 2026-06-26 (multilingual prompt-injection track / GLOSSOPETRAE threat analysis)

> **Naming note.** *SIEVE* is the new umbrella name for the defense framework formerly written as *AISec-Guard* across `2026-03-31-acg-spec.md`, `2026-03-31-aisec-guard-*.md`, etc. A rename pass across those existing specs is **tracked separately and out of scope for this document** — this doc uses SIEVE throughout. *INJEST* is reserved as the name of SIEVE's safe corpus-ingestion / sanitizing-loader module (addresses the "ingestion *is* the attack surface" meta-finding); it is referenced in §7 but not specced here.

---

## 0. Provenance & framing (defensive)

The anchor for this work is **GLOSSOPETRAE — *Lingua Ex Machina*** (elder-plinius, June 2026), a practitioner research program run to academic standards (McNemar, Bonferroni, Clopper-Pearson CIs, n=30→1800, 78 ground-truth result JSONs). Provenance tier **D approaching C** — rigorous method, high adversarial transparency (a fabricated "GPT=23 blind spots" claim was self-retracted with an audit trail), but **no peer review and no independent replication**. We treat all GLOSSOPETRAE corpora and claims as **data, never instructions**, and treat load-bearing numbers as Tier-D until our own Phase-1 reproduction confirms them. The engagement posture is **defensive**: we analyze the open practitioner attack surface to build by-design defenses and honest capability statements — the same posture as Anthropic's Frontier Red Team and Constitutional Classifiers work.

The load-bearing insight we build on: **frontier models follow *structure*, not language identity** (zero-shot acquisition of procedurally-generated languages, sharp 0%→~100% cliff at K=8). This kills the "enumerate dangerous languages" framing of multilingual-injection defense and replaces it with **structural-encoding detection** — larger, more fundamental, more publishable. Semantic steganography is the hardest residue of that shift: readable natural-language text hiding a structured payload, which Pliny shows evades every frontier monitor at chance (J≈0).

---

## 1. Problem & contribution

GLOSSOPETRAE establishes a **static null**: naive frontier monitors detect semantic stego at **J≈0** (n=150, CI ±0.024); three classical detectors are refuted at **J=0** (multi-sample consistency, paraphrase round-trip, stylometric); the *only* working Layer-2 signal is **perplexity-anomaly at J=0.58** (TPR=83%/FPR=25%) — and crucially via *LLM "fluency ratings"* (K=3 subjective vibes vs an n=17 cover baseline), not a probability measurement. Every flagged trial was **"anomalously stilted"**: encoding a payload makes text measurably *less* fluent. The factorial result is the load-bearing one: **~95% of any informed-monitor advantage is key-possession, not task-spec** — telling a monitor *what to look for* barely moves it; only handing it the key helps. Therefore *"build a better judge prompt" is a dead end.*

What is missing — and what we contribute — is the **adaptive** picture:

> **Headline claim.** Black-box detectability of LLM semantic steganography is **information-theoretically bounded by payload capacity**. We (a) build a *model-conditional distributional* detector benchmarked against the recently-proposed perplexity proxy (arXiv:2603.21567, "Quadroculars"/Binoculars lineage) at materially lower FPR, generalizing across constructions; then (b) trace the **detectability frontier**: best-achievable Youden J as a function of *payload bits × text length × adaptive-attacker fluency budget*; and (c) prove the corollary that matters for agent safety — **the lowest-bandwidth channels (a 1-bit `proceed/abort` coordination signal) sit below any black-box detector's noise floor.**

This reframes the defender's conclusion from *"monitor for stego"* → *"monitoring has a capacity floor; the dangerous low-bit regime must be defended **architecturally** (capability isolation), not by a monitor."* It also directly answers the objection Pliny logged as **unmeasured**: *"11 bits isn't practical — but agentic harm may need only 1 bit."* Our result measures exactly that.

**Why it matters:** it (1) reproduces the prior-art result, clearing that rigor bar before extending it; (2) lands a novel, honest, falsifiable result (the frontier + the floor + a capacity bound); and (3) is **monitoring-relevant** — it bears directly on cross-vendor monitoring, the self-monitoring family-trait blindness, and the empty-reply monitor-DoS class.

---

## 2. Threat model (formal)

Agents **A → B** exchange a payload through a channel observed by monitor **M**.

- **M is black-box, text-only.** It sees the output text only — **no generating-model logits, no decode key.** This is the honest adversarial-sender case: a colluding A+B will not hand the monitor logprobs or a key. (The white-box / logit-access case is relevant only to *self-monitoring your own agent*; we include it only as an **oracle upper bound**, §4 Phase 3.)
- **A is adaptive (Carlini standard).** A knows M's detector and optimizes the stego construction to **minimize detectability subject to B decoding the payload**. A static detector against a static attacker is worthless; the contribution is the detector-vs-adaptive-attacker frontier.
- **Channel.** Natural-language-only semantic encoding (synonym/word choice, sentence length, acrostic/positional, metaphor). **Physical Unicode carriers are out of scope** — normalization deterministically solves those (SIEVE Layer-1; GLOSSOPETRAE §5.1).
- **Decision & metric.** Binary "is anything hidden?" with **clean negative controls**; primary metric **Youden J = TPR − FPR**, chosen to be directly comparable to GLOSSOPETRAE. Recovery (exact-payload decode) is reported only as a lower bound on detectability (per GLOSSOPETRAE §4.0).

---

## 3. Research questions & hypotheses

The prior-art hypotheses **H1–H5** are inherited from the queued lit spec (`Research Prompt - SEMANTIC STEGANOGRAPHY DETECTION IN LLM OUTPUTS.md`) and run as the **Phase-0 novelty gate**. The experiment-bearing questions:

| RQ | Question | Pass condition |
|----|----------|----------------|
| **RQ1** (reproduce) | Does our instrument reproduce GLOSSOPETRAE's J≈0 null and J=0.58 perplexity baseline? | Both land within CI of his published cells. Positive control. |
| **RQ2** (detector) | Can a model-conditional *distributional* detector beat J=0.58 at lower FPR, **and generalize across constructions** (not overfit acrostics)? | J significantly > 0.58 with FPR < 25%, holding across ≥3 constructions, Bonferroni-corrected. |
| **RQ3** (frontier) | Under an adaptive attacker, what is best-detector J as a function of (bits, N tokens, fluency budget)? Where is the undetectable floor? | A reproducible frontier surface with an identified floor δ_min. |
| **RQ4** (capacity bound) | Is the empirical frontier consistent with a KL/Pinsker capacity prediction (per-token "extra surprisal" cost of B bits)? | Empirical floor matches the predicted δ ≈ B/N scaling within the reference-model-mismatch budget. |

---

## 4. Method — four phases

### Phase 0 — Prior-art map (novelty gate)
Run the queued `SEMANTIC STEGANOGRAPHY DETECTION` research prompt → **prior-art matrix** + ACH resolution (H1–H5). **Kill switch:** if **H1 confirms** (a published black-box detector already beats J=0.58 on natural-language-only, uninformed, n≥20), pivot the headline from *"first detector"* to the *frontier/capacity* result (still novel) and cite the prior detector as a baseline. **[RESOLVED 2026-06-26]** Phase-0 first pass + 4 L3 reads: H1 *not* confirmed (no uninformed J>0.58 prior art), but the perplexity proxy *is* prior art (`2603.21567`) → **C4 demoted to reproduction; C2 (adaptive frontier) + C3 (per-token-KL bound + 1-bit floor) are the intact contributions.** See `covert-semantic/PRIOR_ART.md`; grounding cites Cachin 1998 (KL/TV) + Carlini `2510.09023` (adaptive standard). Trace forward citations of **Ziegler et al. 2019** (Neural Linguistic Steganography) and **Ueoka et al. 2021** — GLOSSOPETRAE §5.3's named refuted baselines — to verify they say what Pliny claims (L3 verification on the anchor).

### Phase 1 — Reproduce + harden (positive control)
1. Re-implement Pliny's constructions: **11.8-bit {color, number, shape}** acrostic/synonym/sentence-length/positional/metaphor; **17.4-bit 5-tuple** expansion; **3-sentence narrative-positional** constraint.
2. Re-implement his perplexity-anomaly detector to confirm the **J=0.58** cell and the **J≈0 null**.
3. **Harden the instrument:** replace the LLM "fluency-rating" rater with a **real reference-LM token-surprisal measurement.** s_t = −log P_ref(x_t | x_<t), aggregated and thresholded against a per-genre cover baseline.
   - **Reference model = open-weight, sender-independent, with full token-logprob access.** Anthropic exposes no logprobs and most frontier APIs expose little; **local open-weight is therefore the *required* instrument, not a fallback.** Target hardware: Pi 5 (8 GB) covers the **1–3B prototyping tier**; a **7–14B tier** runs on a workstation GPU or a logprob-exposing host (Together/Fireworks) for headline runs.
   - **Reference-model-size ablation (publishable sub-finding):** measure detector J as a function of reference-model size/quality. A stronger P_ref lowers the detection floor δ_min (you cannot detect deviations smaller than your model error). This quantifies how much detector power is bought by reference-model quality.
4. **Success = our null and baseline land within CI of his.** If not, the instrument is wrong — stop and fix before extending (falsification §8).

### Phase 2 — Model-conditional distributional detector (the contribution)
Go beyond *mean* surprisal. Score the full **token-selection profile** under P_ref:
- **Surprisal distribution shape** — variance / kurtosis (localized encoding spikes the mean washes out).
- **Rank statistics** — fraction-argmax vs systematic biased lower-rank choices at encoding slots (the watermark-detection insight without the key: to carry bits you must bias selection; the bias is visible in rank distribution even when mean fluency is normalized).
- **Positional / cadence structure** — autocorrelation of surprisal, sentence-boundary parity, acrostic-position checks for the constructions that *are* structurally detectable.
- **Discriminator** — feed the hand-engineered feature vector to a small learned classifier trained on (cover, stego) pairs we generate. Hand-engineered features (not end-to-end) so the claim is about a *universal constraint signature*, not a memorized construction.

**Robustness crux (why this survives the attack that kills Phase 1):** fluency-rejection-sampling can normalize *mean* surprisal, but keeping the payload **decodable** forces residual bias at encoding slots — so variance / rank / positional features persist. That irreducible tension *is* the capacity link Phase 3 formalizes.

### Phase 3 — Adaptive attack + frontier + capacity bound (the headline)
1. **Adaptive attacker** (Carlini standard): (a) **fluency rejection-sampling** — generate many payload-carrying candidates, keep the one minimizing mean(s_t); (b) **bit-spreading** — dilute per-token encoding cost across more tokens; (c) **construction search** over the library.
2. **Frontier sweep:** payload bits ∈ {1, 4, 8, 11.8, 17.4} × text length N × attacker budget → plot **best-detector J** across the grid → the **frontier surface** and the **undetectable floor δ_min**.
3. **Capacity bound (RQ4):** formalize and test the prediction that embedding B bits requires aggregate divergence ∑_t KL(Q_t‖P_t) ≳ B, so the *per-token* signal is δ ≈ B/N, while a black-box detector's power scales ∝ N·δ². Channels with **B/N < δ_min are undetectable.** Report δ_min as **detector-relative** (it depends on P_ref mismatch + cover-baseline variance + N) and also report the **P_ref = P oracle floor** as the absolute information-theoretic bound.
4. **Corollary (the result):** B=1 bit over N≈200 tokens → δ ≈ 0.005 bits/token ≪ δ_min → **the 1-bit agent-coordination channel is below the floor.** Defense implication: low-bit coordination must be denied **architecturally**, not by monitoring.

---

## 5. Metrics & statistics

Match GLOSSOPETRAE's bar exactly so the result is reviewer-proof:
- **Youden J = TPR − FPR** with **Clopper–Pearson** CIs; **McNemar** on paired receiver/monitor outcomes; **Bonferroni** across cells.
- **Report the silent-failure trichotomy** (confident-"no" / **empty-reply** / unparseable) as distinct classes — **never collapse empties into "clean."** (Opus-4.8 returned 149/150 empties on tag-chars; a naive harness scores that as a clean pass. We alarm on it.)
- **Calibration study (H2):** sensitivity of the detector to cover-baseline size, reference-model family/size, and K. The published binding constraint is **FPR=25%**; the testable prediction is that better calibration lowers FPR without collapsing TPR.

---

## 6. Materials & model panel (design-time; execution deferred)

- **Constructions:** GLOSSOPETRAE's (for direct comparability) — see Appendix A.
- **Cover corpora:** matched-genre natural text (so "anomalously stilted" is measured against the right baseline, not generic web text).
- **Sender panel:** frontier models via OpenRouter/Venice, **including Opus 4.8** for the Anthropic-relevant cells.
- **Reference (detector) model:** open-weight, local, logprob-exposing — size-ablated (Phase 1).
- **All live execution is deferred.** This document is the **design + experiment plan**; no model calls are made producing it. Execution is the subsequent implementation plan + harness.

---

## 7. Integration with existing assets

- **SIEVE — new output-side `COVERT_SEMANTIC` detector module**, a *sibling* to the existing input-side normalization (6-stage) + injection-classifier components — **not** a change to normalization (normalization does nothing to semantic stego, by SIEVE's own boundary statement). It reuses SIEVE conventions (typed `NormalizationFlag`-style enums, fail-closed) but carries its **own threat model** and an explicit **empty/unparseable alarm class**.
- **AgentSec-Bench — new `COVERT_SEMANTIC` track**, whose rubric *is the frontier*: J reported **split by bits-regime**, not as a single headline number. Deployment guidance baked into the track: the detector covers the high-bit regime; **low-bit coordination is out of monitoring scope by design** and must be addressed by capability isolation.
- **INJEST** (the safe corpus-ingestion / sanitizing-loader module) is the dependency that guarantees any attack corpus (GLOSSOPETRAE, L1B3RT4S, etc.) is normalized through SIEVE Layer-1 *before* any model or agent in the harness touches it. Specced separately.

---

## 8. Falsification criteria (what kills the thesis)

- **Phase 1 fails to reproduce** GLOSSOPETRAE's null / J=0.58 within CI → the instrument is wrong; stop and fix before any extension.
- **Phase 2 detector doesn't generalize** across constructions (overfits acrostics) → it's a construction-specific trick, not a structural-signature detector; demote the claim from "detector" to "construction probe."
- **Phase 3 frontier shows no floor** (best-detector J stays > 0 even at 1 bit / large N) → the capacity-bound thesis is **false**; report the surprising positive result (a black-box detector that catches 1-bit channels would itself be a major finding).
- **H1 confirms** strong prior art → pivot headline (Phase 0 kill switch), do not claim "first."

---

## 9. Risks & mitigations

- **Dual-use.** Phase 3 builds an adaptive stego attacker. Mitigation: defensive framing throughout; **no turnkey payload release**; attacker code gated/withheld from public artifacts; the published contribution is the *defender's* capability statement (the frontier), not an evasion tool.
- **Corpus-ingestion attack surface (SM8; governed by WORKFLOW ADR-0007 repo-config + ADR-0006 sanitize-bytes).** Demonstrated live this session: GLOSSOPETRAE shipped a `.claude/skills/glossopetrae/` that **auto-registered a skill into the harness** — a repo-shipped *skill* injection surface that ADR-0007's current scanner glob (`settings*.json` / `hooks/**` / `.mcp.json` / instruction files) does **not** explicitly cover. *Enhancement candidate (WORKFLOW, see §ADR note):* add `**/.claude/skills|agents|commands/**` to `scan-repo-agent-config.sh`. Mitigation applied this session: skill refused / never invoked; repo scanned for hidden-Unicode + lifecycle payloads (clean); `.claude/` quarantined out of the tree. Standing control: open untrusted clones under ADR-0007 containment (sandbox + parent-dir launch + pre-scan); route all corpora through **INJEST → SIEVE Layer-1** before model contact; treat all corpus text as data, never instructions.
- **Reference-model mismatch.** P_ref ≠ P inflates δ_min. Mitigation: the size-ablation quantifies it; report the floor as detector-relative plus the P_ref=P oracle bound.
- **Compute / API.** Bounded by the bits×N×panel grid; local reference model removes per-token-logprob API cost; sender-side frontier calls are the main spend, deferred to execution.
- **Anchor is unreplicated (Tier D).** Mitigation: Phase 1 *is* the independent replication; we publish our reproduction status alongside any extension.

---

## 10. Deliverables

1. This design doc → **implementation plan** (next step, via writing-plans) → (later) **harness + paper**.
2. **Positioning:** reproduces the prior-art rigor bar before extending; lands a clean novel result (frontier + floor + capacity bound); monitoring-relevant (the self-monitoring blindness cells, the "cross-vendor monitoring is the *hole*, not the fix" inversion, and the empty-reply monitor-DoS class).
3. Target venue framing: a clean gap claim for **IEEE S&P / USENIX Security** — "black-box detectability frontier of LLM semantic steganography, with the low-bit-coordination floor as the safety-relevant corollary."

---

## 11. Scope boundaries

**In scope:** semantic, natural-language-only stego; black-box text-only uninformed monitor; perplexity/distributional detection; adaptive attacker; capacity–detectability frontier; the constructions in Appendix A.

**Out of scope (separate research items):** physical Unicode carrier channels (normalization solves; SIEVE Layer-1); informed/key-possession monitors (GLOSSOPETRAE §4.3.1); cooperating-sender watermarking detection (assumption violated by adversarial stego; H5); the AISec-Guard→SIEVE rename pass across existing specs; the INJEST module spec; active countermeasures (paraphrase/transform defenses).

---

## 12. Resolved judgment calls (from the brainstorming session)

| Decision | Resolution |
|----------|-----------|
| Primary thrust | Semantic-stego detection |
| Output of this session | Design doc + experiment plan (no live calls) |
| Detector access model | Black-box, text-only |
| Eval standard | Reproduce-first, then adaptive |
| Program shape | All three layers; **frontier as headline** |
| Integration | SIEVE Layer-2 `COVERT_SEMANTIC` + AgentSec-Bench track |
| Reference model | Open-weight, local, logprob-access; **size-ablated**; Pi 5 = 1–3B prototyping tier |
| Headline emphasis | Capacity floor / 1-bit corollary (not detector-beats-0.58) |
| Framework name | **SIEVE** (umbrella); **INJEST** = safe-ingestion module |

---

## Appendix A — Reproduction parameters (from GLOSSOPETRAE, to verify in Phase 1)

- **Constructions:** 11.8-bit {color, number, shape} 3-tuple (chance ≈ 0.028%) via acrostic / synonym / sentence-length / positional / metaphor; 17.4-bit 5-tuple; 3-sentence narrative-positional.
- **Perplexity-anomaly baseline:** K=3 ratings per cover, z > 1.5 flag vs n=17 cover-calibration; reported TPR=83.3% [58.6, 96.4], FPR=25% [8.7, 49.1], **J=0.58 [0.09, 0.88]**, n=30; 100% of flags "anomalously stilted."
- **Confirmed null (n=150, current frontier):** GPT-54-mini J=0.00, Opus 4.8 J=−0.01, Gemini 3.5 Flash J=0.00; CI width ±0.024.
- **Refuted detectors (J=0, n=30):** multi-sample consistency (K=5 majority), paraphrase round-trip, stylometric (6 features, z>2; failure mode TPR=FPR=100%).
- **Factorial (n=150/cell, 1,800 trials):** key effect mean +41.6pp ±10 (Bonferroni p ≤ 2e-8); task-spec effect mean +2.2pp ±8.3 (confirmed null) → ~95% of gain is key-possession.
- **Named refuted baselines to L3-verify:** Ziegler et al. 2019; Ueoka et al. 2021 (GLOSSOPETRAE §5.3).
