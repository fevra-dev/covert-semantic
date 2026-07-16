# PRIOR_ART — Phase 0 Novelty Gate (first pass)

**Date:** 2026-06-26 · **Status:** FIRST PASS (web-search saturation not yet reached; 3 collision papers pending L3 read) · **Feeds:** the kill-switch decision for the COVERT_SEMANTIC design (`2026-06-26-covert-semantic-detection-design.md`).

**Question (H1 / kill-switch):** Has any published work achieved **Youden J > 0.58** on **uninformed-monitor** detection of **natural-language-only** (no Unicode carrier) **frontier-LLM semantic steganography** at n ≥ 20?

**Headline (tentative):** The space is **far more crowded (2024–2026) than GLOSSOPETRAE's framing implies.** No paper *yet confirmed* to beat J=0.58 in the strict uninformed setting — but "first detector" and "first perplexity proxy" framings are **at risk** pending L3 reads. The defensible novelty is shifting (as the design's kill-switch anticipated) toward **(a) the adaptive-attacker detectability frontier, (b) the 1-bit agent-coordination floor corollary, (c) rigorous reproduction** — *not* "first detector."

---

## Prior-art matrix (first pass)

| Source | Year | Type | Informed? | Relevance / collision |
|---|---|---|---|---|
| Ziegler, Deng, Rush — *Neural Linguistic Steganography* ([D19-1115](https://aclanthology.org/D19-1115/)) | 2019 | **Generation** (arithmetic coding, low-KL) | n/a (attack) | GLOSSOPETRAE §5.3 anchor — **L3-verified real**. Generation, not detection; the low-KL ideal our frontier references. |
| Ueoka, Murawaki, Kurohashi — *Frustratingly Easy Edit-based…MLM* ([2104.09833](https://arxiv.org/abs/2104.09833)) | 2021 | **Edit generation** | n/a (attack) | GLOSSOPETRAE §5.3 anchor — **L3-verified real**. "More secure vs automatic detection" = attack-side. |
| *Towards Next-Gen Steganalysis: LLMs Unleash…* ([2405.09090](https://arxiv.org/abs/2405.09090)) | 2024 | Detection | **Informed** (fine-tuned Bloomz/Llama on stego/cover) | High accuracy, but **supervised on the generator** → not the uninformed regime. Relevant to **Phase 2** (learned discriminator). |
| *Pseudo-label Domain Adaptation for Zero-Shot Text Steganalysis* ([2406.18565](https://arxiv.org/pdf/2406.18565)) | 2024 | Detection | Partial ("zero-shot" = source-domain labels + pseudo-labels) | "Zero-shot" ≠ uninformed; cross-domain transfer, still trained. Does not clearly meet H1's condition. |
| *Linguistic Steganalysis via LLMs: Two Modes…Strongly Concealed Stego* ([2406.04218](https://arxiv.org/pdf/2406.04218)) | 2024 | Detection (LLM-based) | **L3 NEEDED** | Collision candidate — "strongly concealed" detection via LLMs. Check informed/uninformed + metric. |
| **Now You (Still) See Me: Detecting Evasive Stego Payloads in LLMs** ([2606.09411](https://arxiv.org/html/2606.09411)) | 2026 | Detection (evasive) | **L3 CRITICAL** | Jun-2026, *evasive* payload detection — closest to our adaptive setting. Could pre-empt Phase 2/3. |
| **Kolmogorov Complexity Bounds for LLM Stego + a Perplexity-Based Detection Proxy** ([2603.21567](https://arxiv.org/pdf/2603.21567)) | 2026 | **Theory + detection** | **L3 CRITICAL** | Strongest collision: overlaps **both** our Phase-1 perplexity baseline **and** the Phase-3 capacity bound. Must read before claiming either as novel. |
| *A Decision-Theoretic Formalisation of Stego w/ Applications to LLM Monitoring* ([2602.23163](https://arxiv.org/pdf/2602.23163)) | 2026 | Theory / framing | **L3 NEEDED** | Collision on our threat-model framing (LLM monitoring). |
| *Hidden in Plain Text: Steganographic Collusion in LLMs* ([2410.03768](https://arxiv.org/html/2410.03768v2)) | 2024 | Attack + mitigation | — | Multi-agent **collusion** = our 1-bit `proceed/abort` corollary's space. Cite for the agent-safety angle. |
| *Provably Secure Generative Linguistic Steganography* ([2106.02011](https://arxiv.org/pdf/2106.02011)) | 2021 | Attack (zero-KL) | n/a | The dark-side bound: zero-KL stego is undetectable by construction → our frontier's floor at the limit. |
| Cachin — *An Information-Theoretic Model for Steganography* | 1998 | Theory | — | Foundational: KL security; total-variation bounds any detector. **Grounds our RQ4 capacity bound.** |
| *Steganographic Potentials of Language Models* ([2505.03439](https://www.emergentmind.com/papers/2505.03439)); *Hide & Seek in Embedding Space* ([2601.22818](https://arxiv.org/html/2601.22818)) | 2025–26 | Attack/detection | — | Frontier-capability + geometry-based detection context. |

---

## ACH resolution (updated probabilities)

- **H1** — prior art beats J>0.58 uninformed: **20–35% → ~45–55% (elevated, UNRESOLVED).** *Moved by:* the dense 2024–26 LLM-steganalysis literature. *But:* most is supervised/known-generator or domain-transfer (scope inflation); none yet **confirmed** in the strict uninformed-no-key-no-examples regime. **Resolve via L3 reads of 2603.21567, 2606.09411, 2406.04218.**
- **H2** — perplexity-anomaly improvable to J≥0.8: **~50% (unchanged).** *Moved by:* 2603.21567 explicitly formalizes a perplexity-based detection proxy — perplexity is a real, actively-studied signal (supports feasibility) **but no longer novel to us**; our angle is calibration + adaptive robustness, not "first proxy."
- **H3** — the 3 refuted GLOSSOPETRAE baselines are genuinely weak: **65–75% → ~75% (supported).** *Moved by:* the field has moved past hand-crafted stylometry to learned/LLM steganalysis; the refuted baselines are indeed inadequate.
- **H4** — info-theoretic detectability bounds exist: **45–55% → ~85% (confirmed).** *Moved by:* Cachin (1998) KL + total-variation detector bound; capacity-upper-bound papers; 2603.21567 Kolmogorov bound. Our RQ4 should *cite and build on* this, not claim it.
- **H5** — watermarking detection transfers: **~15% (unchanged, low).** Watermark detection assumes a cooperating sender; violated by adversarial stego.

---

## Kill-switch verdict (tentative) + novelty pivot

**Not a hard kill, but a partial fire.** The design's kill-switch said: *if strong prior art exists, pivot the headline from "first detector" to the frontier/capacity result.* The first pass supports that pivot:

1. **Drop any "first detector / first perplexity proxy" framing** — both have 2024–26 prior art. Reframe the Phase-1 perplexity baseline as **reproduction of a known signal** (cite 2603.21567 + GLOSSOPETRAE), not a contribution.
2. **Sharpen the contribution to the defensible core:** the **adaptive-attacker detectability frontier** (J vs bits × N × fluency-budget) tied to the **Cachin/TV KL bound**, plus the **1-bit agent-coordination floor** corollary (cite the collusion literature, 2410.03768). This appears genuinely under-explored — *pending* confirmation that 2606.09411 (evasive detection) and 2603.21567 (bounds) didn't already trace it.
3. **Reproduction value stands regardless** — an independent uninformed-monitor reproduction at the frontier is still worth publishing.

**Confidence:** moderate, first-pass. The pivot is robust; the *exact* residual novelty depends on the 3 L3 reads.

---

## Boundary statement + next step

This first pass covers arXiv/ACL/ScienceDirect surface search on linguistic steganalysis, LLM stego detection, and info-theoretic bounds. **Not yet done:** L3 close-reads (the dominant failure mode is scope inflation — "detection" papers that are actually informed/known-generator). **Immediate next step (fan-out, ADR-0009 discovery):** L3-read the three collision papers — **2603.21567**, **2606.09411**, **2602.23163** (+ 2406.04218) — verifying for each: (a) natural-language-only (no Unicode), (b) genuinely uninformed monitor (no key, no target-generator labels), (c) n ≥ 20, (d) a Youden-J-comparable metric. One subagent per paper is the clean parallelization. Update H1 + this file after.

---

## RESOLVED — L3 verdicts on the collision papers (2026-06-26, supersedes the "pending" flags above)

Four collision papers L3-read (one subagent each, cross-checked). **H1 is NOT confirmed: no published work cleanly beats J=0.58 on uninformed, black-box, natural-language, frontier-LLM semantic-stego detection at n≥20.** One paper (`2603.21567`) *dents* the perplexity-baseline novelty.

| Paper | L3 verdict | Deciding evidence |
|---|---|---|
| `2406.04218` (LSGC) | **LEAVE-INTACT** | Fully **supervised** (6:2:2 train/test on labeled pairs from target generators VAE-Stega/LLsM); Acc/F1 only, no J; no adaptive attacker; best ≈J0.64 is *supervised*. Fails C1's uninformed condition. |
| `2606.09411` ("Now You Still See Me") | **LEAVE-INTACT** | **White-box, decode-key-aware, activation-level** probes of fine-tuned **trojans**; encoding = embedding-space buckets (not human-semantic). Adaptive evasion is in the *activation* domain (white-box). C2/C3 overlap only by analogy on a different surface; C1/C4 untouched. |
| `2602.23163` (Decision-theoretic monitoring) | **LEAVE-INTACT** | **Utility-based** "steganographic gap" (Generalised V-Information), **not** an info-theoretic capacity bound; no KL / per-token / bandwidth-floor; no adaptive game (sandbagging noted, not analyzed); no perplexity. Incompleteness results are framework-local, not a universal detector bound. C3 fully intact. |
| **`2603.21567`** (Kolmogorov + perplexity proxy) | **DENTS** | **The real collision.** Uninformed perplexity-vs-baseline steganalysis ("Quadroculars"; lineage = Binoculars AI-text detector) on NL semantic stego = our **C4 concept**. **But**: toy color encoder (3 bits/sentence), n=300, **only a paired t-test — no ROC/TPR/FPR/J**, so it cannot claim/refute J>0.58 and is not at 11–17-bit frontier scale. Kolmogorov bound (asymptotic, uncomputable) ≠ our operational per-token-KL; **no 1-bit corollary**. |

### Resolution & sharpened thesis
- **C4 (perplexity proxy) — demoted from contribution to *reproduction*.** It is prior art (`2603.21567`). Phase 1 reframes as *reproducing + rigorously J-scoring* the proxy at frontier scale (true cover baseline, calibrated threshold, full ROC) — the prior work is preliminary (one toy scheme, one t-test).
- **C1 — drop "first detector."** Reframe: first *rigorous, frontier-scale, uninformed, J-scored* evaluation, benchmarked against the `2603.21567` proxy.
- **C2 (black-box adaptive detectability frontier) — INTACT; strongest novel claim.** No paper traces best-J vs (payload-bits × text-length × fluency-budget) against a *black-box text* detector. Ground in Carlini "The Attacker Moves Second" (`2510.09023`).
- **C3 (operational per-token-KL bound + 1-bit agent-coordination floor) — INTACT.** Distinct from `2603.21567` (Kolmogorov) and `2602.23163` (utility-gap). Ground in Cachin (1998) KL/total-variation; the 1-bit corollary is unique — connect to collusion (`2410.03768`).

### Citation set (load-bearing)
`2603.21567` (perplexity proxy + Kolmogorov bound — closest prior art: cite + beat) · Cachin 1998 (KL/TV detectability) · `2510.09023` Carlini (adaptive standard) · `2602.23163` (decision-theoretic LLM monitoring — related framing) · `2606.09411` (white-box activation evasion — complementary surface) · `2410.03768` (LLM steganographic collusion — agent-safety motivation).

### ACH final
- **H1: ~50% → ~20% (RESOLVED — not beaten in the uninformed setting).**
- **H2:** reframed — the target is *beating the `2603.21567` proxy with rigorous J-scored eval*, not "first proxy."
- **H4: ~85% (confirmed)** — Cachin KL/TV bound is the grounding (cite, don't claim).
- **Kill-switch:** does NOT fire for C2/C3 (intact); **fires for C4** (perplexity = prior art → reproduction).

### Closely-related monitor-evasion work (cross-check, 2026-06-26)

The closest *monitor-evasion* work in the literature is two recent agent-monitoring
benchmarks. L3-read; both **LEAVE-INTACT**, and crucially both are **behavioral**, not semantic:

| Paper | Verdict | Why complement, not collision |
|---|---|---|
| **SLEIGHT-Bench** (`2605.16626`, Roger & Benton) | LEAVE-INTACT (C1–C4) | 40 hand-crafted transcripts; **behavioral** evasion of agent monitors by exploiting monitor *capability gaps* (state-tracking, counting, long-context) — not semantic/linguistic concealment. Black-box text-level. **No steganography, no info-theoretic bound, no adaptive frontier** (attacks human-authored, "no iteration"). Opus 4.6 catches 32% @1%FPR; 50% never caught. |
| **SHADE-Arena** (`2506.15740`, Benton) | LEAVE-INTACT (C1–C4) | Single-agent **behavioral** sabotage vs a black-box transcript-monitor (best AUC 0.87; humans 0.65). No inter-agent covert comms, no low-bandwidth signaling, no info-theory, no adaptive optimization. |

**Synthesis — novelty holds even against these benchmarks; they become motivation:**
- C2 (adaptive detectability **frontier**) and C3 (capacity bound + 1-bit floor) are **absent** from both — neither has an attacker-budget axis or any information theory. Our work sits in the **gap between** these *behavioral*-evasion benchmarks and the *steganalysis* literature: **semantic concealment in the output token-distribution, with an info-theoretic floor.**
- SLEIGHT-Bench + SHADE-Arena empirically show black-box monitors have systematic blind spots even for *high-bandwidth behavioral* attacks; **C3 explains the deeper theoretical floor for *low-bandwidth semantic/covert* signals.** Cite both as motivation; Carlini `2510.09023` for the adaptive standard.
