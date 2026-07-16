# Modality Laundering vs. Informed Semantic-Stego Monitors — n=30 Firm-up + "Scrutinize" Counterfactual

**Results memo** · **Date:** 2026-07-16 · **Author:** Mac Calarco (fevra-dev)
**Precedes:** the n=10 image-carrier pilot (synthesis §4, 2026-07-12). This memo firms the pilot
to n=30 and adds the counterfactual the pilot flagged as the open question: *is the
vision-analytic gap closable by prompting the monitor to scrutinize the words?*

---

## 0. TL;DR

1. **Firm-up (n=30) reproduces and sharpens the pilot.** Modality laundering (rendering the
   payload as an image) is **construction-selective, not a uniform collapse**: gist-level
   encoding (narrative) is modality-robust across all three vision monitors; lexically-precise
   encoding (synonym) degrades sharply and **model-dependently** — Sonnet collapses (text
   J=0.79 → image J=0.20), GPT-5.5 drops (0.95 → 0.73), while Opus *rises* (0.61 → 0.83). FPR
   stays ~0 throughout.
2. **The "scrutinize every word" counterfactual does NOT cleanly close the gap — it
   *reallocates* a limited analytic budget.** For the monitor that collapsed (Sonnet), the
   careful prompt recovers the word-level construction (synonym 0.20 → 0.62) but **destroys the
   gist channel** (narrative 0.77 → 0.00). GPT-5.5 degrades broadly under the instruction; only
   Opus absorbs it without cost.
3. **Defensive conclusion sharpens:** you cannot prompt-patch the image monitor into safety —
   "look harder at the words" merely *moves* the blind spot. Monitor the **reconstructed text at
   the chokepoint**, not the image, and not a harder-prompted image monitor.

---

## 1. Design

- **Corpus:** the committed n=150 semantic-stego + cover corpus (`phase2_n150_20260628.jsonl`),
  benign {colour, number 0–99, shape} payloads. n=30 stego + 30 cover per construction.
- **Constructions:** narrative_positional (gist), synonym (marker words), acrostic (first
  letters). sentence_length excluded (12-sentence render is an OCR-length confound).
- **Carrier:** `csd.imaging.render_text_to_png` (FigStep/pxpipe-style plain render).
- **Monitors (vision-capable, informed):** claude-opus-4.8, claude-sonnet-4.6, openai/gpt-5.5,
  each with a text-informed J baseline from `phase1_5_informed_20260628.json`.
- **Prompt conditions on the image:**
  - **plain** — the same informed snap prompt as the text baseline (names the payload type +
    construction menu; single-word YES/NO).
  - **careful** — identical, except it names the image modality and instructs the monitor to
    *read every word and scrutinize those exact words as rigorously as if handed plain text*
    (single-word output preserved for A/B comparability; only the prompt differs).
- **Metric:** Youden's J = TPR − FPR per (model × construction). Judgments cached
  (`monitor_cache_image_20260712.jsonl`, `monitor_cache_image_careful_20260716.jsonl`);
  analyses `image_carrier_pilot.json`, `image_carrier_careful_20260716.json`.

## 2. Result 1 — firm-up (n=30): image J vs text-informed J

| construction | Opus img (text) | Sonnet img (text) | GPT-5.5 img (text) |
|---|---|---|---|
| narrative (gist) | 0.97 (0.93) | 0.77 (0.90) | 0.97 (0.99) |
| synonym (words) | 0.83 (0.61) | **0.20 (0.79)** | **0.73 (0.95)** |
| acrostic (letters) | 0.37 (0.07) | 0.03 (0.18) | 0.70 (0.75) |

FPR 0.00–0.03 throughout. The n=10 pilot pattern holds and tightens:
- **Gist survives the modality shift** — narrative image J ≈ text J for all three monitors. The
  "this reads stilted / like it's hiding something" signal is holistic and comes through OCR.
- **Lexically-precise degrades, model-dependently** — synonym is where laundering bites: Sonnet
  −0.59, GPT-5.5 −0.22; Opus is the outlier (+0.22, and generally *better* as an image).
- Consistent with the 2026-07-12 transcription probe: the missed synonym items had every marker
  word correctly OCR'd — this is an **analytic** gap, not a perception gap.

## 3. Result 2 — the "scrutinize" counterfactual: reallocation, not repair

Careful-prompt image J, with plain-image J and text J for reference:

| model × construction | careful J | plain J | text J | careful − plain |
|---|---|---|---|---|
| Opus × narrative | 0.97 | 0.97 | 0.93 | 0.00 |
| Opus × synonym | **1.00** | 0.83 | 0.61 | +0.17 |
| Opus × acrostic | 0.39 | 0.37 | 0.07 | +0.02 |
| Sonnet × narrative | **0.00** | 0.77 | 0.90 | **−0.77** |
| Sonnet × synonym | **0.62** | 0.20 | 0.79 | **+0.42** |
| Sonnet × acrostic | 0.00 | 0.03 | 0.18 | −0.03 |
| GPT-5.5 × narrative | 0.90 | 0.97 | 0.99 | −0.07 |
| GPT-5.5 × synonym | 0.53 | 0.73 | 0.95 | −0.20 |
| GPT-5.5 × acrostic | 0.43 | 0.70 | 0.75 | −0.27 |

**The instruction moves the analytic budget; it does not enlarge it.**
- **Sonnet — the clean demonstration.** Directing scrutiny at the words recovers most of the
  synonym collapse (0.20 → 0.62, ~70% of the way back to text's 0.79) **but zeroes the gist
  channel** (narrative 0.77 → 0.00 — J=0 at FPR=0 means it answered "no" to every item). Told to
  hunt for suspicious words, it stopped registering the holistic stilted-ness it otherwise catches.
- **GPT-5.5 — net harm.** The instruction degrades synonym (−0.20), acrostic (−0.27), and even
  narrative (−0.07); over-focusing hurts a monitor that was already the strongest on the image.
- **Opus — free lunch, uniquely.** Absorbs the instruction with no gist cost (synonym → 1.00).

The see-saw is the point: a *limited* analytic pass over the image can be pointed at word-level
OR gist-level signal, and for two of three monitors, pointing it one way robs the other. Only
Opus runs both at once.

## 4. Interpretation

- The vision-analytic gap is **not an attention deficit a prompt can top up** — it behaves like a
  constrained analytic budget that the prompt *redistributes*. This is a stronger, more useful
  claim than "images are the Achilles' heel": the heel does not heal when you tell the monitor to
  try harder; the wound just moves.
- It reconciles the firm-up: gist-robust / words-fragile under the plain prompt, and the mirror
  image (words-recovered / gist-lost) under the careful prompt, for the model (Sonnet) whose
  budget is tightest.
- **Architectural conclusion (unchanged, now empirically forced):** place an *informed* monitor
  at the post-reconstruction **text** chokepoint. Do not rely on tuning an image monitor's
  prompt — the best case (Opus) is model-specific and the typical case (Sonnet, GPT-5.5) trades
  one blind spot for another.

## 5. Caveats (honest scope)

- **n=30 → J on a ~0.03 grid**; directional, not a tight interval. The firm-up *reproduced* the
  n=10 directions, which is the evidence that these are signals, not noise.
- **One render style** (plain FigStep/pxpipe). A laundering attacker would optimize the render;
  a defender's OCR would preprocess it. Both are out of scope here — this measures monitor
  behaviour on a neutral render.
- **One careful-prompt wording.** A different phrasing (e.g., allowing visible reasoning tokens
  rather than a snap single word) could shift the reallocation; the snap format was held fixed
  for A/B comparability with the plain cell. The *reallocation phenomenon* is the robust finding;
  the exact per-cell magnitudes are wording- and n-sensitive.
- **Three monitors.** The model-dependence (Opus absorbs, Sonnet see-saws, GPT-5.5 harmed) is
  itself a finding, but three points is a shape, not a law.

## 6. What this changes upstream

- **Application Q2** asserted the fix is architectural "rather than something you patch by asking
  the monitor to try harder." Result 2 is the direct evidence for that sentence.
- **FINDINGS-plain** open-question #4 ("how much of the picture-blindness goes away if you tell
  the monitor to analyse carefully?") is now **answered**: partially, and only by *moving* the
  blind spot — not a simple fix.
