# covert-semantic

**Can one AI hide a small structured secret inside ordinary text — and can another AI catch it?**
A black-box study of semantic-steganography *detectability*, mapped along two independent axes:
what the monitor **knows**, and where the monitor **sits** (the form the payload arrives in).

This is a defensive-measurement project. It generates a *benign* payload corpus (a colour, a
number 0–99, a shape) hidden in fluent prose, and measures how well frontier models, acting as
monitors, detect it. No attack tooling; the artifact of interest is the *monitor's* behaviour.

For the plain-language tour, read **[`FINDINGS-plain.md`](FINDINGS-plain.md)** first.

---

## What we found

1. **A trained distributional detector learns a *style* of hiding, not hiding in general.**
   In-distribution AUC 0.94–1.00 (Holm-significant on all four constructions), but the win does
   not uniformly survive a held-out construction (leave-one-construction-out). Honestly demoted
   from a universal detector to a *construction/family probe*. An ensemble acts as a robustness
   floor, not a peak.

2. **Asked cold, a frontier monitor is blind to hidden payloads — including its own.** A naive
   *"is anything hidden here?"* monitor reproduces the near-total null (Youden's J ≈ 0) across
   constructions, including a 100%-decodable acrostic. Only OpenAI-lineage models
   spontaneously detect (gpt-5.5 J = 0.53; gpt-5.4 J = 0.19) at 0% FPR; a *stronger*
   non-OpenAI model (Grok) stays blind — so it tracks model **lineage**, not raw capability.

3. **Tell the monitor what to look for, and the blindness mostly lifts.** An *informed* prompt
   that names the payload type and the construction menu — but never the secret — recovers most
   detection (pooled J: self-monitor 0 → 0.44, DeepSeek 0 → 0.72, gpt-5.5 0.53 → 0.77). This
   *contradicts* prior art claiming task-spec barely helps. Two honest wrinkles: one model
   (Grok) became trigger-happy (FPR ≈ 0.39 — alarm, not skill), and structural constructions
   (acrostic, sentence-length) stay hard except where a specific model happens to have the knack.

4. **Render the same payload as an image, and a *warned* monitor goes soft again — not because
   it can't read it.** Modality laundering (text → picture-of-text, n=30/cell) is
   *construction-selective*: gist-level hiding (narrative) survives the shift; lexically-precise
   hiding (planted synonyms) degrades model-dependently (Sonnet text J 0.79 → image 0.20). A
   verbatim-transcription probe rules out OCR failure — the missed items had every marker word
   transcribed correctly and were *still* waved through. It is a **vision-vs-text analytic gap**:
   the word-level scrutiny that catches stego in text does not fire the same way over an image of
   the same text, even with perfect OCR.

5. **And you can't prompt your way out of it.** Instructing the image monitor to "scrutinize every
   word as rigorously as plain text" does not add analytic care — it *reallocates* a limited
   budget: the monitor that collapsed (Sonnet) recovers the word-level construction (synonym
   0.20 → 0.62) but goes blind to the gist channel (narrative 0.77 → 0.00); GPT-5.5 degrades
   broadly; only Opus absorbs the instruction. The blind spot relocates rather than closes —
   which is why the defense is architectural, not a better prompt: monitor the reconstructed
   **text** at the chokepoint. (See `docs/2026-07-16-…-modality-n30-results.md`.)

**The one-line synthesis:** the machines aren't blind — they look carefully only when you *tell*
them to, and only when the message arrives as *words*. The defensive recipe follows directly:
tell the monitor what to look for, inspect the payload as reconstructed **text** at the
chokepoint (never the image), and watch its false-positive rate per model.

---

## Reproduce

Requires [`uv`](https://docs.astral.sh/uv/) and a local reference model (see
[`REFERENCE_MODEL_SETUP.md`](REFERENCE_MODEL_SETUP.md)). Live monitor runs need an
OpenRouter API key (`OPENROUTER_API_KEY`); the full test suite runs offline.

```bash
uv sync
PYTHONPATH=. uv run python -m pytest        # offline test suite

# live cross-model monitor panel (cached; resumable):
OPENROUTER_API_KEY=... PYTHONPATH=. uv run python experiments/phase1_5_cross_monitor.py \
    --live --corpus experiments/corpus/phase2_n150_20260628.jsonl
```

Every live judgment is cached to a `.jsonl` in `experiments/` and committed as evidence, so
analyses re-run offline and for free. `experiments/*.json` are the committed result matrices.

## Layout

| Path | What |
|---|---|
| `csd/` | the package: corpus, constructions, monitors, detectors, stats, imaging |
| `tests/` | offline test suite (deterministic mocks for every live path) |
| `experiments/` | run scripts + cached judgments + result matrices + the n=150 corpus |
| `docs/` | design + per-phase results memos (the rigor record) |
| `PRIOR_ART.md` | the novelty gate — what is reproduction vs. contribution |
| `FINDINGS-plain.md` | plain-language summary |

## Honesty & attribution

Verdicts here are pre-registered and reported as found: the perplexity baseline is labelled an
honest *reproduction* of prior art (not a first); the Phase-2 universal claim is *demoted* to
per-construction where held-out evidence did not support it; caveats (in-sample calibration,
coarse pilot n, trigger-happy FPR) are stated, not buried.

This is AI-assisted research: the design, framing, verdicts, and honest self-demotions are the
author's; an AI coding assistant was used to implement and test the instrument under that
direction. Findings reflect the state of the runs recorded in `experiments/` and are research
results, not production guarantees.
