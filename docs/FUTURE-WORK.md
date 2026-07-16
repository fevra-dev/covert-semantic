# Future work — the agenda these results open

The completed study maps semantic-stego detectability along two axes (what the monitor *knows*;
what *form* the payload arrives in). Five directions follow directly from its findings and its
honest limitations. Each notes what is already known, what it would take to resolve, and why it
matters for monitoring-based safety.

---

## 1. The physical × semantic cross-product

**Question.** This work studies *semantic/rhetorical* hiding in unmodified text. Public red-team
tooling (see the README's Related-work note) covers the *physical/character* channel — Unicode
tags, emoji variation selectors, ciphers, QR/image renders. **No one has composed the two.**

**Experiment.** A full factorial: {acrostic, synonym, sentence-length, narrative} × {plain text,
image render, Unicode-tag overlay, emoji-selector, QR} × {informed monitor}. Does stacking a
physical carrier on a semantic payload *multiply* evasion, and does a reconstruct-then-monitor
chokepoint hold against the composite?

**Needs.** A clean multimodal harness and frontier-monitor access at a scale API budgets can't
sustain. **Why it matters.** It measures whether the two independently-studied evasion families
interact — the case a real monitoring pipeline actually faces.

## 2. A rigorous adaptive attacker (the detectability frontier)

**Question.** The current detectors face a *static* sender. Random-combo bypass tools are a crude
adaptive attacker; the disciplined version optimizes the payload against the live monitor.

**Experiment.** Black-box optimization of the payload against a real informed monitor to trace the
**detectability frontier** — best achievable Youden J as a function of *payload bits × text length
× attacker budget* — down to the low-bandwidth limit (the 1-bit `proceed/abort` coordination
signal), tied to the Cachin/TV information-theoretic bound. Evaluate to the "attacker moves second"
standard (Carlini, `2510.09023`).

**Needs.** A sustained optimization loop against production monitors. **Why it matters.** It turns
"a monitor detects at chance / recovers when informed" into a *quantified capacity limit*, and
locates the regime that must be defended architecturally rather than by monitoring.

## 3. The reallocation gap — make it a non-tradeoff, then look inside

**Question.** Prompting an image monitor to "scrutinize the words" *relocates* its blind spot
(word-level recovered, gist-level lost); only one model ran both passes at once. Two follow-ups:

- **Behavioral.** Can a monitor be prompted or lightly tuned to run gist *and* word-level analysis
  together — a two-headed chokepoint monitor that pays no reallocation cost?
- **Mechanistic.** Is the "analytic budget" visible internally — do image tokens route away from
  the circuits that text tokens engage for word-level scrutiny? This is where the black-box result
  meets interpretability.

**Needs.** Model-internal access for the mechanistic half. **Why it matters.** It would convert a
measured limitation into either a fix (the two-headed monitor) or a mechanistic account of why
vision-routed text is under-analyzed.

## 4. The two deferred open questions

- **Why the OpenAI-lineage detection edge?** A weaker OpenAI model out-detects a stronger
  non-OpenAI one, so it tracks lineage, not capability. Resolving *post-training exposure vs. task
  skill* needs either a second detecting lineage or training-process insight.
- **Does a reasoning budget expand the analytic pass, or just relocate it?** The reasoning-on
  condition was deferred here (unreliable via a third-party router); with direct frontier API
  access it is a clean A/B against the snap-judgment condition used throughout.

## 5. The through-line — a reconstruct-then-monitor chokepoint

Every direction converges on one shippable primitive the study's conclusion already points to:
**reconstruct any laundered carrier back to text, then run an informed monitor on the reconstructed
text, and calibrate its false-positive rate per model.** Build it, and evaluate ASR reduction
across the full physical × semantic × modality attack surface from (1). That is the step from
"interesting measurements" to a monitoring component robust across the surface an adversary can
actually reach.

---

*All five follow from evidence already in this repository; none is claimed as a result here. They
are the questions the completed work makes askable — and answerable — next.*
