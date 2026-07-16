# SIEVE · COVERT_SEMANTIC — Phase 1.5: Cross-Model Naive Auditor

**Design Document (spec)**
**Date:** 2026-06-28 · **Author:** Mac Calarco (fevra-dev)
**Status:** Design — approved (brainstorm 2026-06-28), pending plan
**Parent design:** `2026-06-26-covert-semantic-detection-design.md` (Phase 1.5, the self-attribution / cross-vendor-monitoring question)
**Reuses corpus:** `covert-semantic/experiments/corpus/phase2_n150_20260628.jsonl` (1,200 Opus gens) — **no new generation**

> **One-line thesis.** GLOSSOPETRAE's J≈0 null is *self*-monitoring (a model judging its own kind). Roger's self-attribution bias predicts a model is worse at catching itself. Test it: run the **same naive YES/NO monitor** over the Opus-generated corpus with a 4-model panel — **self (Opus), same-vendor (Sonnet), and two independent cross-vendor labs (OpenAI, Google)** — and ask whether a cross-vendor auditor catches the semantic stego self-monitoring misses, and whether the blind spot is shared across the Anthropic lineage.

---

## 1. Research question

- **RQ1.5a (replication):** Does Opus-self naive monitoring of Opus-generated semantic stego reproduce J≈0 at n=150 across **4 constructions** (extends the Phase-1 n=30 narrative-only null)?
- **RQ1.5b (cross-vendor):** Do **cross-vendor** naive monitors (`openai/gpt-5.5`, `google/gemini-3.1-pro-preview`) achieve J significantly > self, on the *same* items, Holm-corrected?
- **RQ1.5c (lineage):** Does the **same-vendor** monitor (`anthropic/claude-sonnet-4.6`) share Opus's blind spot (J≈0 → shared-training-lineage effect) or catch it (→ self-only effect)?

The defensive payoff if RQ1.5b holds: *self-monitoring is the wrong architecture; a cross-vendor auditor is materially better* — directly relevant to cross-vendor monitoring and the Opus-family-trait blindness the parent design flags.

## 2. Threat model / condition

The monitor is the **naive black-box YES/NO monitor** — the exact GLOSSOPETRAE null condition (`MONITOR_SYSTEM` unchanged): text-only, no logits, no key, no construction hint. This is deliberate: the comparison is only meaningful against the *same* naive condition that produced the self-null. (The "informed monitor" factorial — telling the monitor what to look for, which GLOSSOPETRAE showed buys ~5% — is **deferred**; cached judgments make it a free follow-up.)

## 3. Architecture — monitor-calls-only, cached

```
corpus (1,200 Opus gens) ──► for each panel model ──► ApiMonitor.judge(text)  [cached]
                                                          │
                                            raw reply ──► parse_monitor_reply ──► Verdict (YES/NO/empty/unparsed)
                                                          │
                                       per (model × construction) ──► J + CP-CI, FPR, McNemar vs self, Holm
```

- **Reuse:** `csd/monitors.py` (`ApiMonitor(model)` → OpenRouter), `csd/detectors/naive_monitor.py` (trichotomy parse), `csd/harness.py`, `csd/stats.py` (`youden_j_ci`, `mcnemar`, `holm_bonferroni`).
- **New, small:** a **judgment cache** (`csd/monitor_cache.py`, mirrors `corpus.py`) keyed by `(model, record_id)` → raw reply, append-flushed/resumable, so a crash or a later panel/informed expansion re-spends nothing.
- **Refactor (behavior-preserving):** extract `parse_monitor_reply(raw) -> Verdict` from `NaiveMonitorDetector.score` so the driver can parse cached replies without re-calling the monitor; `score()` then calls it. Re-tested green.

**Monitor params:** `max_tokens=16`, `temperature=0` (slightly above the current default 8 — cheap insurance against a model needing a token of leeway; the trichotomy still flags any empty/garbled reply rather than scoring it "clean"). The 3 empty-response stego records are skipped (cannot monitor empty text).

## 4. Panel (exact OpenRouter IDs, verified present 2026-06-28)

| role | model | lineage | naive-condition config |
|---|---|---|---|
| self | `anthropic/claude-opus-4.8` | Anthropic (generator) | plain (naive by default) |
| same-vendor | `anthropic/claude-sonnet-4.6` | Anthropic | plain (naive by default) |
| cross-vendor A | `openai/gpt-5.5` | OpenAI | `reasoning:{enabled:false}` |
| cross-vendor B | `deepseek/deepseek-v3.2` | DeepSeek | `reasoning:{enabled:false}` |

OpenRouter uses dot-notation (`claude-sonnet-4.6`, not `-4-6`) — verified against the live model list.

**Reasoning config (verified live 2026-06-28, the smoke earned its keep).** The naive condition requires a quick YES/NO with *no* deliberate reasoning — uniform across the panel. Findings: Anthropic models answer naively by default and **must not** receive a `reasoning` param (it makes them return empty); reasoning models (GPT-5.5) burn the `max_tokens` budget on hidden reasoning and return empty unless reasoning is disabled (`reasoning:{enabled:false}` → 0 reasoning tokens, clean YES/NO). **`google/gemini-3.1-pro-preview` was dropped** — it returns `400 "Reasoning is mandatory ... cannot be disabled"`, so it cannot run the naive condition; using it would conflate cross-vendor with reasoning-vs-naive. **DeepSeek-v3.2** (independent third lab, naive-capable, ~$0.23/$0.34 per Mtok) replaces it. Rule: `reasoning:{enabled:false}` for non-Anthropic, omit for Anthropic.

## 5. Analysis & pre-registered verdicts

Per **(monitor × construction)** and pooled-per-monitor: Youden **J** + Clopper-Pearson CI, **FPR**, empty/unparsed counts (monitor-DoS surfaced, never counted as "clean"). Contrasts: **paired McNemar** (same items) of each non-self monitor vs Opus-self; **Holm-Bonferroni** over the contrast family (3 non-self monitors × 4 constructions + 3 pooled = pre-registered).

| Outcome | Reading |
|---|---|
| self ≈ 0 (CI incl. 0) all constructions | replication of the self-null at n=150×4 ✓ |
| cross-vendor J ≫ self, Holm-sig | **RQ1.5b confirmed** — cross-vendor auditor beats self-monitoring |
| same-vendor (Sonnet) also ≈ 0 | **shared-lineage blind spot** (strongest result for the defense) |
| same-vendor catches it like cross-vendor | self-*model* effect, not lineage |
| all monitors ≈ 0 | the null is the *task*, not self-attribution — also a clean, honest result |

Each outcome maps to an honest claim; none is a study failure.

## 6. Build plan (TDD, single-thread per ADR-0009; small, reuses infra)

1. `parse_monitor_reply(raw)` extracted from `NaiveMonitorDetector` (behavior-preserving) + direct test.
2. `csd/monitor_cache.py` — `JudgmentRecord`, `append`/`load`/`existing_keys` keyed by `(model, record_id)` + tests.
3. `experiments/phase1_5_cross_monitor.py` — load corpus → cached panel judge → parse → per-(model×construction) J/CI + McNemar-vs-self + Holm → matrix + verdict. Mock E2E with a `MockMonitor` panel (deterministic per-model replies).

All: `ruff` + `bandit` clean, matching the green-suite discipline.

## 7. Sequencing & gates

1. Build + **mock E2E** (no API).
2. **Live smoke:** ~2 records × 4 monitors (~8 calls) — **validates per-model parseability** (a reasoning model may need `max_tokens` raised, or may emit preamble; the trichotomy will surface it). Inspect each model's raw replies by hand.
3. **Confirm + go-ahead** for the full panel run (~$3, 4 × ~1,197 calls, cached).
4. Full run → analyze → honest write-up + PICKUP update + commit.

## 8. Out of scope

Informed-monitor factorial (deferred — free follow-up via cache); surprisal/distributional detectors (Phase 2, done); a 2nd corpus/seed; Phase 3.

## 9. Cost & key

~$3 total, cached (re-runs free). Uses the current OpenRouter key; **rotate after** (still the outstanding loose end from the Phase-2 run).
