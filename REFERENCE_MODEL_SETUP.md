# Reference-Model Setup — local GGUF logprobs for the COVERT_SEMANTIC detector

The detector's instrument is a **reference language model** that returns **per-token log-probabilities** for a given text. From those we compute surprisal `s_t = -log2 P(token | context)`; high mean surprisal = "anomalously stilted" = the semantic-stego signature (design §4 Phase 1). This guide gets a logprob-capable local model running.

> **Scope:** this sets up the *reference* model (the detector). The *sender* (the frontier model that crafts stego) is a separate API call, deferred to live runs.

---

## 0. The one gotcha that wastes a day

**`ollama` does NOT expose per-token input logprobs.** Its API returns generated text, not the logprob of each token in a *given* string. It is therefore **unusable** as a surprisal instrument. Do not reach for it.

Use one of:
- **`llama-cpp-python`** (embedded binding) — what the `csd` package uses. Recommended.
- **`llama.cpp` `server`** (`/completion` with `"n_probs"`) — for an always-on Pi/Umbrel host.
- **HF `transformers`** (exact logits) — heavier install (torch ≈ 2 GB), simplest API, slowest on CPU. Fallback only.

---

## 1. Model choice

| Want | Pick | Why |
|---|---|---|
| Cleanest surprisal | **base** (pretrained) model, not `-Instruct` | instruct-tuning skews next-token probs |
| Small / Pi-friendly | Qwen2.5-1.5B (base) or Llama-3.2-1B (base) | ~1.5 GB at Q8, fast prefill |
| Headline runs | 3B–7B base on a workstation | the design's reference-size ablation (bigger P_ref → lower detection floor) |
| Independence from sender | any open-weight | frontier senders are Anthropic/OpenAI/Google — any local model is family-independent for free |

**Quantization:** use **Q8_0** (or **F16**) for the instrument — you read probabilities directly, so minimize quantization noise. Reserve Q4_K_M for the Pi's tightest RAM only. Treat precision as one axis of the size-ablation, not a fixed choice.

---

## 2. Path A — this Intel Mac (prototype now, no new hardware)

2-core Kaby Lake + AVX2 + 16 GB. Surprisal is prefill-only, so this is enough to build and validate the whole Phase-1 pipeline.

```bash
# from the project dir (created by Task 1; for now make it):
mkdir -p ~/Apps/"Fellow Program"/research/ai-security/covert-semantic
cd ~/Apps/"Fellow Program"/research/ai-security/covert-semantic

# isolated env via uv (bare python3 isn't on PATH here; uv is)
uv venv
uv pip install "llama-cpp-python>=0.2.80" "huggingface_hub[cli]"
# if the wheel build fails (needs a compiler): brew install cmake  &&  retry
```

Download a base GGUF at Q8 (verify the exact repo/filename on HF — GGUF publishers vary; bartowski is a reliable quantizer):

```bash
mkdir -p ~/models
uv run hf download Qwen/Qwen2.5-1.5B-Instruct-GGUF \
    qwen2.5-1.5b-instruct-q8_0.gguf --local-dir ~/models
# official Qwen repo = trusted provenance. Verify integrity before first load:
shasum -a 256 ~/models/qwen2.5-1.5b-instruct-q8_0.gguf
# expected (this file, 2026-06): d7efb072e7724d25048a4fda0a3e10b04bdef5d06b1403a1c93bd9f1240a63c8
# for the base-vs-instruct ablation, add a base GGUF later (instruct is fine to start).
```

Smoke-test logprobs end-to-end:

```bash
uv run python - <<'PY'
import math
import numpy as np
from llama_cpp import Llama
llm = Llama(model_path="/Users/fevra/models/qwen2.5-1.5b-instruct-q8_0.gguf",
            n_ctx=4096, logits_all=True, verbose=False)

def surprisals(text):
    # generation-free: one eval over exactly these tokens; read per-position logits.
    toks = llm.tokenize(text.encode("utf-8"), add_bos=True)
    llm.reset(); llm.eval(toks)
    out = []
    for i in range(len(toks) - 1):                  # scores[i] predicts toks[i+1]
        logits = np.array(llm.scores[i], dtype=np.float64)
        m = logits.max(); lse = m + math.log(float(np.exp(logits - m).sum()))
        out.append(-(logits[toks[i + 1]] - lse) / math.log(2))   # natural log -> bits
    return out

nat = surprisals("the cat sat on the warm mat")
stilt = surprisals("the chromatic feline reposed upon the thermal textile")
print("natural mean :", round(sum(nat)/len(nat), 3))   # ~7.1 bits/token
print("stilted mean :", round(sum(stilt)/len(stilt), 3))  # ~9.9 bits/token
assert sum(stilt)/len(stilt) > sum(nat)/len(nat), "stego should be more surprising"
print("OK — stego text is measurably more surprising (the detector's signal exists)")
PY
```

**Expected:** the stilted line has higher mean surprisal. That single assertion proves the instrument works.

---

## 3. Path B — Pi 5 as an always-on logprob server

For unattended harness runs, host the model on the Pi and call it over HTTP. Use **llama.cpp's server** (ollama still won't work).

```bash
# on the Pi 5 (Raspberry Pi OS 64-bit, 8 GB)
sudo apt update && sudo apt install -y cmake build-essential libcurl4-openssl-dev
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j4
# fetch a model (Q4_K_M for the Pi's RAM, or Q8 if it fits)
./build/bin/llama-server -hf bartowski/Qwen2.5-1.5B-GGUF:Q4_K_M --host 0.0.0.0 --port 8080
```

Get per-token logprobs of a text via the server (`n_probs` + `prompt`-only):

```bash
curl -s http://<pi-ip>:8080/completion -d '{
  "prompt":"the chromatic feline reposed upon the thermal textile",
  "n_predict":0, "n_probs":1, "post_sampling_probs":false
}' | python3 -c "import sys,json,math; d=json.load(sys.stdin); \
print([round(-math.log2(t['completion_probabilities'][0]['prob']),3) for t in d.get('completion_probabilities',[])][:8])"
```

(If `n_predict:0` returns no prompt probs on your build, use `"n_predict":1` and read the prompt-token probs from the response; server logprob fields vary by llama.cpp version — pin the version once it works.)

---

## 4. Path C — Umbrel

Umbrel is a host OS, not a model runtime. If it's an **x86 mini-PC** (NUC-class), treat it exactly like Path A/B running headless: install llama.cpp (or the `llama-cpp-python` server) as a service and call it over your LAN. If Umbrel is on a **Pi 4**, prefer the Pi 5 (Path B) — the Pi 4 is meaningfully slower. Umbrel's value here is *always-on*, not speed; the Mac is faster for interactive dev.

---

## 5. Wiring into `csd` (the verified extraction)

Use the **generation-free `scores` extraction**: tokenize, run one `eval` over exactly those tokens, read the per-position logits, log-softmax the realized next token. **Do NOT use `create_completion(echo=True)`** — with `max_tokens=0` it still *generates* a continuation and folds those tokens into `token_logprobs`, contaminating the measurement (caught in the smoke test: it reported 135 "tokens" for a 7-token sentence and ~22 bits/token, near-random).

```python
def surprisals(self, text: str) -> list[float]:
    import numpy as np
    toks = self._llm.tokenize(text.encode("utf-8"), add_bos=True)
    self._llm.reset(); self._llm.eval(toks)
    out = []
    for i in range(len(toks) - 1):                       # scores[i] predicts toks[i+1]
        logits = np.array(self._llm.scores[i], dtype=np.float64)
        m = logits.max()
        lse = m + math.log(float(np.exp(logits - m).sum()))
        out.append(-(logits[toks[i + 1]] - lse) / math.log(2))   # natural log -> bits
    return out
```

Requires `logits_all=True` in the constructor (so `scores` holds every position). Verified on Qwen2.5-1.5B-Q8: a 7-token sentence → 6 surprisals, mean ≈ 7.1 bits; the stilted variant ≈ 9.9 bits — the signal.

---

## 6. What "good" looks like (validation)

- Smoke test (§2) passes: stilted > natural mean surprisal.
- Phase-1 mock CLI runs offline (`python experiments/phase1_reproduce.py --mock`).
- Phase-1 **live** (deferred): naive-null cell J≈0 (CI includes 0); hardened perplexity-anomaly cell J within CI of **0.58**. If not → instrument wrong; stop (design §8).
- **Size/precision ablation:** rerun the live reproduction across {1B, 1.5B, 3B} × {Q4, Q8} and record J vs reference size — a publishable sub-finding (does a stronger P_ref push the detection floor down?).

---

## 7. Dependency + model-trust note (ADR-0001 / ADR-0006)

**Vet result (2026-06-26, OSV):** `huggingface_hub` — clean. `llama-cpp-python` — one CVE, **CVE-2024-34359 ("Llama Drama")**: RCE via Jinja2 **server-side template injection in a GGUF's chat-template metadata**. Fixed in **0.2.72**; the project pins `>=0.2.80`, so we're patched. Two defense-in-depth properties hold anyway:
- **We use the generation-free `tokenize`+`eval`+`scores` path (§5), not chat completion** — so the GGUF's `chat_template` is **never rendered**, and the SSTI path is not exercised at all.
- **Treat every GGUF as untrusted input** (this session's theme: model/corpus ingestion is an attack surface — ADR-0006). Download only from reputable publishers, and **verify the SHA256** against the HF model card before first load:
  `shasum -a 256 ~/models/Qwen2.5-1.5B-Q8_0.gguf`

Pin `llama-cpp-python>=0.2.72` as a hard floor in `pyproject.toml`. Models live in `~/models` (outside the repo); **never commit weights**.
