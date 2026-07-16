# What We Found — In Plain Words

A plain-language summary of the covert-semantic research. Written to be read, not decoded.
(For the numbers and rigor, see the dated design/results memos alongside this file.)

---

## The question

**Can an AI hide a small secret inside normal-looking text — and can another AI catch it?**

The secret is tiny and structured: a colour, a number, a shape. "Hidden" means woven into
ordinary prose so a casual reader notices nothing.

By the end we'd split that into **two** questions, because they turned out to have different
answers:

1. Can a monitor **see** the secret (if it's looking at the text)?
2. Does the secret even **reach** the monitor as text it can read — or is it smuggled past in
   some other form?

---

## The words you need

- **Steganography** — hiding a message inside something ordinary, so nobody knows a message is
  even there. (Encryption scrambles a message everyone can *see* is secret; steganography hides
  the fact that there's a message at all.)
- **Monitor** — an AI given one job: read something, answer one question — *"Is anything hidden
  here: yes or no?"*
- **Self-monitoring** — the *same* AI that wrote the text is the one checking it. Like
  proofreading your own writing.
- **False positive** — crying wolf: calling innocent text "hidden." A monitor that says *yes* to
  everything catches every secret but is worthless.
- **Detection score (J)** — one number, 0 to 1. **0 = guessing. 1 = perfect.** It rewards
  catching real secrets and subtracts for crying wolf. We use it throughout.
- **Modality** — the *form* a message arrives in: as text, or as a **picture of text** (a
  screenshot). **OCR** is the act of reading the words back out of a picture.

---

## The big picture (the one idea that ties it all together)

A monitor can fail in **two completely separate ways**:

1. **It doesn't know to look.** Asked cold, it shrugs and says "nothing here."
2. **The message never reaches it as text.** It's hidden inside a picture, or changed form
   somewhere along the way, so the monitor is inspecting the wrong thing.

Think of it as two dials: **what the monitor knows**, and **where the monitor sits** (what form
the message is in when it arrives). Our four findings map both dials. That's what makes the work
feel whole rather than a pile of separate results.

---

## The findings

### 1. A trained detector learns a *style* of hiding — not hiding in general.

We built a detector that studies the statistical fingerprint of hidden text. It works well on
styles of hiding it has *seen before* — but drops sharply on a *new* style it hasn't.

So it's really a **style-recognizer, not a universal secret-detector.** We named it honestly: a
"construction probe." Useful, but narrower than it first looked.

### 2. Asked cold, an AI is blind to hidden messages — even its own.

We showed AIs the hidden text and asked plainly, *"Anything hidden here?"*

- Most models — **including the very model that wrote the text** — said *no* to essentially
  everything. Blind.
- **Only OpenAI's models spontaneously caught the secrets.**
- A strong model from a different maker (xAI's Grok) was *just as blind* — so this isn't about
  being "smarter." It was specific to OpenAI's models. (We checked this carefully: a *weaker*
  OpenAI model still beat a *stronger* non-OpenAI one, so it tracks the model family, not raw
  capability.)

### 3. Tell the AI *what to look for*, and the blindness mostly vanishes.

We changed one thing. Instead of *"anything hidden?"*, we said: *"this text may hide a
colour/number/shape using an acrostic, or synonyms, or a word-count pattern…"* — naming the
**kinds** of hiding, but never the secret itself.

The blind models **woke up:**

| model | asked cold | told what to look for |
|---|---|---|
| the model that wrote it | 0.00 (blind) | **0.44** |
| a different-maker model (DeepSeek) | 0.00 (blind) | **0.72** |
| OpenAI's best | 0.53 | **0.77** |

So the earlier "blindness" was mostly this: **nobody had told them what to look for.** The OpenAI
advantage was mostly that OpenAI's models *check on their own.*

**Two honest wrinkles:**

- **One model started crying wolf.** Grok's score jumped too — but it flagged ~40% of *innocent*
  text as well. That's alarm, not skill. Telling a model what to look for can make it **paranoid,
  not sharp.**
- **Some hiding-styles stay hard.** "First letter of each sentence" and "count the words in each
  sentence" need mechanical checking most models can't do at a glance — except a couple that
  happen to be good at exactly that. Skill here is **specific to the model.**

### 4. Hide the message in a *picture*, and even a warned AI misses it — and not because it can't read it.

Now the second dial. We took the same hidden-message texts, turned them into **pictures of the
text** (like a screenshot), and asked the *warned* monitors — looking at the picture — *"anything
hidden?"*

It was **not** a clean "it all falls apart." It depended on the **kind** of hiding:

| how the secret is hidden | as text | as a picture |
|---|---|---|
| by overall *feel* (text that reads oddly stilted) | caught | **still caught** |
| by specific *marker words* (planted synonyms) | caught | **missed** (for some models) |

- Hiding by **gist** survives the picture — the "this reads like it's hiding something" feeling
  comes through either way.
- Hiding by **exact words** slips through — the monitor stops catching it once it's a picture.

**The twist that makes this the strongest result:** we checked whether the monitor could even
*read* the words in the picture. It could — **perfectly.** It transcribed "azure, calm, quiet"
correctly and *still* said "nothing hidden." So it isn't that the monitor couldn't see the words.
It's that **when the words arrive as a picture, it doesn't think about them as carefully.**
Reading is not the same as analysing. Changing the *form* of the message quietly switched off the
careful checking — even in a monitor we'd told exactly what to look for.

---

### 5. We tried the obvious fix — "look harder at the words" — and it doesn't cleanly work.

The natural next question: what if you just *tell* the picture-reading monitor to slow down and
scrutinise every word, as carefully as if it were plain text? We tried exactly that.

It doesn't add care — it **moves** it. The monitors have a limited budget of careful attention,
and pointing it at the words robs the "overall feel" channel:

- One monitor (Claude Sonnet) that had gone soft on marker-word hiding got **much better at it**
  when told to scrutinise the words — but at the same time went **completely blind** to the *gist*
  kind of hiding it had been catching. It found what we pointed it at and stopped noticing
  everything else.
- Another (GPT-5.5) just got **worse across the board** — over-focusing hurt it.
- Only one (Opus) could do both at once.

So there's no free "try harder" switch. Telling the monitor to look harder at one thing turns off
its guard somewhere else — differently for each model. **The blind spot doesn't close; it
relocates.** Which is exactly why the fix has to be structural, not a better pep talk: turn the
picture back into words, and check the words.

---

## What it adds up to

- A naive *"is anything hidden?"* monitor is the **wrong tool** — it misses almost everything.
- **Telling the monitor what to look for** recovers most of the detection, cheaply — *dial 1*.
- But that only helps if the message reaches the monitor **as text**. Hide it in a picture and a
  warned monitor goes soft again — *dial 2* — even though it can read the picture fine.
- So the recipe is: **tell the monitor what to look for, AND inspect the message as text** (turn
  any picture back into words first, then check the words) — **and** watch that it isn't just
  crying wolf.
- And you **can't** rescue the picture case just by telling the monitor to look harder — that only
  *moves* its blind spot, differently for each model. The fix has to be structural.
- Two of our results pushed back on things people believed: (a) a well-known paper said "telling
  the monitor what to look for barely helps" — here it helped *a lot*; and (b) a monitor that
  reads a picture perfectly can *still* fail to reason about it.

---

## Still open (the interesting angles)

1. **Why do OpenAI's models check on their own** when others don't? Training habit? Something in
   how they're tuned? We don't know yet — the most interesting thread.
2. **Can we make the paranoid model sharp** — by asking it for a *confidence level* instead of a
   flat yes/no, then setting our own bar?
3. **The hardest hiding-styles** (the mechanical, structural ones) — what actually lets a model
   catch them?
4. **The picture blindness — partly answered (2026-07-16).** We asked whether telling the monitor
   "this is a picture, analyse the words as carefully as you would text" fixes it. It doesn't
   cleanly: it *reallocates* the monitor's limited attention (rescues word-level hiding, loses
   gist-level), differently for each model — so the useful result is the opposite of the hoped-for
   one, and the defense has to be structural. Still open: does giving the monitor room to *reason*
   (not just a one-word snap answer) *expand* the budget instead of just moving it?

---

*One line to remember: the machines aren't blind — but they only look carefully when you **tell
them to**, and only when the message arrives as **words**. Change the form, and even a warned
monitor stops thinking.*
