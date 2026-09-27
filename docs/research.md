# Research log

What was tried, on what data, with what result. Append a dated entry for every experiment. Keep numbers here and methods in notebooks.

## Radar

| Technique / model | Why interesting | Status |
|---|---|---|
| Structured outputs (pydantic-ai, any provider) | typed results, no JSON parsing | in use |
| TypeSafe **Jev** (System One, non-autoregressive, calibrated probs) | 100× cheaper/faster claims, confidence routing | notebook 05, needs `TYPESAFE_API_KEY` |
| Confidence routing (cheap model → LLM on low confidence) | accuracy per dollar | notebook 05 |
| Synthetic labeled data (LLM-generated chats) | eval without private data | in use; watch generator/judge bias |
| Embeddings + clustering / topic discovery | unsupervised structure in chats | notebook 06 |
| Audio: Whisper-class local vs Gemini audio / `gemini-*-transcribe` | voice-note-heavy chats | backlog |
| LLM-as-judge + human spot checks | grade free-text outputs; adjudicate noisy gold | notebook 07 |
| Prompt optimization (DSPy-style) against the eval set | better prompts by search, judged by eval/judge | backlog |
| Distillation: small classifier trained on LLM/Jev labels or Jev probabilities | cost at scale | backlog |

## Log

### 2026-09-27 · Pipeline smoke test (not a benchmark)
- 17/20 Spanish tweets (`tweet_sentiment_multilingual`), `google:gemini-flash-lite-latest` → `gemini-3.5-flash-lite`: accuracy 0.59, macro-F1 0.58, κ 0.43, p50 1.05 s, $0.09 / 1k. The other 3 failed on the free-tier quota before `rpm` pacing was added. n is too small to conclude anything.
- Free tier: 15 req/min and 500 req/day per model (`gemini-3.5-flash-lite`); quotas are per model, so `gemini-3.1-flash-lite` still works when 3.5 is exhausted. Use `rpm=12`.

### 2026-09-27 · Notebook 04 dry run (n=10 tweets/lang, 9 chats requested)
- `gemini-flash-lite-latest` (→ gemini-3.5-flash-lite), tweets: es acc 0.70 / F1 0.65 · en acc 0.90 / F1 0.92, ~$0.09 / 1k. Misses look like sarcasm or label noise (e.g. "muchas gracias bonita ❤" gold=neutral).
- Synthetic: only 4/9 chats generated (`gemini-flash-latest` free tier = 20 req/day). Flash-Lite got 4/4 initial and final arcs right on those. Too few to mean anything.
- Next: the full run (100 tweets/lang, 45 chats) needs a paid Gemini tier or an Anthropic key; then run notebook 05 once `TYPESAFE_API_KEY` is available.

### 2026-09-27 · Notebook 03 on the synthetic WhatsApp fixture
- Parse → anonymize → `SENTIMENT_ARC` with `gemini-flash-lite-latest` ran end to end on 2 sessions and 7 messages. Both arcs were plausible (negative → neutral, neutral → positive) with Spanish justifications, and redaction kept amounts like $50.000. Needs a real export to evaluate.

### 2026-09-27 · Notebook 04 full run (free tier, Flash-Lite only)
Model `google:gemini-flash-lite-latest` → gemini-3.5-flash-lite, temperature 0. Claude and `gemini-flash-latest` were not run (no Anthropic key; flash free tier = 20 req/day).

| Data | n | coverage | accuracy | macro-F1 | κ | p50 latency | $/1k |
|---|---|---|---|---|---|---|---|
| Tweets es (`tweet_sentiment_multilingual` test) | 100 | 0.99 | 0.69 | 0.67 | 0.53 | 0.74 s | 0.09 |
| Tweets en | 100 | 1.00 | 0.72 | 0.71 | 0.58 | 0.69 s | 0.09 |
| Synthetic es chats, initial sentiment | 45 | 1.00 | 0.82 | 0.81 | 0.73 | 0.97 s | 0.39 |
| Synthetic es chats, final sentiment | 45 | 1.00 | 0.73 | 0.68 | 0.60 | 0.97 s | 0.39 |

- The one missing tweet was a Gemini `PROHIBITED_CONTENT` block, recorded as an error row as designed.
- Tweet misses look like sarcasm and label ambiguity (e.g. "muchas gracias bonita ❤" gold=neutral).
- **Arc errors all sit on the neutral boundary.** Gold-neutral finals were predicted positive 12/15 times; gold-positive initials were predicted neutral 7/15. Reading them, polite closings ("¡qué bien! Gracias 👍") read as positive although the generator was asked for neutral. This is label ambiguity in the synthetic set more than model error. Negative was never confused (15/15 both ends).
- Caveat: generator and judge are the same model family here. Next: generate with Claude, judge with Gemini (and vice versa); tighten the neutral definition in the generator prompt and/or the task instructions.

### 2026-09-27 · Notebook 06 topic discovery
- Synthetic chats (45, 8 domains), customer messages only, KMeans with k by silhouette:

| Embedder | chosen k | NMI | ARI | NMI @ true k=8 |
|---|---|---|---|---|
| `gemini-embedding-001` | 9 | **0.97** | **0.94** | 0.92 |
| `gemini-embedding-2` | 9 | 0.89 | 0.78 | 0.90 |

- The only "error" with embedding-001: banking split into two sensible subtopics (unrecognized charges vs card activation/use). The LLM names matched the domains (e.g. "Recargas no reflejadas", "Cambio de fecha de vuelo").
- Tweets (100 es): 4 clusters that follow genre/tone (social replies, thanks and affection, moods, daily life), not topics, as expected for random tweets.
- Without `language=`, the namer mixed Spanish and English names across clusters, so `label_clusters(language=...)` now pins it.
- Synthetic chats are easy for clustering (one clear domain each); expect lower scores on real multi-topic chats.

### 2026-09-27 · Notebook 07 LLM-as-judge (judge = predictor = `gemini-flash-lite-latest`, so self-preference is possible)
Rubric judge (`ARC_JUDGE`) on 45 sentiment-arc outputs × 3 variants, 0 errors:

| Variant | initial supported | final supported | faithful | mean score |
|---|---|---|---|---|
| original | 0.98 | 1.00 | 1.00 | 4.98 |
| swapped justification (from another domain) | 0.71 | 0.78 | **0.00** | 1.42 |
| flipped final label | 1.00 | **0.00** | 0.00 | 1.96 |

- It caught 100% of both corruptions with 0 false alarms on originals, but these corruptions are **too easy** (cross-domain swap, polarity flip). Next: harder corruptions (same-domain swaps, neutral↔positive shifts, one invented detail).
- The 1–5 `score` hits the ceiling (4.98 on originals), so it carries little information. The bool checks are the useful signal.
- On the 12 originals where the prediction disagreed with synthetic gold, the judge said the model's final label was supported **12/12**. That fits the gold-noise finding from notebook 04, but a same-family judge can't separate that from leniency. Re-run with a Claude judge.
- Pairwise adjudication is **incomplete**: the Flash-Lite free tier is 500 requests/day and ran out mid-section. The few completed pairs show strong **position bias**: option A was picked only 7% of the time and order consistency was 0.75. That is exactly why every pair is asked in both orders. Re-run tomorrow; cached calls are skipped.
- Cost: rubric judging was $0.70/1k vs $0.39/1k for the predictions it grades. Judge a sample, not everything.
- The blind human spot-check sheet was written to `data/processed/judge_spotcheck.csv` (not labeled yet).

### 2026-09-27 · Notebook 08 pension-fund triage (synthetic, LLM baseline only)
60 synthetic affiliate messages (10 intents × 6, generated and classified by `google:gemini-3.1-flash-lite` because the 3.5 Flash-Lite daily quota was exhausted), one `Triage` task:

| Field | accuracy | macro-F1 | κ |
|---|---|---|---|
| intent (10 processes) | 0.87 | 0.83 | 0.85 |
| urgency (low/medium/high) | **0.52** | 0.40 | 0.23 |
| complaint_risk | 0.88 | 0.85 | 0.71 |
| wants_human | 0.77 | 0.74 | 0.51 |
| vulnerable_customer | 0.87 | 0.85 | 0.71 |
| transfer_risk | 0.90 | 0.89 | 0.78 |

p50 latency 1.1 s, $0.20 / 1k messages (≈ $204 per million).
- Intent routing is already strong. **Urgency is the weak field**: the model rates almost everything "high" (the queue view is dominated by high), and the generator also writes urgent-sounding text for "medium". Urgency needs a sharper rubric (concrete examples per level, a deadline/money-at-risk test) and real labeled data, and it is the best candidate for Jev's `Score` primitive (ordinal) plus calibrated routing.
- The synthetic generator mentioned a real competitor AFP by name in one message. For demos shown to leaders, add "don't name real companies" to the generator prompt or post-filter.
- Jev not run (no `TYPESAFE_API_KEY`). With a key, the notebook produces the Jev vs LLM table, ECE and the routing curve.
