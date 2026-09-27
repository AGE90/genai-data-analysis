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
| LLM-as-judge + human spot checks, prompt optimization (DSPy-style) | improve prompts against eval set | backlog |
| Distillation: small classifier trained on LLM/Jev labels or Jev probabilities | cost at scale | backlog |

## Log

### 2026-09-27 · Pipeline smoke test (not a benchmark)
- 17/20 Spanish tweets (`tweet_sentiment_multilingual`), `google:gemini-flash-lite-latest` → `gemini-3.5-flash-lite`: accuracy 0.59, macro-F1 0.58, κ 0.43, p50 1.05 s, $0.09 / 1k. The other 3 failed on the free-tier quota before `rpm` pacing was added. n is too small to conclude anything.
- Free tier: 15 req/min per model. Use `rpm=12`.

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
