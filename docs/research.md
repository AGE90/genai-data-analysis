# Research log

What was tried, on what data, with what result. Append a dated entry for every experiment. Keep numbers here and methods in notebooks.

## Radar

| Technique / model | Why interesting | Status |
|---|---|---|
| Structured outputs (pydantic-ai, any provider) | typed results, no JSON parsing | in use |
| TypeSafe **Jev** (System One, non-autoregressive, calibrated probs) | 100× cheaper/faster claims, confidence routing | notebook 05, needs `TYPESAFE_API_KEY` |
| Confidence routing (cheap model → LLM on low confidence) | accuracy per dollar | notebook 05 |
| Synthetic labeled data (LLM-generated chats) | eval without private data | in use; watch generator/judge bias |
| Embeddings + clustering / topic discovery | unsupervised structure in chats | backlog |
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
