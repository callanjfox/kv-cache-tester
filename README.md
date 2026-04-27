# kv-cache-tester — agentic trace replay (minimized)

A trimmed-down branch of [kv-cache-tester](https://github.com/callanjfox/kv-cache-tester)
that keeps only the pieces needed for agentic trace replay from a Hugging Face dataset.
Other kv-cache-tester tools (cache-rate sweep, working-set sweep, single-prompt
tester, graph combiners, trace-from-Postgres builder) and their bundled ~1.5 GB of
sample traces have been removed.

## What's new (v0.1, 2026-04-27)

- **`--debug-trace`** now also records integer token IDs per request:
  `prompt_token_ids` (via `apply_chat_template`) and `completion_token_ids`
  (via streaming `logprobs.content`), alongside the full request/response and
  every streamed chunk dumped via `chunk.model_dump()`.
- **Per-model `delta` field abstraction** in `send_request()` — handles
  models that route reasoning tokens through different fields. Default is
  `(content, reasoning_content)` (DeepSeek-R1, Qwen reasoning, etc.); gpt-oss
  via vLLM's harmony parser uses `(content, reasoning)`. Add a model to
  `_MODEL_DELTA_FIELDS` to register a new mapping.
- **Reasoning text now included in `response_text`** so the assistant turn
  appended to conversation history reflects the full `usage.completion_tokens`
  the server reported. Required for replay fidelity with reasoning models when
  `ignore_eos=True` + `max_tokens=N` causes the model to spend the full budget
  on the analysis channel.
- **Input-token metric uses server-authoritative `usage.prompt_tokens`** from
  the streamed final chunk instead of the local `apply_chat_template` count.
  Fixes ISL on gpt-oss harmony where the local template renders the replayer's
  message structure as 2 tokens despite the server seeing ~95k.
- **Per-user content salt prefix** — every `UserSession` lazy-generates an
  8-token salt seeded by `user_id` and prepends it to `conversation[0]`,
  idempotently. Two in-flight users replaying the same trace_id (common with
  `--recycle`) no longer share KV-cache blocks for the canonical prefix.
- **5-second quiesce** between warmup completion and metrics-collector start
  so lingering server-side prefill from the warmup batch isn't captured in
  period 1.
- **Period summary updates**:
  - "Wait time (s)" replaced by "Inter-turn time (s)" sourced from
    `RequestMetrics.delay_expected` (the trace's per-request think_time),
    filtered to `success and delay_expected > 0`.
  - Header counts up: `Period 4 (3:01 / 10:00)` instead of `(... remaining)`.
  - Each line of the period block is now its own log record so the formatter
    stamps `[time] LEVEL` on every line.
- **Removed dead code**: ANSI `Colors` machinery and `--no-color` flag, the
  `MODEL_DEFAULTS` table (qwen3-coder-only, no current sweep matches), the
  request-rate-limiting path (`should_rate_limit_dispatch`, rate_limited
  state, exponential-backoff bookkeeping), the unused `users_added` /
  `rate_limit_events` / `admission_blocked_events` CSV columns, and
  `period_dispatch_delays` / `all_dispatch_delays` accumulators. Net –200 LOC.

## Contents

| File | Purpose |
|------|---------|
| `trace_replay_tester.py` | Replay real agentic coding traces at a fixed concurrency against any OpenAI-compatible endpoint. |
| `requirements.txt` | Python dependencies. |
| `docs/trace_replay_tester.md` | Full reference for `trace_replay_tester.py` flags and behavior. |

## Quick Start

```bash
pip install -r requirements.txt

python trace_replay_tester.py \
    --api-endpoint http://localhost:8000 \
    --hf-dataset semianalysisai/cc-traces-weka-042026 \
    --output-dir output \
    --start-users 8 --max-users 8 \
    --test-duration 1800 \
    --recycle \
    --seed 42
```

The replayer loads multi-turn agentic-coding traces from a Hugging Face dataset
(or a local directory via `--trace-directory`), streams them against the target
endpoint at a fixed concurrency, and writes per-request metrics plus summary
plots to `--output-dir`.

See `docs/trace_replay_tester.md` for the full flag reference.

## Relationship to the upstream branch

This branch is based on `agentx` with the following removed:
- `cache_rate_tester.py`, `working_set_tester.py`, `single_prompt_tester.py`
- `combine_graphs.py`, `regenerate_graphs.py`, `generate_index.py`, `kv_cache_size.py`
- `build_traces_from_neon.py` (Postgres → trace converter)
- `debug_request.py` (ad-hoc streaming-response debugger)
- `launcher/` (alternate vLLM server harness)
- `traces/`, `traces_neon/` (~1.5 GB of bundled sample traces)
- `docs/{cache_rate,working_set,single_prompt,utilities}.md`

`vocabulary.py` is kept because `trace_replay_tester.py` imports it lazily
inside `_ensure_user_text_pool()` when it synthesizes user-message content.

Anything needed to run `trace_replay_tester.py` against a Hugging Face dataset
stays.

## License

See [LICENSE](LICENSE).
