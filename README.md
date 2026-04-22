# kv-cache-tester — agentic trace replay (minimized)

A trimmed-down branch of [kv-cache-tester](https://github.com/callanjfox/kv-cache-tester)
that keeps only the pieces needed for agentic trace replay from a Hugging Face dataset.
Other kv-cache-tester tools (cache-rate sweep, working-set sweep, single-prompt
tester, graph combiners, trace-from-Postgres builder) and their bundled ~1.5 GB of
sample traces have been removed.

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
