# Inference Server Cache Performance Testing Suite

A toolkit for testing and benchmarking LLM inference servers and their KV Cache offload performance.

## Quick Start

```bash
# Install dependencies (using uv - recommended)
uv pip install -r requirements.txt

# Or with pip
pip install -r requirements.txt

# Run a simple single-prompt test
uv run python single_prompt_tester.py \
    --api-endpoint http://localhost:8000 \
    --min-tokens 1000 \
    --max-tokens 128000 \
    --output-dir results

# Test various cache hit rates (sustained mode - default)
uv run python cache_rate_tester.py \
    --api-endpoint http://localhost:8000 \
    --context-sizes 8000 32000 64000 \
    --working-set-size 2000000 \
    --max-ttft 2.0 \
    --output-dir output

# Test performance across different memory tiers
uv run python working_set_tester.py \
    --api-endpoint http://localhost:8000 \
    --context-sizes 30000 \
    --min-working-set-size 100000 \
    --max-working-set-size 5000000 \
    --working-set-increments 10
```

## Core Testing Tools

| Tool | Purpose | Documentation |
|------|---------|---------------|
| **single_prompt_tester.py** | Baseline cache performance with cold start vs cached comparisons | [Docs](docs/single_prompt_tester.md) |
| **cache_rate_tester.py** | Performance at various cache hit rates (0-100%) with fixed working set | [Docs](docs/cache_rate_tester.md) |
| **working_set_tester.py** | Performance across varying working set sizes for memory tier analysis | [Docs](docs/working_set_tester.md) |
| **trace_replay_tester.py** | Replay real agentic coding traces with realistic cache patterns and timing (beta) | [Docs](docs/trace_replay_tester.md) |

### Included Traces

The `traces/` directory contains real-world agentic coding traces for use with `trace_replay_tester.py`. This is a curated subset of a larger dataset: **642 coding conversations with embedded subagent usage**, totaling **112,176 requests** and **13,675,350,647 tokens**. The traces include subagent spawns just as a real user session would — the tool replays these as part of the conversation, capturing the realistic concurrency and cache pressure patterns they create.

For privacy, conversation IDs have been anonymized and cache block hashes do not align across conversations. The trace replay tool accounts for this by generating a configurable shared prefix (`--warm-prefix-pct`) that simulates cross-conversation cache overlap from common tool definitions and system prompts — the default is calibrated to match patterns observed in real production data. We may release additional traces depending on community interest.

| Metric | Min | P25 | Median | P75 | Max | Mean |
|---|---|---|---|---|---|---|
| Starting input tokens | 8,832 | 16,512 | 20,160 | 71,168 | 639,552 | 61,529 |
| Ending input tokens | 10,304 | 65,600 | 115,008 | 149,632 | 785,280 | 133,471 |
| Cache hit rate (per conv) | 38.3% | 91.1% | 96.9% | 98.5% | 99.6% | 93.6% |
| Conversation duration (min) | 0.2 | 11.9 | 60.2 | 163.0 | 11,105.6 | 283.9 |

## Utility Scripts

| Tool | Purpose |
|------|---------|
| **generate_index.py** | Generate HTML dashboard from test results |
| **regenerate_graphs.py** | Regenerate graphs from existing CSV data |
| **combine_graphs.py** | Combine results from multiple runs for comparison |
| **kv_cache_size.py** | Calculate KV cache memory requirements |

See [Utilities Documentation](docs/utilities.md) for details.

## Installation

We recommend using [uv](https://docs.astral.sh/uv/) for fast, reliable Python package management.

```bash
# Clone the repository
git clone https://github.com/callanjfox/kv-cache-tester.git
cd kv-cache-tester

# Install dependencies with uv (recommended)
uv pip install -r requirements.txt

# Or with pip
pip install -r requirements.txt

# Verify installation
uv run python single_prompt_tester.py --help
```

## Understanding the Metrics

- **TTFT (Time To First Token):** Latency before first token generation - primary metric for cache effectiveness
- **TTLT (Time To Last Token):** Total request completion time
- **ITL (Inter-Token Latency):** Time between generated tokens
- **Input/Output Throughput:** Tokens processed/generated per second

## Testing Methodology

For a detailed guide on testing methodology and how to use these tools effectively, see:
[Evaluating Management of KV Cache within an Inference System](https://medium.com/@callan.j.fox/evaluating-management-of-kv-cache-within-an-inference-system-2d7c3d266c3a)

## Real-World Trace Replay

> **Beta:** `trace_replay_tester.py` is currently in beta with active development underway. Updates to come.

The `trace_replay_tester.py` tool replays real agentic coding traces to benchmark inference performance with realistic cache hit patterns, timing, and message structures. See the [Trace Replay Documentation](docs/trace_replay_tester.md) for full details.
