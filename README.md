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

# Replay real agentic coding traces
uv run python trace_replay_tester.py \
    --api-endpoint http://localhost:8000 \
    --trace-directory traces \
    --output-dir output \
    --start-users 5 \
    --max-users 50 \
    --max-ttft 2.0 \
    --test-duration 300
```

## Core Testing Tools

| Tool | Purpose | Documentation |
|------|---------|---------------|
| **single_prompt_tester.py** | Baseline cache performance with cold start vs cached comparisons | [Docs](docs/single_prompt_tester.md) |
| **cache_rate_tester.py** | Performance at various cache hit rates (0-100%) with fixed working set | [Docs](docs/cache_rate_tester.md) |
| **working_set_tester.py** | Performance across varying working set sizes for memory tier analysis | [Docs](docs/working_set_tester.md) |
| **trace_replay_tester.py** | Replay real agentic coding traces with realistic cache patterns and timing (beta) | [Docs](docs/trace_replay_tester.md) |

### Included Traces

The `traces/` directory contains **522 anonymized agentic coding traces** with **100,186 requests**, including 23 traces with nested sub-agent conversations (80 sub-agents total). Generated from real Claude Code sessions captured via [claude-code-proxy](https://github.com/seifghazi/claude-code-proxy) as part of research at [WEKA](https://www.weka.io/) on the [Augmented Memory Grid](https://www.weka.io/resources/solution-brief/weka-augmented-memory-grid/) product. See [agentic-coding-analysis](https://github.com/callanjfox/agentic-coding-analysis) for the trace generation tools.

Traces include:
- **Global hash_ids** (`hash_id_scope: "global"`) — consistent across all conversations and sub-agents for cross-context cache simulation
- **Timing breakdown** — `api_time` (server processing) and `think_time` (client delay) per request, enabling flexible replay timing strategies
- **Sub-agent nesting** — sub-agents embedded in parent traces with their own tool/system tokens and request arrays

All traces are fully anonymized — no conversation IDs, timestamps, or real agent IDs.

| Metric | Min | P25 | Median | P75 | Max | Mean |
|---|---|---|---|---|---|---|
| Starting input tokens | 6,697 | 17,360 | 19,588 | 56,364 | 509,308 | 42,174 |
| Ending input tokens | 6,344 | 76,041 | 122,625 | 148,552 | 636,522 | 130,392 |
| Cache hit rate (per conv) | 25% | 94% | 98% | 99% | 100% | 92% |
| Conversation duration (min) | 0 | 33 | 71 | 161 | 44,073 | 658 |

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

> **Note:** This guide will be updated soon with the latest methodology covering trace replay testing and multi-endpoint support.

## Real-World Trace Replay

> **Beta:** `trace_replay_tester.py` is currently in beta with active development underway. Updates to come.

The `trace_replay_tester.py` tool replays real agentic coding traces to benchmark inference performance with realistic cache hit patterns, timing, and message structures. See the [Trace Replay Documentation](docs/trace_replay_tester.md) for full details.
