# Trace Replay Tester

> **Beta:** This tool is currently in beta with active development underway. Updates to come.

## Overview

`trace_replay_tester.py` replays real agentic coding traces to benchmark LLM inference performance with realistic cache hit patterns, timing, and message structures.

## Features

| Feature | Description |
|---------|-------------|
| **Fire-and-forget async** | Non-blocking request dispatch; users fire immediately when ready |
| **Proportional output attribution** | Output tokens attributed to period when generated (not when request completes) |
| **Timestamp-based metrics** | Input tokens at TTFT, output tokens by chunk timestamp |
| **Adaptive user scaling** | Adds users based on TTFT headroom below threshold |
| **Cache pressure budgeting** | `--max-new-tokens-per-period` limits cache churn |
| **Working set limits** | `--max-working-set-tokens` caps total unique tokens across all users |
| **Trace recycling** | `--recycle` replaces completed users with fresh traces |
| **Deterministic seeds** | `--seed` for reproducible trace selection |
| **Request pairs** | Handles streaming + non-streaming pairs with same hash_ids |
| **Admission control** | `--max-concurrent-requests` limits in-flight requests |
| **Warm prefix caching** | `--warm-prefix-pct` enables cross-conversation cache sharing |

## Included Traces

The `traces/` directory contains real-world agentic coding traces for use with this tool. This is a curated subset of a larger dataset: **642 coding conversations with subagents**, totaling **112,176 requests** and **13,675,350,647 tokens**.

| Metric | Min | P25 | Median | P75 | Max | Mean |
|---|---|---|---|---|---|---|
| Starting input tokens | 8,832 | 16,512 | 20,160 | 71,168 | 639,552 | 61,529 |
| Ending input tokens | 10,304 | 65,600 | 115,008 | 149,632 | 785,280 | 133,471 |
| Cache hit rate (per conv) | 38.3% | 91.1% | 96.9% | 98.5% | 99.6% | 93.6% |
| Conversation duration (min) | 0.2 | 11.9 | 60.2 | 163.0 | 11,105.6 | 283.9 |

## Quick Start

```bash
python trace_replay_tester.py \
    --api-endpoint http://localhost:8000 \
    --trace-directory ../traces \
    --output-dir ./results \
    --start-users 5 \
    --max-users 50 \
    --max-ttft 2.0 \
    --test-duration 300
```

## Command-Line Options

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--api-endpoint` | API server endpoint (e.g., `http://localhost:8000`) |
| `--trace-directory` | Directory containing trace JSON files |
| `--output-dir` | Output directory for results |

### Performance Thresholds

| Argument | Default | Description |
|----------|---------|-------------|
| `--max-ttft` | 2.0 | Maximum acceptable TTFT in seconds |
| `--ttft-metric` | p95 | TTFT metric: `max`, `avg`, or `p95` |
| `--min-output-tokens-per-req` | - | Minimum output tokens/s per request |

### User Management

| Argument | Default | Description |
|----------|---------|-------------|
| `--start-users` | 1 | Initial number of users |
| `--max-users` | 50 | Maximum concurrent users |
| `--recycle` | false | Replace completed users with new traces |
| `--max-new-tokens-per-period` | 500000 | Max cache miss tokens per assessment period |
| `--max-working-set-tokens` | 0 | Max total tokens across all users (0 = unlimited) |

### Trace Filtering

| Argument | Default | Description |
|----------|---------|-------------|
| `--min-requests` | 1 | Minimum requests per trace to include |
| `--max-context` | 128000 | Maximum input tokens per request |

### Timing

| Argument | Default | Description |
|----------|---------|-------------|
| `--max-delay` | 60.0 | Maximum delay between requests (seconds) |
| `--time-scale` | 1.0 | Time scaling factor (0.5 = 2x faster) |
| `--test-duration` | - | Maximum test duration (seconds) |
| `--assessment-period` | 30 | Assessment period duration (seconds) |

### Admission Control

| Argument | Default | Description |
|----------|---------|-------------|
| `--max-concurrent-requests` | 50 | Max in-flight requests (0 = unlimited) |
| `--enable-request-rate-limiting` | false | Rate-limit dispatch when TTFT exceeds threshold |

### Cross-Conversation Cache Sharing

| Argument | Default | Description |
|----------|---------|-------------|
| `--warm-prefix-pct` | 0.5 | Fraction of tool+system tokens to pre-warm (0.0-1.0) |

The warm prefix feature simulates how Claude Code's tool definitions and system prompt are typically already cached across conversations:

- Calculates `max(tool_tokens + system_tokens)` across all traces
- Generates a **canonical shared prefix** used by ALL users
- User 1's first request: all cache misses on shared prefix
- User 2+'s first request: cache hits on the warm prefix portion
- Set to `0` to disable

### Generation Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--temperature` | - | Override temperature for generation |
| `--top-p` | - | Override top_p for generation |
| `--top-k` | - | Override top_k for generation |
| `--repetition-penalty` | - | Override repetition_penalty |

### Other Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--tokenizer` | Qwen/Qwen2.5-Coder-32B-Instruct | Tokenizer for synthetic data |
| `--chunk-size` | 256 | Cache block size in tokens |
| `--seed` | - | Random seed for reproducible trace selection |
| `--verbose` | false | Enable verbose logging |
| `--skip-graphs` | false | Skip graph generation |

## Metric Attribution

| Metric | Attributed When |
|--------|-----------------|
| Input tokens | Prefill completes (TTFT time) |
| Output tokens | Proportionally when tokens are generated (streaming) |
| TTFT stats | Prefill completes in period |
| Cache hit/miss | Prefill completes in period |

## Output Files

| File | Description |
|------|-------------|
| `summary_trace_replay.csv` | Per-assessment-period metrics |
| `detailed_results.csv` | Per-request metrics |
| `user_lifecycle.csv` | User start/complete/truncate events |
| `test_metadata.json` | Configuration and trace stats |
| `progress.json` | Resume state |
| `*.html` | Plotly visualizations |
| `index.html` | Dashboard (via generate_index.py) |

## Trace Format

Traces should be JSON files with this structure:

```json
{
  "id": "f64a8d4c-12d",
  "models": ["claude-sonnet-4-20250514"],
  "block_size": 64,
  "tool_tokens": 11899,
  "system_tokens": 2926,
  "requests": [
    {
      "t": 0.0,
      "type": "s",
      "model": "claude-sonnet-4-20250514",
      "in": 25259,
      "out": 86,
      "hash_ids": [1, 2, 3, ...],
      "input_types": ["text"],
      "output_types": ["text"],
      "stop": ""
    }
  ]
}
```

### Request Fields

| Field | Type | Description |
|-------|------|-------------|
| `t` | float | Timestamp in seconds from conversation start |
| `type` | string | `"s"` = streaming, `"n"` = non-streaming |
| `in` | int | Input token count |
| `out` | int | Output token count |
| `hash_ids` | int[] | Block hash IDs for cache simulation |
| `input_types` | string[] | New content types: `["text"]`, `["tool_result"]` |
| `output_types` | string[] | Response content types |
| `stop` | string | Stop reason: `""`, `"tool_use"`, `"end_turn"` |

### Top-Level Fields

| Field | Type | Description |
|-------|------|-------------|
| `tool_tokens` | int | Tokens in tool definitions (~8-12K for Claude Code) |
| `system_tokens` | int | Tokens in system prompt (~2-3K for Claude Code) |
| `block_size` | int | Token block size (typically 64) |

## Examples

### Basic Test

```bash
python trace_replay_tester.py \
    --api-endpoint http://localhost:8000 \
    --trace-directory ../traces \
    --output-dir ./results \
    --test-duration 300
```

### Production Capacity Test

```bash
python trace_replay_tester.py \
    --api-endpoint http://localhost:8000 \
    --trace-directory ../traces \
    --output-dir ./capacity_test \
    --start-users 10 \
    --max-users 100 \
    --max-ttft 2.0 \
    --ttft-metric p95 \
    --max-new-tokens-per-period 1000000 \
    --test-duration 1800 \
    --recycle \
    --seed 42
```

### Test with Warm Prefix (Cross-Conversation Caching)

```bash
# Default 50% warm prefix
python trace_replay_tester.py \
    --api-endpoint http://localhost:8000 \
    --trace-directory new_traces/full_64_split1day_normalized \
    --output-dir ./warm_test \
    --start-users 5 \
    --warm-prefix-pct 0.5

# Disable warm prefix
python trace_replay_tester.py \
    --api-endpoint http://localhost:8000 \
    --trace-directory new_traces/full_64_split1day_normalized \
    --output-dir ./no_warm_test \
    --warm-prefix-pct 0
```

### Strict Admission Control

```bash
python trace_replay_tester.py \
    --api-endpoint http://localhost:8000 \
    --trace-directory ../traces \
    --output-dir ./admission_test \
    --max-concurrent-requests 20 \
    --max-users 50 \
    --test-duration 600
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        trace_replay_tester.py                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ TraceManager │  │ UserSession  │  │ SyntheticMessageGenerator│  │
│  │              │  │              │  │                          │  │
│  │ - load()     │  │ - user_id    │  │ - generate_user_text()   │  │
│  │ - filter()   │  │ - trace      │  │ - generate_tool_result() │  │
│  │ - sample()   │  │ - state      │  │ - generate_canonical_    │  │
│  │ - stats()    │  │ - metrics    │  │   prefix()               │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                      TestOrchestrator                         │  │
│  │                                                               │  │
│  │ - manage user lifecycle (add/remove/idle)                    │  │
│  │ - check performance thresholds                               │  │
│  │ - coordinate request dispatch with admission control         │  │
│  │ - collect and aggregate metrics                              │  │
│  │ - warm prefix management                                     │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                      APIClient                                │  │
│  │                                                               │  │
│  │ - auto-detect model                                          │  │
│  │ - apply model-specific settings (Qwen3-Coder, etc.)         │  │
│  │ - send streaming/non-streaming requests                      │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## User Lifecycle

```
                    ┌─────────┐
                    │  START  │
                    └────┬────┘
                         │
                    ┌────▼────┐
         ┌──────────│  IDLE   │◄─────────────┐
         │          └────┬────┘              │
         │               │ delay expires     │ request completes
         │               │                   │
         │    ┌──────────┼──────────┐        │
         │    │          │          │        │
         │    │ rate     │          │        │
         │    │ limited  │          │        │
         │    ▼          ▼          │        │
         │ ┌──────────┐ ┌────────┐  │        │
         │ │  RATE    │ │ ACTIVE │──┘        │
         │ │ LIMITED  │ └────┬───┘           │
         │ └────┬─────┘      │               │
         │      │            │               │
         │      │ backoff    │               │
         │      │ expires    │               │
         │      └─────►──────┘               │
         │               │
         │    ┌──────────┴──────────┐
         │    │                     │
    ┌────▼────▼───┐          ┌─────▼──────┐
    │  COMPLETED  │          │ TRUNCATED  │
    └─────────────┘          └────────────┘
         │                        │
         └───────────┬────────────┘
                     │ if --recycle
              ┌──────▼──────┐
              │ NEW SESSION │
              └─────────────┘
```

**Note:** The `RATE_LIMITED` state only occurs when `--enable-request-rate-limiting` is enabled and TTFT exceeds the threshold. Users in this state wait with exponential backoff before retrying.
