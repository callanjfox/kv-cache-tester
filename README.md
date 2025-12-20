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
