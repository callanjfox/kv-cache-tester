# cache_rate_tester.py

The primary testing tool. Measures performance at various cache hit rates (0%, 5%, 10%, ..., 100%) with fixed working set size. This reveals how performance scales with cache efficiency.

## Test Modes

| Mode | Description |
|------|-------------|
| `sustained` (default) | Sustained load testing with continuous concurrency adjustment. Recommended for production capacity planning. |
| `fixed` | Test specific concurrency levels with retries. Useful for benchmarking at known concurrency points. |

## What it does

- Pre-warms a working set of prompts
- Tests each cache hit rate by mixing cached prefixes with unique suffixes
- **Sustained mode**: Continuously adjusts concurrency based on performance thresholds
- **Fixed mode**: Tests specified concurrency levels with retries for stable measurements

## Options

### Required

| Option | Description |
|--------|-------------|
| `--api-endpoint` | Your inference server URL |
| `--context-sizes` | Context lengths to test (e.g., `8000 32000 64000`) |
| `--working-set-size` | Total tokens in working set (e.g., 2000000) |

### Mode Selection

| Option | Description | Default |
|--------|-------------|---------|
| `--mode` | Test mode (`sustained` or `fixed`) | sustained |
| `--fixed-concurrency-levels` | Concurrency levels for fixed mode (e.g., `10 20 40 80`) | - |

### Performance Thresholds

At least one threshold is required:

| Option | Description | Default |
|--------|-------------|---------|
| `--max-ttft` | TTFT threshold in seconds (e.g., 2.0) | - |
| `--ttft-metric` | Metric for TTFT threshold (`max`, `avg`, `p95`) | p95 |
| `--min-tokens-per-req` | Minimum output tokens/s per request | - |
| `--tokens-per-req-metric` | Metric for tokens/req (`avg`, `p5`, `p10`) | avg |

### Timing

| Option | Description | Default |
|--------|-------------|---------|
| `--test-duration` | Max seconds per cache rate test | 300 |
| `--ramp-duration` | Seconds per concurrency level (fixed mode) | 60 |
| `--assessment-period` | Seconds per assessment period (sustained mode) | 30 |
| `--num-retries` | Retry runs at each concurrency level (fixed mode) | 3 |

### Concurrency Control

| Option | Description | Default |
|--------|-------------|---------|
| `--start-concurrency` | Starting concurrent requests | 2 |
| `--concurrency-increment` | Increment step | 2 |
| `--max-concurrency` | Maximum concurrent requests | 1000 |
| `--init-concurrency` | Concurrency for working set initialization | 16 |

### Cache Behavior

| Option | Description | Default |
|--------|-------------|---------|
| `--cache-hit-rates` | Override default rates | 0 5 10 15 ... 100 |
| `--chunk-size` | Cache block alignment in tokens | 256 |
| `--reinit-strategy` | When to re-init cache (`once`, `per_cache_rate`, `per_test`) | once |
| `--random-working-set-selection` | Use random vs. round-robin prompt selection | false |

### Other

| Option | Description | Default |
|--------|-------------|---------|
| `--output-dir` | Output directory | ./output |
| `--output-tokens` | Output tokens per request | 256 |
| `--tokenizer` | Tokenizer model ID | - |
| `--seed` | Random seed for reproducibility | - |
| `--kv-cache-quantization` | 1 for FP8, 2 for FP16 estimation | 2 |
| `--strict-time-window` | Only count requests completed within duration | false |
| `--skip-graphs` | Don't generate graphs | false |
| `--force-restart` | Ignore progress and restart | false |
| `--verbose` | Enable debug logging | false |
| `--brief` | Agent-friendly minimal output | false |

## Example Usage

```bash
# Sustained mode (default) - production capacity planning
uv run python cache_rate_tester.py \
    --api-endpoint http://localhost:8000 \
    --context-sizes 32000 \
    --working-set-size 2000000 \
    --max-ttft 2.0 \
    --output-dir test_output

# Fixed mode - test specific concurrency levels
uv run python cache_rate_tester.py \
    --api-endpoint http://localhost:8000 \
    --context-sizes 32000 \
    --working-set-size 2000000 \
    --mode fixed \
    --fixed-concurrency-levels 10 20 40 80 \
    --max-ttft 2.0 \
    --output-dir fixed_output

# Multiple context sizes with custom cache rates
uv run python cache_rate_tester.py \
    --api-endpoint http://localhost:8000 \
    --context-sizes 8000 32000 64000 128000 \
    --working-set-size 5000000 \
    --cache-hit-rates 0 25 50 75 100 \
    --max-ttft 2.0 \
    --output-dir multi_context

# High concurrency test with stricter TTFT
uv run python cache_rate_tester.py \
    --api-endpoint http://localhost:8000 \
    --context-sizes 30000 \
    --working-set-size 10000000 \
    --max-ttft 1.0 \
    --ttft-metric p95 \
    --max-concurrency 200 \
    --test-duration 600 \
    --output-dir high_load

# Resume from previous run (automatic if progress.json exists)
uv run python cache_rate_tester.py \
    --api-endpoint http://localhost:8000 \
    --context-sizes 32000 64000 \
    --working-set-size 2000000 \
    --max-ttft 2.0 \
    --output-dir test_output
```

## Output Files

| File | Description |
|------|-------------|
| `performance_vs_cache_rate_{context}.html` | Main performance graphs per context |
| `ramp_ctx{context}_cache{rate}.html` | Concurrency ramp analysis per test |
| `input_throughput_comparison.html` | Input throughput across context sizes |
| `output_throughput_comparison.html` | Output throughput across context sizes |
| `output_metrics_comparison.html` | ITL and per-request metrics |
| `ttft_heatmap.html` | 2D heatmap (if multiple context sizes) |
| `summary_*.csv` | Aggregated metrics |
| `detailed_results_*.csv` | Per-request data |
| `phase_metadata_*.csv` | Per-phase timing data |
| `sustained_periods_*.csv` | Period data (sustained mode) |
| `index.html` | Dashboard |
| `run_command_*.sh` | Executable script to reproduce test |

## When to use

- **Sustained mode**: Production capacity planning, understanding sustainable throughput
- **Fixed mode**: Benchmarking at specific concurrency points, comparing configurations
- Understanding performance across cache efficiency spectrum
- Finding optimal working set size for your memory tier

## Pro Tips

- Start with smaller `--working-set-size` to verify setup
- Use `--cache-hit-rates 0 50 100` for quick tests
- In fixed mode, increase `--num-retries` for more stable measurements
- Resume capability lets you add more context sizes later
- Use `--brief` for automation and agent integration
