# working_set_tester.py

Test performance across varying working set sizes to understand memory tier transitions. Shows how throughput degrades as working set exceeds cache capacity. Supports two modes: sustained (default) and fixed.

## What it does

- **Sustained Mode (default):** Continuous test with dynamic working set growth during execution, enabling observation of cache tier effects (HBM -> DRAM -> SSD) over time. Concurrency adjusts automatically based on performance thresholds.
- **Fixed Mode:** Tests multiple working set sizes sequentially at specific concurrency levels (similar to cache_rate_tester.py's fixed mode)
- Useful for testing HBM -> DRAM -> SSD tier transitions
- Can test multiple cache hit rates at each working set size

## Options

### Required

| Option | Description |
|--------|-------------|
| `--api-endpoint` | Your inference server URL |
| `--context-sizes` | Context lengths to test |
| `--min-working-set-size` | Minimum working set in tokens (e.g., 100000) |
| `--max-working-set-size` | Maximum working set in tokens (e.g., 5000000) |
| `--working-set-increments` | Number of size steps to test (e.g., 10) |

### Mode Selection

| Option | Description | Default |
|--------|-------------|---------|
| `--mode` | Test mode (`sustained` or `fixed`) | sustained |
| `--assessment-period` | Seconds between assessments in sustained mode | 30 |
| `--fixed-concurrency-levels` | Specific concurrency levels to test in fixed mode (e.g., `10 20 40 80`) | Required for fixed mode |

### Performance Thresholds

| Option | Description | Notes |
|--------|-------------|-------|
| `--max-ttft` | Maximum acceptable TTFT in seconds | Required for sustained mode (or --min-tokens-per-req) |
| `--min-tokens-per-req` | Minimum output tokens/s per request | Required for sustained mode (or --max-ttft) |

**Sustained mode** requires at least one threshold. Test continues until working set growth is complete or threshold is exceeded.

**Fixed mode** thresholds are optional. If provided, testing stops when any concurrency level exceeds the threshold.

### Cache Behavior

| Option | Description | Default |
|--------|-------------|---------|
| `--cache-hit-rates` | Rates to test | [100] |

Other options are similar to cache_rate_tester.py (concurrency, output tokens, retries, etc.)

## Example Usage

```bash
# Sustained mode (default): single test with dynamic working set growth
uv run python working_set_tester.py \
    --api-endpoint http://localhost:8000 \
    --context-sizes 10000 \
    --min-working-set-size 100000 \
    --max-working-set-size 2000000 \
    --working-set-increments 4 \
    --cache-hit-rates 90 \
    --assessment-period 30 \
    --test-duration 600 \
    --max-ttft 5.0 \
    --output-dir sustained_test

# Fixed mode: test specific concurrency levels at each working set size
uv run python working_set_tester.py \
    --api-endpoint http://localhost:8000 \
    --context-sizes 30000 \
    --min-working-set-size 100000 \
    --max-working-set-size 5000000 \
    --working-set-increments 10 \
    --mode fixed \
    --fixed-concurrency-levels 10 20 40 80 \
    --output-dir memory_tier_test

# Fixed mode with multiple cache hit rates
uv run python working_set_tester.py \
    --api-endpoint http://localhost:8000 \
    --context-sizes 30000 \
    --min-working-set-size 500000 \
    --max-working-set-size 10000000 \
    --working-set-increments 5 \
    --cache-hit-rates 0 50 100 \
    --mode fixed \
    --fixed-concurrency-levels 20 40 80 \
    --output-dir multi_tier_test

# Fixed mode with threshold-based short-circuit
uv run python working_set_tester.py \
    --api-endpoint http://localhost:8000 \
    --context-sizes 30000 \
    --min-working-set-size 1000000 \
    --max-working-set-size 20000000 \
    --working-set-increments 8 \
    --mode fixed \
    --fixed-concurrency-levels 10 20 40 80 160 \
    --max-ttft 3.0 \
    --output-dir threshold_test
```

## Output Files

### Fixed Mode

| File | Description |
|------|-------------|
| `fixed_ctx{context}_ws{ws_size}_cache{rate}.html` | Per-test concurrency analysis graph |
| `performance_vs_working_set_{context}.html` | Main graphs per context |
| `output_throughput_comparison.html` | Cross-context comparison |
| `output_metrics_comparison.html` | ITL and generation metrics |
| `summary_*.csv` | Aggregated results |
| `detailed_results_*.csv` | Per-request data |
| `phase_metadata_*.csv` | Timing data |
| `index.html` | Dashboard |

### Sustained Mode

| File | Description |
|------|-------------|
| `sustained_ctx{context}_ws{ws_size}_cache{rate}.html` | Working set growth analysis graph |
| `sustained_periods_ctx{context}_ws{ws_size}_cache{rate}_{timestamp}.csv` | Period-by-period metrics |
| `detailed_results_{context}_{timestamp}.csv` | Per-request data |
| `index.html` | Dashboard |

## When to use

- Testing performance across different memory tiers (HBM/DRAM/SSD)
- Finding optimal cache sizes for your hardware
- Understanding cache eviction behavior
- Validating tiered caching systems
