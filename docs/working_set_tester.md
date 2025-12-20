# working_set_tester.py

Test performance across varying working set sizes to understand memory tier transitions. Shows how throughput degrades as working set exceeds cache capacity. Both modes support dynamic working set growth during the test.

## What it does

- **Sustained Mode (default):** Continuous test with dynamic working set growth during execution. Concurrency adjusts automatically based on performance thresholds (adaptive). Best for finding optimal concurrency at various working set sizes.
- **Fixed Mode:** Same working set growth, but at a fixed concurrency level (no adjustment). Useful for comparing performance at a known concurrency across different working set sizes.
- Useful for testing HBM -> DRAM -> SSD tier transitions
- Can test multiple cache hit rates

## Options

### Required

| Option | Description |
|--------|-------------|
| `--api-endpoint` | Your inference server URL |
| `--context-sizes` | Context lengths to test |
| `--min-working-set-size` | Minimum working set in tokens (e.g., 100000) |
| `--max-working-set-size` | Maximum working set in tokens (e.g., 5000000) |
| `--working-set-increments` | Number of working set growth steps during the test |

### Mode Selection

| Option | Description | Default |
|--------|-------------|---------|
| `--mode` | Test mode (`sustained` or `fixed`) | sustained |
| `--assessment-period` | Seconds between assessments (both modes) | 30 |
| `--fixed-concurrency` | Fixed concurrency level (fixed mode only) | Required for fixed mode |

### Performance Thresholds (Sustained Mode Only)

| Option | Description | Notes |
|--------|-------------|-------|
| `--max-ttft` | Maximum acceptable TTFT in seconds | Required for sustained mode (or --min-tokens-per-req) |
| `--min-tokens-per-req` | Minimum output tokens/s per request | Required for sustained mode (or --max-ttft) |

**Sustained mode** requires at least one threshold for concurrency adjustment decisions.

**Fixed mode** does not use thresholds (concurrency stays fixed).

### Cache Behavior

| Option | Description | Default |
|--------|-------------|---------|
| `--cache-hit-rates` | Rates to test | [100] |

Other options are similar to cache_rate_tester.py (output tokens, test duration, etc.)

## Example Usage

```bash
# Sustained mode (default): adaptive concurrency with working set growth
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

# Fixed mode: fixed concurrency with working set growth
uv run python working_set_tester.py \
    --api-endpoint http://localhost:8000 \
    --context-sizes 30000 \
    --min-working-set-size 100000 \
    --max-working-set-size 5000000 \
    --working-set-increments 10 \
    --mode fixed \
    --fixed-concurrency 40 \
    --test-duration 300 \
    --output-dir fixed_test

# Fixed mode with multiple cache hit rates
uv run python working_set_tester.py \
    --api-endpoint http://localhost:8000 \
    --context-sizes 30000 \
    --min-working-set-size 500000 \
    --max-working-set-size 10000000 \
    --working-set-increments 5 \
    --cache-hit-rates 0 50 100 \
    --mode fixed \
    --fixed-concurrency 20 \
    --test-duration 120 \
    --output-dir multi_cache_test
```

## Output Files

Both modes produce the same output format:

| File | Description |
|------|-------------|
| `sustained_performance_ctx{context}_cache{rate}_{timestamp}.html` | Performance over time graph (shows working set growth) |
| `sustained_periods_ctx{context}_ws{ws_size}_cache{rate}_{timestamp}.csv` | Period-by-period metrics |
| `index.html` | Dashboard |

## When to use

- Testing performance across different memory tiers (HBM/DRAM/SSD)
- Finding optimal cache sizes for your hardware
- Understanding cache eviction behavior as working set grows
- Comparing adaptive vs fixed concurrency performance
- Validating tiered caching systems
