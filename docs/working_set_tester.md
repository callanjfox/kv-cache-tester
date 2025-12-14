# working_set_tester.py

Test performance across varying working set sizes to understand memory tier transitions. Shows how throughput degrades as working set exceeds cache capacity. Supports two modes: adaptive (default) and sustained.

## What it does

- **Adaptive Mode (default):** Tests multiple working set sizes sequentially with ramp-up/retry phases, similar to cache_rate_tester.py but varying working set size instead of cache hit rate
- **Sustained Mode:** Single continuous test with dynamic working set growth during execution, enabling observation of cache tier effects (HBM -> DRAM -> SSD) over time
- Useful for testing HBM -> DRAM -> SSD tier transitions
- Can test multiple cache hit rates at each working set size (adaptive mode) or single rate (sustained mode)

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
| `--mode` | Test mode (`adaptive` or `sustained`) | adaptive |
| `--assessment-period` | Seconds between assessments in sustained mode | 30 |

### Cache Behavior

| Option | Description | Default |
|--------|-------------|---------|
| `--cache-hit-rates` | Rates to test | [100] |

### Special Options

| Option | Description |
|--------|-------------|
| `--ensure-working-set-coverage` | Guarantees working_set_size worth of input tokens sent (overrides --ramp-duration, adaptive mode only) |

Other options are similar to cache_rate_tester.py (TTFT, concurrency, output, etc.)

## Example Usage

```bash
# Test across memory tiers with 100% cache hit rate
python working_set_tester.py \
    --api-endpoint http://localhost:8000 \
    --context-sizes 30000 \
    --min-working-set-size 100000 \
    --max-working-set-size 5000000 \
    --working-set-increments 10 \
    --output-dir memory_tier_test

# Test multiple cache hit rates at each working set size
python working_set_tester.py \
    --api-endpoint http://localhost:8000 \
    --context-sizes 30000 \
    --min-working-set-size 500000 \
    --max-working-set-size 10000000 \
    --working-set-increments 5 \
    --cache-hit-rates 0 50 100 \
    --output-dir multi_tier_test

# Ensure full working set coverage (adaptive mode)
python working_set_tester.py \
    --api-endpoint http://localhost:8000 \
    --context-sizes 30000 \
    --min-working-set-size 1000000 \
    --max-working-set-size 20000000 \
    --working-set-increments 8 \
    --ensure-working-set-coverage \
    --output-dir coverage_test

# Sustained mode: single test with dynamic growth
python working_set_tester.py \
    --api-endpoint http://localhost:8000 \
    --context-sizes 10000 \
    --min-working-set-size 100000 \
    --max-working-set-size 2000000 \
    --working-set-increments 4 \
    --cache-hit-rates 90 \
    --mode sustained \
    --assessment-period 30 \
    --test-duration 600 \
    --max-ttft 5.0 \
    --output-dir sustained_test
```

## Output Files

### Adaptive Mode

| File | Description |
|------|-------------|
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
| `sustained_periods_ctx{context}_ws{ws_size}_cache{rate}_{timestamp}.csv` | Period-by-period metrics |
| `detailed_results_{context}_{timestamp}.csv` | Per-request data |

Note: Automatic graph generation not yet implemented for sustained mode.

## When to use

- Testing performance across different memory tiers (HBM/DRAM/SSD)
- Finding optimal cache sizes for your hardware
- Understanding cache eviction behavior
- Validating tiered caching systems
