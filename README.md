# Inference Server Cache Performance Testing Suite
A toolkit for testing and benchmarking Large Language Model (LLM) inference servers and their KV Cache offload performance for optimal cache hit rates. There is an accompanying guide on how I use this coming soon to a public link. 

  

## Quick Start

  

```bash
# Install dependencies
pip install -r requirements.txt

# Run a simple single-prompt test
python single_prompt_tester.py \
--api-endpoint http://localhost:8000 \
--min-tokens 1000 \
--max-tokens 128000 \
--output-dir results

# Test various cache hit rates
python cache_rate_tester.py \
--api-endpoint http://localhost:8000 \
--context-sizes 8000 32000 64000 \
--working-set-size 2000000 \
--output-dir output

# Test performance across different memory tiers
python working_set_tester.py \
--api-endpoint http://localhost:8000 \
--context-sizes 30000 \
--min-working-set-size 100000 \
--max-working-set-size 5000000 \
--working-set-increments 10
```

## Table of Contents
- [Core Testing Tools](#core-testing-tools)
- [single_prompt_tester.py](#single_prompt_testerpy)
- [cache_rate_tester.py](#cache_rate_testerpy)
- [working_set_tester.py](#working_set_testerpy)
- [Utility Scripts](#utility-scripts)
- [generate_index.py](#generate_indexpy)
- [regenerate_graphs.py](#regenerate_graphspy)
- [combine_graphs.py](#combine_graphspy)
- [kv_cache_size.py](#kv_cache_sizepy)
- [Installation](#installation)
- [Understanding the Metrics](#understanding-the-metrics)

  

---

  

## Core Testing Tools
### single_prompt_tester.py
**Purpose:** Measure baseline cache performance with simple cold start vs. cached prompt comparisons. This is the simplest tool and provides quick insights into whether caching is working at all.

**What it does:**
- Tests at doubling context sizes (1K, 2K, 4K, 8K, ...)
- For each size, sends a unique prompt (cold start) then repeats the same prompt (100% cached)
- Measures TTFT speedup from caching

**Key Options:**
- `--api-endpoint`: Your inference server URL (required)
- `--min-tokens`: Minimum context size (default: 1000)
- `--max-tokens`: Maximum context size (default: 128000)
- `--output-tokens`: Tokens to generate per request (default: 256)
- `--num-iterations`: Iterations per context size (default: 5)
- `--output-dir`: Where to save results (default: ./single_prompt_output)
- `--tokenizer`: HuggingFace tokenizer ID (default: Qwen/Qwen2.5-Coder-32B-Instruct)

**Example Usage:**

```bash
# Basic test with defaults
python single_prompt_tester.py \
--api-endpoint http://localhost:8000

# Test up to 256K context with more iterations
python single_prompt_tester.py \
--api-endpoint http://localhost:8000 \
--min-tokens 2000 \
--max-tokens 256000 \
--num-iterations 10 \
--output-dir baseline_results

# Test with custom tokenizer
python single_prompt_tester.py \
--api-endpoint http://localhost:8000 \
--tokenizer meta-llama/Llama-3.3-70B-Instruct \
--output-dir llama_test
```

 
**Output:**
- `single_prompt_performance.html`: Interactive graphs showing TTFT, TTLT, and speedup
- `summary_table.html`: Table with statistical breakdown
- `single_prompt_results_*.csv`: Raw data
- `index.html`: Dashboard linking all results

**When to use:**
- Quick smoke test to verify caching is working
- Understanding baseline performance of a single memory tier
- Measuring maximum achievable speedup at various context sizes

**Limitations:**
- Only tests 0% and 100% cache hit rates
- Single request at a time (no concurrency)
- Fixed working set (all prompts cached)

  

---
### cache_rate_tester.py
**Purpose:** The primary testing tool. Measures performance at various cache hit rates (0%, 5%, 10%, ..., 100%) with fixed working set size. This reveals how performance scales with cache efficiency.

**What it does:**
- Pre-warms a working set of prompts
- Tests each cache hit rate by mixing cached prefixes with unique suffixes
- Automatically ramps concurrency to find peak throughput
- Retries at peak concurrency for stable measurement
  
**Key Options:**
**Required:**
- `--api-endpoint`: Your inference server URL
- `--context-sizes`: Context lengths to test (e.g., `8000 32000 64000`)
- `--working-set-size`: Total tokens in working set (e.g., 2000000)

**Inference Session Tuning:**
- `--output-tokens`: Tokens per request (default: 256)
- `--max-ttft`: TTFT threshold in seconds (default: 2.0)
- `--ttft-metric`: Which metric to use for threshold - `max`, `avg`, or `p95` (default: p95)
- `--test-duration`: Max seconds per cache rate test (default: 300)
- `--ramp-duration`: Seconds per concurrency level during ramp (default: 60)
- `--num-retries`: Retry runs at peak concurrency (default: 3)

**Concurrency Control:**
- `--start-concurrency`: Starting concurrent requests (default: 2)
- `--concurrency-increment`: Increment step (default: 2)
- `--max-concurrency`: Maximum concurrent requests (default: 1000)
- `--init-concurrency`: Concurrency for working set initialization (default: 16)

  

**Cache Behavior:**
- `--cache-hit-rates`: Override default rates (default: 0 5 10 15 ... 100)
- `--chunk-size`: Cache block alignment in tokens (default: 256)
- `--reinit-strategy`: When to re-init cache - `once`, `per_cache_rate`, or `per_test` (default: once)
- `--random-working-set-selection`: Use random vs. round-robin prompt selection


**Other:**
- `--output-dir`: Output directory (default: ./output)
- `--tokenizer`: Tokenizer model ID
- `--seed`: Random seed for reproducibility
- `--kv-cache-quantization`: 1 for FP8, 2 for FP16 estimation (default: 2)
- `--strict-time-window`: Only count requests that completed within duration window
- `--skip-graphs`: Don't generate graphs
- `--force-restart`: Ignore progress and restart
- `--verbose`: Enable debug logging

**Example Usage:**

```bash
# Basic test with single context size
python cache_rate_tester.py \
--api-endpoint http://localhost:8000 \
--context-sizes 32000 \
--working-set-size 2000000 \
--output-dir test_output

# Test multiple context sizes with custom cache rates
python cache_rate_tester.py \
--api-endpoint http://localhost:8000 \
--context-sizes 8000 32000 64000 128000 \
--working-set-size 5000000 \
--cache-hit-rates 0 25 50 75 100 \
--output-dir multi_context

# High concurrency test with stricter TTFT
python cache_rate_tester.py \
--api-endpoint http://localhost:8000 \
--context-sizes 30000 \
--working-set-size 10000000 \
--max-ttft 1.0 \
--ttft-metric p95 \
--max-concurrency 200 \
--test-duration 600 \
--output-dir high_load

# Resume from previous run
# Automatically resumes if progress.json exists
python cache_rate_tester.py \
--api-endpoint http://localhost:8000 \
--context-sizes 32000 64000 \
--working-set-size 2000000 \
--output-dir test_output

```

  

**Output:**
- `performance_vs_cache_rate_{context}.html`: Main performance graphs per context
- `ramp_ctx{context}_cache{rate}.html`: Concurrency ramp analysis per test
- `input_throughput_comparison.html`: Input throughput across context sizes
- `output_throughput_comparison.html`: Output throughput across context sizes
- `output_metrics_comparison.html`: ITL and per-request metrics
- `ttft_heatmap.html`: 2D heatmap (if multiple context sizes)
- `summary_*.csv`: Aggregated metrics
- `detailed_results_*.csv`: Per-request data
- `phase_metadata_*.csv`: Per-phase timing data
- `index.html`: Dashboard
- `run_command_*.sh`: Executable script to reproduce test

 
**When to use:**
- Primary tool for comprehensive cache testing
- Understanding performance across cache efficiency spectrum
- Finding optimal working set size for your memory tier
- Comparing different server configurations

  

**Pro Tips:**
- Start with smaller `--working-set-size` to verify setup
- Use `--cache-hit-rates 0 50 100` for quick tests
- Increase `--num-retries` for more stable measurements
- Resume capability lets you add more context sizes later

---

### working_set_tester.py
**Purpose:** Test performance across varying working set sizes to understand memory tier transitions. Shows how throughput degrades as working set exceeds cache capacity.

**What it does:**
- Similar to cache_rate_tester.py but varies working set size instead of cache hit rate
- Useful for testing HBM → DRAM → SSD tier transitions
- Can test multiple cache hit rates at each working set size

**Key Options:**
**Required:**
- `--api-endpoint`: Your inference server URL
- `--context-sizes`: Context lengths to test
- `--min-working-set-size`: Minimum working set in tokens (e.g., 100000)
- `--max-working-set-size`: Maximum working set in tokens (e.g., 5000000)
- `--working-set-increments`: Number of size steps to test (e.g., 10)

**Cache Behavior:**
- `--cache-hit-rates`: Rates to test (default: [100])

**Other options:** Similar to cache_rate_tester.py (TTFT, concurrency, output, etc.)

**Special Options:**
- `--ensure-working-set-coverage`: Guarantees working_set_size worth of input tokens sent (overrides --ramp-duration)
**Example Usage:**
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
--api-edpoint http://localhost:8000 \
--context-sizes 30000 \
--min-working-set-size 500000 \
--max-working-set-size 10000000 \
--working-set-increments 5 \
--cache-hit-rates 0 50 100 \
--output-dir multi_tier_test

# Ensure full working set coverage
python working_set_tester.py \
--api-endpoint http://localhost:8000 \
--context-sizes 30000 \
--min-working-set-size 1000000 \
--max-working-set-size 20000000 \
--working-set-increments 8 \
--ensure-working-set-coverage \
--output-dir coverage_test

```


**Output:**

- `performance_vs_working_set_{context}.html`: Main graphs per context
- `output_throughput_comparison.html`: Cross-context comparison
- `output_metrics_comparison.html`: ITL and generation metrics
- `summary_*.csv`: Aggregated results
- `detailed_results_*.csv`: Per-request data
- `phase_metadata_*.csv`: Timing data
- `index.html`: Dashboard

  

**When to use:**
- Testing performance across different memory tiers (HBM/DRAM/SSD)
- Finding optimal cache sizes for your hardware
- Understanding cache eviction behavior
- Validating tiered caching systems



---

  

## Utility Scripts
### generate_index.py
**Purpose:** Generate HTML dashboard from test results.
**Usage:**

```bash

# Automatically called by main tools
python generate_index.py ./output
python generate_index.py ./output 1.6  # with version

```

  

**What it creates:**
- Links to all graphs
- Summary statistics
- Test configuration
- KV cache estimates
---

  

### regenerate_graphs.py
**Purpose:** Regenerate graphs from existing test data using updated logic. Useful when graph generation code improves.

**Usage:**

```bash
# For cache_rate_tester.py results
python regenerate_graphs.py output_directory

# For working_set_tester.py results
python regenerate_graphs.py output_directory --tool working_set

```

  

**Example:**
```bash
# Regenerate after running cache_rate_tester.py
python regenerate_graphs.py ./my_test_output

# Regenerate working_set_tester.py graphs
python regenerate_graphs.py ./memory_tier_test --tool working_set
```

  

**What it does:**
- Loads summary_*.csv and detailed_results_*.csv
- Recalculates retry-based metrics
- Regenerates all main comparison graphs
- Regenerates ramp graphs (for cache_rate tests)
- Uses latest graph generation logic


**When to use:**
- After updating graph generation code
- To apply new visualization improvements to old data
- To regenerate with different styling
- When graphs are missing but CSV data exists


---


### combine_graphs.py
**Purpose:** Combine results from multiple test runs into unified comparison graphs. Great for A/B testing configurations.
**Usage:**

```bash
python combine_graphs.py \
--output-dirs run1/output run2/output run3/output \
--labels "Baseline" "Optimized" "Production" \
--output combined_results
```

**Options:**
- `--output-dirs`: Paths to output directories (required)
- `--labels`: Labels for each run (optional, uses dir names if omitted)
- `--output`: Output directory (default: ./combined_output)
- `--verbose`: Enable debug logging

**Example:**

```bash
# Compare two configurations
python combine_graphs.py \
--output-dirs baseline/output optimized/output \
--labels "Before" "After" \
--output comparison
  
# Compare three memory configurations
python combine_graphs.py \
--output-dirs hbm_only/output hbm_dram/output hbm_dram_ssd/output \
--labels "HBM Only" "HBM+DRAM" "HBM+DRAM+SSD" \
--output memory_comparison

# Auto-generated labels
python combine_graphs.py \
--output-dirs test1/output test2/output \
--output comparison
# Uses "test1" and "test2" as labels

```

  

**Output:**
- `input_throughput_ctx{size}.html`: Combined input throughput per context
- `output_throughput_ctx{size}.html`: Combined output throughput per context
- `ttft_comparison_ctx{size}.html`: Combined TTFT metrics per context
- `summary_comparison.csv`: Summary table
- `summary_comparison.html`: HTML table
- `index.html`: Dashboard


**When to use:**
- Comparing different server configurations
- A/B testing optimization changes
- Comparing hardware setups
- Evaluating software versions

**Pro Tips:**

- Ensure all runs used same `--context-sizes`
- Runs can have different cache hit rates (graphs will overlay)
- Uses retry-based metrics for consistency
- Best with 2-5 configurations (too many gets cluttered)
---

### kv_cache_size.py

**Purpose:** Calculate KV cache memory requirements for any HuggingFace model.

**Usage:**
```bash
python kv_cache_size.py MODEL_ID
python kv_cache_size.py --precision fp16 MODEL_ID
python kv_cache_size.py --batch-size 4 MODEL_ID
python kv_cache_size.py --custom-tokens 5000 50000 500000 MODEL_ID
```

**Options:**
- `model_id`: HuggingFace model identifier (required)
- `--precision`: `fp8` or `fp16` (default: fp8)
- `--custom-tokens`: Custom token counts to calculate

**Example:**
```bash
# Basic calculation
python kv_cache_size.py meta-llama/Llama-3.3-70B-Instruct
```

 

**When to use:**
- Planning cache capacity
- Understanding model memory requirements
- Comparing model architectures
- Validating test configurations

  
---

  

## Installation

  
```bash
# Clone or download this repository
git clone <your-repo-url>

# Install dependencies
pip install -r requirements.txt

# Verify installation
python single_prompt_tester.py --help

```

  
