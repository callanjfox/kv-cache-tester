# Utility Scripts

## generate_index.py

Unified HTML dashboard generator that auto-detects test type and creates appropriate output.

```bash
# Automatically called by main tools
python generate_index.py ./output
python generate_index.py ./output 1.6  # with version
```

**Supported test types (auto-detected):**
- `single_prompt` - Cold start vs cached performance
- `cache_rate` - Performance across cache hit rates
- `working_set` - Performance across working set sizes
- `sustained` - Continuous mode with adaptive concurrency
- `combined` - Multi-run comparison

**What it creates:**
- Test-type-specific header and title
- Configuration table (from progress.json or metadata.json)
- Performance summary statistics
- KV cache estimates (for cache/working set tests)
- Links to all graphs (organized by type)
- Data files section
- Test-type-specific interpretation notes

**Config sources:**
- `progress.json` - Used by cache_rate_tester and working_set_tester
- `metadata.json` - Used by single_prompt_tester

---

## regenerate_graphs.py

Regenerate graphs from existing test data using updated logic. Useful when graph generation code improves.

```bash
# For cache_rate_tester.py results
python regenerate_graphs.py output_directory

# For working_set_tester.py results
python regenerate_graphs.py output_directory --tool working_set
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

## combine_graphs.py

Combine results from multiple test runs into unified comparison graphs. Great for A/B testing configurations.

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--output-dirs` | Paths to output directories | **required** |
| `--labels` | Labels for each run | directory names |
| `--output` | Output directory | ./combined_output |
| `--verbose` | Enable debug logging | false |

### Example Usage

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

### Output Files

| File | Description |
|------|-------------|
| `input_throughput_ctx{size}.html` | Combined input throughput per context |
| `output_throughput_ctx{size}.html` | Combined output throughput per context |
| `ttft_comparison_ctx{size}.html` | Combined TTFT metrics per context |
| `summary_comparison.csv` | Summary table |
| `summary_comparison.html` | HTML table |
| `index.html` | Dashboard |

### When to use

- Comparing different server configurations
- A/B testing optimization changes
- Comparing hardware setups
- Evaluating software versions

### Pro Tips

- Ensure all runs used same `--context-sizes`
- Runs can have different cache hit rates (graphs will overlay)
- Uses retry-based metrics for consistency
- Best with 2-5 configurations (too many gets cluttered)

---

## kv_cache_size.py

Calculate KV cache memory requirements for any HuggingFace model.

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `model_id` | HuggingFace model identifier | **required** |
| `--precision` | `fp8` or `fp16` | fp8 |
| `--custom-tokens` | Custom token counts to calculate | - |

### Example Usage

```bash
# Basic calculation
python kv_cache_size.py meta-llama/Llama-3.3-70B-Instruct

# With custom precision
python kv_cache_size.py --precision fp16 MODEL_ID

# With custom token counts
python kv_cache_size.py --custom-tokens 5000 50000 500000 MODEL_ID
```

### When to use

- Planning cache capacity
- Understanding model memory requirements
- Comparing model architectures
- Validating test configurations
