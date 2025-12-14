# single_prompt_tester.py

Measure baseline cache performance with simple cold start vs. cached prompt comparisons. This is the simplest tool and provides quick insights into whether caching is working at all.

## What it does

- Tests at doubling context sizes (1K, 2K, 4K, 8K, ...)
- For each size, sends a unique prompt (cold start) then repeats the same prompt (100% cached)
- Measures TTFT speedup from caching

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `--api-endpoint` | Your inference server URL | **required** |
| `--context-sizes` | Specific context sizes to test (e.g., `8000 32000 64000`). Overrides min/max-tokens. | - |
| `--min-tokens` | Minimum context size (used with doubling if --context-sizes not specified) | 1000 |
| `--max-tokens` | Maximum context size (used with doubling if --context-sizes not specified) | 128000 |
| `--output-tokens` | Tokens to generate per request | 256 |
| `--num-iterations` | Iterations per context size | 5 |
| `--output-dir` | Where to save results | ./single_prompt_output |
| `--tokenizer` | HuggingFace tokenizer ID | Qwen/Qwen2.5-Coder-32B-Instruct |

## Example Usage

```bash
# Basic test with defaults (doubling from 1K to 128K)
python single_prompt_tester.py \
    --api-endpoint http://localhost:8000

# Test specific context sizes
python single_prompt_tester.py \
    --api-endpoint http://localhost:8000 \
    --context-sizes 8000 32000 64000 \
    --output-dir specific_sizes

# Test a single context size
python single_prompt_tester.py \
    --api-endpoint http://localhost:8000 \
    --context-sizes 32000 \
    --num-iterations 10 \
    --output-dir single_size_test

# Test up to 256K context with more iterations (doubling mode)
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

## Output Files

| File | Description |
|------|-------------|
| `single_prompt_performance.html` | Interactive graphs showing TTFT, TTLT, and speedup |
| `summary_table.html` | Table with statistical breakdown |
| `single_prompt_results_*.csv` | Raw data |
| `index.html` | Dashboard linking all results |

## When to use

- Quick smoke test to verify caching is working
- Understanding baseline performance of a single memory tier
- Measuring maximum achievable speedup at various context sizes

## Limitations

- Only tests 0% and 100% cache hit rates
- Single request at a time (no concurrency)
- Fixed working set (all prompts cached)
