# single_prompt_tester.py

Measure baseline cache performance with simple cold start vs. cached prompt comparisons. This is the simplest tool and provides quick insights into whether caching is working at all.

## What it does

- Tests at doubling context sizes (1K, 2K, 4K, 8K, ...) or specific sizes
- For each size, sends a unique prompt (cold start) then repeats the same prompt (100% cached)
- Measures TTFT speedup from caching
- Optionally sends multiple concurrent prompts and/or repeats cached prompts

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `--api-endpoint` | Your inference server URL | **required** |
| `--context-sizes` | Specific context sizes to test (e.g., `8000 32000 64000`). Overrides min/max-tokens. | - |
| `--min-tokens` | Minimum context size (used with doubling if --context-sizes not specified) | 1000 |
| `--max-tokens` | Maximum context size (used with doubling if --context-sizes not specified) | 128000 |
| `--output-tokens` | Tokens to generate per request | 256 |
| `--num-iterations` | Iterations per context size | 5 |
| `--concurrent-prompts`, `-n` | Send N prompts simultaneously | 1 |
| `--cached-repeats`, `-r` | Repeat cached prompt N times | 1 |
| `--output-dir` | Where to save results | ./single_prompt_output |
| `--tokenizer` | HuggingFace tokenizer ID | Qwen/Qwen2.5-Coder-32B-Instruct |
| `--seed` | Random seed for reproducibility | - |
| `--verbose` | Enable debug logging | false |
| `--brief` | Agent-friendly minimal output | false |

## Example Usage

```bash
# Basic test with defaults (doubling from 1K to 128K)
uv run python single_prompt_tester.py \
    --api-endpoint http://localhost:8000

# Test specific context sizes
uv run python single_prompt_tester.py \
    --api-endpoint http://localhost:8000 \
    --context-sizes 8000 32000 64000 \
    --output-dir specific_sizes

# Test with concurrent prompts
uv run python single_prompt_tester.py \
    --api-endpoint http://localhost:8000 \
    --context-sizes 32000 \
    --concurrent-prompts 10 \
    --output-dir concurrent_test

# Test with multiple cached repeats
uv run python single_prompt_tester.py \
    --api-endpoint http://localhost:8000 \
    --context-sizes 32000 \
    --cached-repeats 5 \
    --output-dir repeat_test

# Test up to 256K context with more iterations (doubling mode)
uv run python single_prompt_tester.py \
    --api-endpoint http://localhost:8000 \
    --min-tokens 2000 \
    --max-tokens 256000 \
    --num-iterations 10 \
    --output-dir baseline_results

# Test with custom tokenizer
uv run python single_prompt_tester.py \
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
- Testing server behavior under concurrent load (with `-n`)

## Limitations

- Only tests 0% and 100% cache hit rates
- Fixed working set (all prompts cached)
