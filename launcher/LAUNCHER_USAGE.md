# vLLM Engine Launcher Usage

## Quick Start

### 1. Setup Your Configuration

```bash
# Copy the example config
cp config.example.yaml config.yaml

# Edit with your paths (minimum required: venvs.dynamo, venvs.lmcache)
vim config.yaml
```

### 2. Verify Setup with Dry Run

**IMPORTANT:** Tests can run for a long time. It is highly recommended to run with `--dry-run` first to verify your parameters and configuration are correct.

```bash
# Verify configuration and see what commands will execute
python vllm_engine_launcher.py --config-file config.yaml --model Qwen/Qwen2.5-7B --dry-run
```

The dry run will show:
- Which backends will be tested
- Environment variables that will be set
- Exact vLLM server commands that will be executed
- Output directories

### 3. Run Tests

Once you've verified the dry run output looks correct:

```bash
# Single prompt test (default)
python vllm_engine_launcher.py --config-file config.yaml --model Qwen/Qwen2.5-7B

# Cache rate test
python vllm_engine_launcher.py --config-file config.yaml --mode cache-rate --model Qwen/Qwen2.5-7B

# Working set test
python vllm_engine_launcher.py --config-file config.yaml --mode working-set --model Qwen/Qwen2.5-7B
```

## Configuration Methods

Settings can be provided via (in priority order):

1. **Config file** (recommended): `--config-file config.yaml`
2. **Environment variables**: Prefix with `KVTEST_` (e.g., `KVTEST_VENV_DYNAMO`)
3. **Command-line arguments**: Override anything

## Required Settings

Set these in your `config.yaml`:

```yaml
venvs:
  dynamo: "/path/to/dynamo/venv"
  lmcache: "/path/to/lmcache/.venv"
```

## Common Options

```yaml
output:
  base_dir: "./results"  # Where to save results

model:
  hf_home: "/path/to/hf_cache"  # Optional, uses HF default if not set
  default_model: "Qwen/Qwen2.5-7B"

server:
  default_port: 8002
  gpu_memory_utilization: 0.80
  tensor_parallel_size: 1

backends:
  enabled:  # Which backends to test (or use --configs flag)
    - dynamo-cpu
    - dynamo-weka
    - lmcache-dram
    - lmcache-weka
    - vanilla-nep
```

See `config.example.yaml` for all available options.

## Test Modes

### Single Prompt (`--mode single-prompt`, default)
Test performance across varying context sizes.

```bash
python vllm_engine_launcher.py \
  --config-file config.yaml \
  --model Qwen/Qwen2.5-7B \
  --min-tokens 1000 \
  --max-tokens 32768
```

### Cache Rate (`--mode cache-rate`)
Test performance at varying cache hit rates.

```bash
python vllm_engine_launcher.py \
  --config-file config.yaml \
  --mode cache-rate \
  --model Qwen/Qwen2.5-7B
```

### Working Set (`--mode working-set`)
Test performance with varying working set sizes.

```bash
python vllm_engine_launcher.py \
  --config-file config.yaml \
  --mode working-set \
  --model Qwen/Qwen2.5-7B \
  --cache-hit-rate 50
```

## Testing Specific Backends

```bash
python vllm_engine_launcher.py \
  --config-file config.yaml \
  --model Qwen/Qwen2.5-7B \
  --configs dynamo-cpu dynamo-weka vanilla-nep
```

Available: `dynamo-cpu`, `dynamo-weka`, `lmcache-dram`, `lmcache-weka`, `lmcache-nixl`, `vanilla-nep`, `vanilla-pc`

## Debugging

### Capturing Server Logs

To debug server issues, capture vLLM engine output logs to files:

```bash
# Set debug logs directory in config.yaml
output:
  debug_logs_dir: "/path/to/debug/logs"

# Or use environment variable
export KVTEST_OUTPUT_DEBUG_LOGS_DIR="./debug_logs"

# Or pass via CLI (overrides config and env vars)
python vllm_engine_launcher.py \
  --config-file config.yaml \
  --model Qwen/Qwen2.5-7B \
  --debug-logs ./debug_logs
```

Server output will be written to timestamped log files like:
- `debug_logs/Dynamo-CPU_20251114_123045.log`
- `debug_logs/LMCache-WEKA_20251114_130215.log`

These logs contain:
- Model loading progress
- Memory allocation details
- KV cache configuration
- Runtime errors and warnings

## Environment Variables

As an alternative to config files:

```bash
export KVTEST_VENV_DYNAMO=/path/to/dynamo/venv
export KVTEST_VENV_LMCACHE=/path/to/lmcache/.venv
export KVTEST_OUTPUT_BASE_DIR=./results

python vllm_engine_launcher.py --model Qwen/Qwen2.5-7B
```

See the source code for the full list of `KVTEST_*` environment variables.

## Troubleshooting

**Always start with a dry run:**
```bash
# Verify configuration before running
python vllm_engine_launcher.py --config-file config.yaml --model Qwen/Qwen2.5-7B --dry-run
```

**View current configuration:**
```bash
python vllm_engine_launcher.py --config-file config.yaml --print-config
```

**Verify paths exist:**
```bash
ls /path/to/dynamo/venv/bin/python
ls /path/to/lmcache/.venv/bin/python
```

**Enable debug logging:**
```bash
# Add to config.yaml
output:
  debug_logs_dir: "./debug_logs"
```

**Get help:**
```bash
python vllm_engine_launcher.py --help
```
