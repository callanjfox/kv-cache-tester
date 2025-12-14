# KV Cache Tester

A toolkit for testing LLM inference server KV cache performance.

## Tools Overview

### Core Testing Tools

- **single_prompt_tester.py** - Baseline cache test: cold start vs 100% cached. Simple smoke test for cache functionality.
- **cache_rate_tester.py** - Primary tool: tests performance across cache hit rates (0-100%) with fixed working set. Finds peak throughput via concurrency ramping.
- **working_set_tester.py** - Memory tier testing: varies working set size to observe HBM/DRAM/SSD transitions.

### Utilities

- **generate_index.py** - Creates HTML dashboard from results
- **regenerate_graphs.py** - Rebuilds graphs from CSV data
- **combine_graphs.py** - Combines multiple test runs for comparison
- **kv_cache_size.py** - Calculates KV cache memory requirements

## Current Work

Simplifying documentation structure:
- Moved detailed docs from README to `docs/` directory
- README now has brief overview with links to detailed docs

## Documentation Files

- `docs/single_prompt_tester.md` - single_prompt_tester usage
- `docs/cache_rate_tester.md` - cache_rate_tester usage
- `docs/working_set_tester.md` - working_set_tester usage
- `docs/utilities.md` - utility scripts usage

## Key Metrics

- **TTFT** - Time to First Token (cache effectiveness indicator)
- **TTLT** - Time to Last Token (total latency)
- **ITL** - Inter-Token Latency
- **Throughput** - Input/output tokens per second

## Test Server Scripts (../tools/)

Scripts for running a test server when testing changes to kv-cache-tester:

- **start_server.sh** - Start vLLM with Llama 3.3 8B, TP=2, port 8000
- **stop_server.sh** - Stop the test server and release GPU memory

Usage:
```bash
cd ../tools
./start_server.sh          # Start server (uses GPUs 6,7 by default)
# ... run tests against http://localhost:8000 ...
./stop_server.sh           # Stop server when done
```

IMPORTANT: Always run `stop_server.sh` after testing to release GPU memory.
