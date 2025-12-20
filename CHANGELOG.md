# Changelog

All notable changes to KV Cache Tester will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- **Unified working_set_tester modes**: Fixed mode now behaves like sustained mode
  - Both modes use assessment periods for stats collection
  - Both modes support working set growth during the test
  - Fixed mode keeps concurrency constant (no ramping up/down)
  - Changed `--fixed-concurrency-levels` to `--fixed-concurrency` (single value)
  - Summary shows min-max working set range (e.g., "51K-205K")
- **Per-test token summary**: After each cache hit rate test completes, displays total processed input/output tokens
  - Shows total requests, input tokens (with M suffix), output tokens (with M suffix)
  - Aligned with graph calculations for consistency
  - Works in both modes: fixed and sustained
- **Final summary table**: At end of all tests, displays comprehensive results table
  - Shows all test results sorted by context size and cache hit rate
  - Columns: Context, Cache%, Requests, Input Tok, Output Tok, Input/s, Output/s, Avg TTFT, Concurrency
  - Includes grand totals for requests and tokens processed
  - Provides quick overview of entire test run
- **Brief mode token totals**: Extended `--brief` output with token statistics
  - Added requests, input_tokens, output_tokens columns to CSV
  - Added total_requests, total_input_tokens, total_output_tokens summary lines
- **Documentation restructure**: Moved detailed tool documentation to `docs/` directory
  - `docs/single_prompt_tester.md` - single_prompt_tester usage
  - `docs/cache_rate_tester.md` - cache_rate_tester usage
  - `docs/working_set_tester.md` - working_set_tester usage
  - `docs/utilities.md` - utility scripts usage
- **Simplified README**: Now has brief overview with links to detailed docs
- **`--context-sizes` option for single_prompt_tester.py**: Specify exact context sizes to test (e.g., `--context-sizes 8000 32000 64000`) instead of using min/max-tokens doubling
- **Unified index.html generator**: `generate_index.py` now auto-detects test type and generates appropriate dashboard for all tools (single_prompt, cache_rate, working_set, sustained, combined)
  - Consistent styling across all test types
  - Auto-detects test type from output files
  - Reads config from progress.json or metadata.json
  - single_prompt_tester.py now uses the unified generator
- **`--brief` mode for all tools**: Agent-friendly output mode for single_prompt_tester, cache_rate_tester, and working_set_tester
  - Suppresses verbose logging, only shows warnings/errors
  - Outputs minimal, parseable CSV-like format at completion
  - Format: key-value header, blank line, CSV header, CSV data rows, blank line, output path
  - Useful for automation and integration with AI agents
- **`--concurrent-prompts` / `-n` option for single_prompt_tester.py**: Send N prompts simultaneously
  - Uses `asyncio.gather` to fire all prompts at once
  - Useful for testing server behavior under concurrent load
  - Reports per-prompt metrics plus batch-level statistics (avg TTFT, batch time)
  - Both unique (cold) and cached (warm) tests send all N prompts concurrently
- **`--cached-repeats` / `-r` option for single_prompt_tester.py**: Run cached prompt multiple times
  - After the unique prompt, repeat the cached prompt N times (default: 1)
  - Useful for measuring cache hit consistency and warm-up effects
  - Each repeat is labeled `(repeat 1/N)`, `(repeat 2/N)`, etc.
  - All cached repeats are aggregated together in summaries and graphs
- **Improved CLI output for single_prompt_tester.py**:
  - Added per-context-size summary after all iterations complete
  - Added final summary table at end showing all context sizes
  - Added blank lines between iterations for better readability
  - Brief mode now shows per-context summaries during testing and formatted final table
- `.gitignore` file to exclude test artifacts (`output/`, `*.log`, `__pycache__/`, etc.)

### Changed
- `generate_graphs()` function now requires `config: TestConfig` parameter in both tools
- Updated comparison logging to only show when strict mode is NOT enabled (since filtered data is already being used)
- **Resume behavior improved**: When resuming tests, loaded aggregated results are now filtered against `progress.json` to remove any partial/incomplete results from interrupted runs
  - Only results marked as completed in `progress.json` are kept
  - Partial results are discarded and tests are re-run completely
  - Provides clean, consistent data and prevents duplicate entries

### Fixed
- **working_set_tester index.html now shows all context sizes**
  - Previously, when testing multiple context sizes with multiple cache hit rates, only the last context size would appear in the dashboard
  - Now uses unified `generate_index.py` like other tools for consistent behavior
- **Critical: `--strict-time-window` flag now works correctly**
  - Fixed throughput calculations to filter requests BEFORE calculation when `--strict-time-window` is enabled
  - Previously, the flag would filter requests but throughput was calculated from unfiltered data, causing incorrect metrics
  - Now all calculations (aggregated metrics, ramp decisions, and graphs) properly respect the strict time window
- **Fixed duplicate entries crash in graph generation**
  - Added deduplication in `generate_graphs()` to handle resumed tests
  - Prevents "Index contains duplicate entries, cannot reshape" error in heatmap generation
  - Keeps most recent entry when same test configuration runs multiple times
- **Ramping logic now respects `--strict-time-window`**
  - Ramp phase TTFT threshold checks now use filtered metrics when strict mode is enabled
  - Peak concurrency selection is now based on strict window performance
  - Binary search refinement also respects strict window filtering
  - This ensures that concurrency decisions are made based on "in-window" request performance only
- **Graph generation now respects `--strict-time-window`**
  - Variability bands in graphs now calculated from filtered data when strict mode is enabled
  - Previously graphs recalculated from phase metadata without checking the flag
  - Now both main graph lines and variability calculations use consistent filtering

### Removed
- **Adaptive mode**: Removed `--mode adaptive` option from cache_rate_tester.py
  - Sustained mode (default) provides better production capacity planning
  - Fixed mode covers specific concurrency level testing needs
  - Reduces code complexity (~550 lines removed)
  - Only `sustained` and `fixed` modes remain
- **Dead code cleanup**: Removed unused `generate_continuous_index` function (~220 lines)
  - Was superseded by unified `generate_index.py` which auto-detects test type
- **Redundant working_set_tester code**: Removed ~130 lines of `run_fixed_concurrency_mode` function
  - Fixed mode now uses the same `run_continuous_mode` function as sustained mode
- **Duplicate index generation**: Removed `generate_sustained_index_html` function (~250 lines)
  - Was causing index.html to only show last context size when testing multiple
  - All tools now use unified `generate_index.py` for dashboard generation

## Notes

### What `--strict-time-window` Does

When `--strict-time-window` is enabled:
- **Only requests that completed within the ramp duration window are used for all calculations**
- Throughput metrics reflect only "in-window" performance
- Ramp decisions (TTFT threshold checks, peak concurrency selection) use filtered metrics
- Graphs show filtered data with variability calculated from filtered requests
- Late-completing requests still finish gracefully but are excluded from metrics

This is useful for understanding steady-state performance without being affected by cleanup overhead from requests that started near the end of the time window but finished late.
