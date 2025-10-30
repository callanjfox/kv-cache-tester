# Changelog

All notable changes to KV Cache Tester will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Fixed
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

### Changed
- `generate_graphs()` function now requires `config: TestConfig` parameter in both tools
- Updated comparison logging to only show when strict mode is NOT enabled (since filtered data is already being used)
- **Resume behavior improved**: When resuming tests, loaded aggregated results are now filtered against `progress.json` to remove any partial/incomplete results from interrupted runs
  - Only results marked as completed in `progress.json` are kept
  - Partial results are discarded and tests are re-run completely
  - Provides clean, consistent data and prevents duplicate entries

### Added
- `.gitignore` file to exclude test artifacts (`output/`, `*.log`, `__pycache__/`, etc.)

## Notes

### What `--strict-time-window` Does

When `--strict-time-window` is enabled:
- **Only requests that completed within the ramp duration window are used for all calculations**
- Throughput metrics reflect only "in-window" performance
- Ramp decisions (TTFT threshold checks, peak concurrency selection) use filtered metrics
- Graphs show filtered data with variability calculated from filtered requests
- Late-completing requests still finish gracefully but are excluded from metrics

This is useful for understanding steady-state performance without being affected by cleanup overhead from requests that started near the end of the time window but finished late.
