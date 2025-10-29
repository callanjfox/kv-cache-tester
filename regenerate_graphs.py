#!/usr/bin/env python3
"""
Regenerate Graphs Tool

Re-generates graphs from existing cache_rate_tester.py or working_set_tester.py
output directories using the latest graph generation logic.

Usage:
    # For cache_rate_tester.py output
    python regenerate_graphs.py cacherate_final-noprefix-s256-b32768-dp4-tp2-dram320

    # For working_set_tester.py output
    python regenerate_graphs.py my_output --tool working_set
"""

import argparse
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from cache_rate_tester import generate_graphs as generate_cache_graphs, generate_ramp_graph, AggregatedMetrics, RequestMetrics, Colors, init_logger
from working_set_tester import generate_graphs as generate_ws_graphs, AggregatedMetrics as WSAggregatedMetrics
import pandas as pd
import numpy as np


def load_summary_csv(output_dir: Path):
    """Load the most recent summary CSV from output directory"""
    summary_files = list(output_dir.glob("summary_*.csv"))

    if not summary_files:
        raise FileNotFoundError(f"No summary_*.csv files found in {output_dir}")

    # Get most recent
    most_recent = max(summary_files, key=lambda p: p.stat().st_mtime)
    print(f"  Loading summary from: {most_recent.name}")

    df = pd.read_csv(most_recent)
    return df


def load_detailed_results(output_dir: Path, context_size: int):
    """Load the most recent detailed results CSV for a context size"""
    detailed_files = list(output_dir.glob(f"detailed_results_{context_size}_*.csv"))

    if not detailed_files:
        return None

    # Get most recent
    most_recent = max(detailed_files, key=lambda p: p.stat().st_mtime)
    print(f"  Loading detailed results from: {most_recent.name}")

    df = pd.read_csv(most_recent)
    return df


def regenerate_ramp_graphs(output_dir: Path, summary_df: pd.DataFrame, max_ttft: float = 2.0):
    """Regenerate ramp graphs for each context size and cache hit rate"""
    print(f"\n{Colors.PHASE}Regenerating ramp graphs...{Colors.ENDC}")

    for context_size in summary_df['context_size'].unique():
        # Load detailed results for this context size
        detailed_df = load_detailed_results(output_dir, context_size)

        if detailed_df is None:
            print(f"  ⚠️  No detailed results found for context {context_size:,}")
            continue

        # Convert to RequestMetrics objects for each cache hit rate
        for cache_hit_rate in summary_df[summary_df['context_size'] == context_size]['cache_hit_rate'].unique():
            # Filter to this cache hit rate
            cache_df = detailed_df[detailed_df['cache_hit_rate'] == cache_hit_rate]

            if len(cache_df) == 0:
                continue

            # Get peak concurrency from summary
            peak_conc = summary_df[
                (summary_df['context_size'] == context_size) &
                (summary_df['cache_hit_rate'] == cache_hit_rate)
            ]['peak_concurrency'].iloc[0]

            # Convert DataFrame rows to RequestMetrics objects
            metrics = []
            for _, row in cache_df.iterrows():
                metric = RequestMetrics(
                    request_id=str(row['request_id']),
                    phase_id=str(row['phase_id']),
                    cache_hit_rate=int(row['cache_hit_rate']),
                    context_size=int(row['context_size']),
                    cached_tokens=int(row['cached_tokens']),
                    unique_tokens=int(row['unique_tokens']),
                    output_tokens=int(row['output_tokens']),
                    ttft=float(row['ttft']),
                    ttlt=float(row['ttlt']),
                    generation_time=float(row['generation_time']),
                    total_time=float(row['total_time']),
                    launch_time=float(row['launch_time']),
                    finish_time=float(row['finish_time']),
                    concurrency_level=int(row['concurrency_level']),
                    itl=float(row['itl'])
                )
                metrics.append(metric)

            # Generate ramp graph
            print(f"  Generating ramp graph for context={context_size:,}, cache_rate={cache_hit_rate}%")
            generate_ramp_graph(metrics, context_size, cache_hit_rate, max_ttft, peak_conc, str(output_dir))

    print(f"  ✓ Ramp graphs regenerated")


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate graphs from existing test output",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to output directory (e.g., cacherate_final-noprefix-s256-b32768-dp4-tp2-dram320)"
    )

    parser.add_argument(
        "--tool",
        type=str,
        choices=["cache_rate", "working_set"],
        default="cache_rate",
        help="Which tool's output (default: cache_rate)"
    )

    args = parser.parse_args()

    # Validate output directory
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Error: Output directory does not exist: {output_dir}")
        sys.exit(1)

    print(f"\n{Colors.HEADER}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}Regenerating Graphs{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}Output directory: {output_dir}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}Tool: {args.tool}_tester{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}\n")

    try:
        # Load summary data
        print(f"{Colors.PHASE}Loading summary data...{Colors.ENDC}")
        df = load_summary_csv(output_dir)

        # Convert to metrics objects
        if args.tool == "cache_rate":
            metrics = []
            for _, row in df.iterrows():
                metric = AggregatedMetrics(
                    context_size=int(row['context_size']),
                    cache_hit_rate=int(row['cache_hit_rate']),
                    input_tokens_per_sec=float(row['input_tokens_per_sec']),
                    output_tokens_per_sec=float(row['output_tokens_per_sec']),
                    avg_ttft=float(row['avg_ttft']),
                    median_ttft=float(row['median_ttft']),
                    p95_ttft=float(row['p95_ttft']),
                    p99_ttft=float(row['p99_ttft']),
                    avg_ttlt=float(row['avg_ttlt']),
                    median_ttlt=float(row['median_ttlt']),
                    p95_ttlt=float(row['p95_ttlt']),
                    p99_ttlt=float(row['p99_ttlt']),
                    avg_output_tokens=float(row['avg_output_tokens']),
                    avg_itl=float(row['avg_itl']),
                    median_itl=float(row['median_itl']),
                    p95_itl=float(row['p95_itl']),
                    p99_itl=float(row['p99_itl']),
                    peak_concurrency=int(row['peak_concurrency']),
                    total_requests=int(row['total_requests']),
                    test_duration=float(row['test_duration'])
                )
                metrics.append(metric)
        else:
            metrics = []
            for _, row in df.iterrows():
                metric = WSAggregatedMetrics(
                    context_size=int(row['context_size']),
                    working_set_size=int(row['working_set_size']),
                    cache_hit_rate=int(row['cache_hit_rate']),
                    input_tokens_per_sec=float(row['input_tokens_per_sec']),
                    output_tokens_per_sec=float(row['output_tokens_per_sec']),
                    avg_ttft=float(row['avg_ttft']),
                    median_ttft=float(row['median_ttft']),
                    p95_ttft=float(row['p95_ttft']),
                    p99_ttft=float(row['p99_ttft']),
                    avg_ttlt=float(row['avg_ttlt']),
                    median_ttlt=float(row['median_ttlt']),
                    p95_ttlt=float(row['p95_ttlt']),
                    p99_ttlt=float(row['p99_ttlt']),
                    avg_output_tokens=float(row['avg_output_tokens']),
                    avg_itl=float(row['avg_itl']),
                    median_itl=float(row['median_itl']),
                    p95_itl=float(row['p95_itl']),
                    p99_itl=float(row['p99_itl']),
                    peak_concurrency=int(row['peak_concurrency']),
                    total_requests=int(row['total_requests']),
                    test_duration=float(row['test_duration'])
                )
                metrics.append(metric)

        print(f"  Loaded {len(metrics)} test results")

        # Generate main comparison graphs
        print(f"\n{Colors.PHASE}Regenerating main comparison graphs...{Colors.ENDC}")
        if args.tool == "cache_rate":
            generate_cache_graphs(metrics, str(output_dir))
        else:
            generate_ws_graphs(metrics, str(output_dir))

        # Generate ramp graphs for cache_rate tests
        if args.tool == "cache_rate":
            regenerate_ramp_graphs(output_dir, df, max_ttft=2.0)

        print(f"\n{Colors.SUCCESS}{Colors.BOLD}{'='*80}{Colors.ENDC}")
        print(f"{Colors.SUCCESS}{Colors.BOLD}✓ All graphs regenerated successfully!{Colors.ENDC}")
        print(f"{Colors.SUCCESS}{Colors.BOLD}{'='*80}{Colors.ENDC}")
        print(f"{Colors.OKGREEN}Graphs saved to: {output_dir}{Colors.ENDC}")
        print(f"{Colors.OKGREEN}Main graphs now include peak RAMP in retry averages{Colors.ENDC}")
        if args.tool == "cache_rate":
            print(f"{Colors.OKGREEN}Ramp graphs now show ONLY ramp phase data (no retries){Colors.ENDC}")
        print()

    except Exception as e:
        print(f"\n{Colors.FAIL}Error: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
