#!/usr/bin/env python3
"""
Combined Graph Generator for Cache Rate Tester

This tool combines results from multiple cache_rate_tester.py output directories
into unified comparison graphs showing input/output throughput across different
test runs.

Usage:
    python combine_graphs.py \\
        --output-dirs run1/output run2/output run3/output \\
        --labels "Baseline" "Optimized" "Production" \\
        --output combined_output

The tool will generate:
- Combined input throughput graphs (one per context size)
- Combined output throughput graphs (one per context size)
- Speedup comparison graphs
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    INFO = '\033[97m'
    DEBUG = '\033[90m'
    METRIC = '\033[96m'
    SUCCESS = '\033[92m'
    PHASE = '\033[95m'


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""

    FORMATS = {
        logging.DEBUG: Colors.DEBUG + '[%(asctime)s] DEBUG - %(message)s' + Colors.ENDC,
        logging.INFO: Colors.INFO + '[%(asctime)s] INFO - %(message)s' + Colors.ENDC,
        logging.WARNING: Colors.WARNING + '[%(asctime)s] WARNING - %(message)s' + Colors.ENDC,
        logging.ERROR: Colors.FAIL + '[%(asctime)s] ERROR - %(message)s' + Colors.ENDC,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def init_logger(name: str, level=logging.INFO) -> logging.Logger:
    """Initialize logger with console handler"""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Console handler with colors
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(ColoredFormatter())
    logger.addHandler(console_handler)

    return logger


logger = init_logger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Combine cache_rate_tester.py results from multiple runs into unified comparison graphs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare three test runs with custom labels
  python combine_graphs.py \\
      --output-dirs baseline/output optimized/output final/output \\
      --labels "Baseline" "With Optimization" "Final Config" \\
      --output combined_results

  # Compare two runs with default labels
  python combine_graphs.py \\
      --output-dirs run1/output run2/output \\
      --output comparison
        """
    )

    parser.add_argument(
        "--output-dirs",
        type=str,
        nargs='+',
        required=True,
        help="Paths to cache_rate_tester.py output directories to combine (e.g., run1/output run2/output)"
    )

    parser.add_argument(
        "--labels",
        type=str,
        nargs='+',
        help="Labels for each output directory in the graphs. Must match the number of --output-dirs. "
             "If not specified, will use directory names."
    )

    parser.add_argument(
        "--output",
        type=str,
        default="./combined_output",
        help="Output directory for combined graphs (default: ./combined_output)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()


def load_summary_data(output_dir: Path) -> pd.DataFrame:
    """
    Load the most recent summary CSV from an output directory

    Returns:
        DataFrame with summary metrics
    """
    summary_files = list(output_dir.glob("summary_*.csv"))

    if not summary_files:
        raise FileNotFoundError(f"No summary_*.csv files found in {output_dir}")

    # Get most recent summary file
    most_recent = max(summary_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"  Loading {most_recent.name}")

    df = pd.read_csv(most_recent)
    return df


def load_and_calculate_metrics(output_dir: Path):
    """
    Load data and calculate retry-based metrics using the exact same logic as generate_graphs()

    This imports and calls the actual generate_graphs() logic from cache_rate_tester.py to ensure
    100% consistency between individual graphs and combined graphs.

    Returns:
        DataFrame with retry-based metrics calculated the same way as regenerate_graphs.py
    """
    # Import the generate_graphs function which does all the calculation
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from cache_rate_tester import load_existing_aggregated_results

    # Load summary data (for structure)
    summary_df = load_summary_data(output_dir)

    # The generate_graphs() function loads detailed results and phase metadata internally
    # and calculates retry-based metrics. We need to extract those calculated values.
    #
    # However, generate_graphs() generates HTML files, not return data.
    # So we need to load the detailed results and phase metadata and replicate
    # the exact calculation from generate_graphs().

    # Actually, let's directly use the calculation code from cache_rate_tester.py
    # by importing the necessary functions and data structures
    from cache_rate_tester import load_phase_metadata as crt_load_phase_metadata

    try:
        # Load phase metadata using cache_rate_tester's function
        phase_metadata_by_test = crt_load_phase_metadata(str(output_dir))

        # Load detailed results
        detailed_dfs = {}
        for context_size in summary_df['context_size'].unique():
            csv_files = list(output_dir.glob(f"detailed_results_{context_size}_*.csv"))
            if csv_files:
                most_recent = max(csv_files, key=lambda p: p.stat().st_mtime)
                detailed_dfs[context_size] = pd.read_csv(most_recent)

        if not phase_metadata_by_test or not detailed_dfs:
            logger.warning(f"  Missing phase metadata or detailed results, using summary values")
            return summary_df

        # Calculate retry-based metrics using EXACT same logic as generate_graphs()
        enhanced_data = []

        for _, row in summary_df.iterrows():
            context_size = int(row['context_size'])
            cache_rate = int(row['cache_hit_rate'])
            enhanced_row = row.to_dict()

            test_key = (context_size, cache_rate)
            if test_key in phase_metadata_by_test and context_size in detailed_dfs:
                phases = phase_metadata_by_test[test_key]
                detailed_df = detailed_dfs[context_size]
                peak_conc = int(row['peak_concurrency'])

                # EXACT filter logic from cache_rate_tester.py line 2095-2098
                retry_phases = [p for p in phases
                              if p.concurrency_level == peak_conc and
                                 (p.phase_type == "RETRY" or
                                  (p.phase_type == "RAMP" and p.concurrency_level == peak_conc))]

                if retry_phases:
                    # EXACT calculation logic from cache_rate_tester.py lines 2104-2148
                    retry_throughputs = {'input': [], 'output': []}
                    retry_ttft_stats = {'avg': [], 'median': [], 'p95': [], 'p99': []}

                    for phase in retry_phases:
                        phase_requests = detailed_df[
                            (detailed_df['phase_id'] == phase.phase_id) &
                            (detailed_df['cache_hit_rate'] == cache_rate) &
                            (detailed_df['concurrency_level'] == peak_conc)
                        ]

                        if len(phase_requests) > 0:
                            duration = phase.duration

                            total_input = (phase_requests['cached_tokens'] + phase_requests['unique_tokens']).sum()
                            total_output = phase_requests['output_tokens'].sum()

                            retry_throughputs['input'].append(total_input / duration)
                            retry_throughputs['output'].append(total_output / duration)

                            phase_ttfts = phase_requests['ttft'].values
                            if len(phase_ttfts) > 0:
                                retry_ttft_stats['avg'].append(np.mean(phase_ttfts))
                                retry_ttft_stats['median'].append(np.median(phase_ttfts))
                                retry_ttft_stats['p95'].append(np.percentile(phase_ttfts, 95))
                                retry_ttft_stats['p99'].append(np.percentile(phase_ttfts, 99))

                    # Update with calculated values
                    if retry_throughputs['input']:
                        enhanced_row['input_tokens_per_sec'] = np.mean(retry_throughputs['input'])
                        enhanced_row['output_tokens_per_sec'] = np.mean(retry_throughputs['output'])

                    if retry_ttft_stats['avg']:
                        enhanced_row['avg_ttft'] = np.mean(retry_ttft_stats['avg'])
                        enhanced_row['median_ttft'] = np.mean(retry_ttft_stats['median'])
                        enhanced_row['p95_ttft'] = np.mean(retry_ttft_stats['p95'])
                        enhanced_row['p99_ttft'] = np.mean(retry_ttft_stats['p99'])

            enhanced_data.append(enhanced_row)

        return pd.DataFrame(enhanced_data)

    except Exception as e:
        logger.warning(f"  Error calculating retry-based metrics: {e}")
        logger.warning(f"  Falling back to summary values")
        return summary_df


def load_all_data(output_dirs: List[Path], labels: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Load summary data from all output directories and enhance with retry-based metrics

    Uses the EXACT same calculation logic as regenerate_graphs.py to ensure consistency.

    Returns:
        Dictionary mapping label -> DataFrame
    """
    data = {}

    for output_dir, label in zip(output_dirs, labels):
        logger.info(f"Loading data for '{label}' from {output_dir}")
        try:
            # Load and calculate using the exact same logic as cache_rate_tester.py
            logger.info(f"  Calculating retry-based metrics (using cache_rate_tester.py logic)...")
            df = load_and_calculate_metrics(output_dir)

            data[label] = df
            logger.info(f"  ✓ Loaded {len(df)} test results")
        except Exception as e:
            logger.error(f"  Failed to load data: {e}")
            raise

    return data


def generate_combined_throughput_graphs(
    data: Dict[str, pd.DataFrame],
    output_dir: Path,
    metric: str = "input"
):
    """
    Generate combined throughput comparison graphs

    Args:
        data: Dictionary mapping label -> DataFrame
        output_dir: Output directory for graphs
        metric: "input" or "output" for which throughput metric to graph
    """
    metric_field = f"{metric}_tokens_per_sec"
    metric_title = "Input" if metric == "input" else "Output"

    # Get all unique context sizes across all datasets
    all_context_sizes = set()
    for df in data.values():
        all_context_sizes.update(df['context_size'].unique())

    all_context_sizes = sorted(all_context_sizes)

    # Define colors for different runs
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#FFA500', '#8B4513', '#2F4F4F']

    # Generate a graph for each context size
    for context_size in all_context_sizes:
        logger.info(f"Generating {metric} throughput graph for context size {context_size:,}")

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        color_idx = 0
        for label, df in data.items():
            # Filter to this context size
            df_ctx = df[df['context_size'] == context_size].sort_values('cache_hit_rate')

            if len(df_ctx) == 0:
                logger.warning(f"  No data for '{label}' at context size {context_size:,}")
                continue

            color = colors[color_idx % len(colors)]

            # Primary y-axis: Absolute throughput (solid line)
            fig.add_trace(
                go.Scatter(
                    x=df_ctx['cache_hit_rate'],
                    y=df_ctx[metric_field],
                    mode='lines+markers',
                    name=f'{label}',
                    line=dict(width=2, color=color),
                    marker=dict(size=6),
                    hovertemplate=f'{label}<br>Cache Rate: %{{x}}%<br>{metric_title}: %{{y:,.0f}} tok/s<extra></extra>'
                ),
                secondary_y=False
            )

            # Secondary y-axis: Speedup relative to 0% cache (dashed line)
            baseline = df_ctx[metric_field].iloc[0] if len(df_ctx) > 0 and df_ctx[metric_field].iloc[0] > 0 else 1
            speedup = df_ctx[metric_field] / baseline

            fig.add_trace(
                go.Scatter(
                    x=df_ctx['cache_hit_rate'],
                    y=speedup,
                    mode='lines+markers',
                    name=f'{label} (speedup)',
                    line=dict(width=2, dash='dash', color=color),
                    marker=dict(size=6, symbol='diamond'),
                    hovertemplate=f'{label}<br>Cache Rate: %{{x}}%<br>Speedup: %{{y:.2f}}x<extra></extra>'
                ),
                secondary_y=True
            )

            color_idx += 1

        # Update axes
        fig.update_xaxes(
            title_text="Cache Hit Rate (%)",
            range=[0, 100],
            dtick=10,
            gridcolor='lightgray',
            showgrid=True
        )
        fig.update_yaxes(
            title_text=f"{metric_title} Tokens/s",
            secondary_y=False
        )
        fig.update_yaxes(
            title_text="Speedup (relative to 0% cache)",
            secondary_y=True
        )

        # Update layout
        fig.update_layout(
            title=f"{metric_title} Throughput Comparison<br>Context: {context_size:,} tokens",
            hovermode='x unified',
            height=600,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        # Save graph
        filename = output_dir / f"{metric}_throughput_ctx{context_size}.html"
        fig.write_html(filename)
        logger.info(f"  Saved {filename.name}")


def generate_combined_ttft_graph(
    data: Dict[str, pd.DataFrame],
    output_dir: Path
):
    """
    Generate combined TTFT comparison graphs for each context size

    Args:
        data: Dictionary mapping label -> DataFrame
        output_dir: Output directory for graphs
    """
    # Get all unique context sizes across all datasets
    all_context_sizes = set()
    for df in data.values():
        all_context_sizes.update(df['context_size'].unique())

    all_context_sizes = sorted(all_context_sizes)

    # Define colors for different runs
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#FFA500', '#8B4513', '#2F4F4F']

    # Generate a graph for each context size
    for context_size in all_context_sizes:
        logger.info(f"Generating TTFT comparison graph for context size {context_size:,}")

        # Create subplot with 3 rows for different TTFT metrics
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                'Average TTFT',
                'P95 TTFT',
                'P99 TTFT'
            ),
            vertical_spacing=0.12,
            specs=[[{"secondary_y": False}],
                   [{"secondary_y": False}],
                   [{"secondary_y": False}]]
        )

        color_idx = 0
        for label, df in data.items():
            # Filter to this context size
            df_ctx = df[df['context_size'] == context_size].sort_values('cache_hit_rate')

            if len(df_ctx) == 0:
                logger.warning(f"  No data for '{label}' at context size {context_size:,}")
                continue

            color = colors[color_idx % len(colors)]

            # Row 1: Average TTFT
            fig.add_trace(
                go.Scatter(
                    x=df_ctx['cache_hit_rate'],
                    y=df_ctx['avg_ttft'],
                    mode='lines+markers',
                    name=f'{label}',
                    line=dict(width=2, color=color),
                    marker=dict(size=6),
                    hovertemplate=f'{label}<br>Cache Rate: %{{x}}%<br>Avg TTFT: %{{y:.3f}}s<extra></extra>',
                    legendgroup=label
                ),
                row=1, col=1
            )

            # Row 2: P95 TTFT
            fig.add_trace(
                go.Scatter(
                    x=df_ctx['cache_hit_rate'],
                    y=df_ctx['p95_ttft'],
                    mode='lines+markers',
                    name=f'{label}',
                    line=dict(width=2, color=color),
                    marker=dict(size=6),
                    hovertemplate=f'{label}<br>Cache Rate: %{{x}}%<br>P95 TTFT: %{{y:.3f}}s<extra></extra>',
                    legendgroup=label,
                    showlegend=False
                ),
                row=2, col=1
            )

            # Row 3: P99 TTFT
            fig.add_trace(
                go.Scatter(
                    x=df_ctx['cache_hit_rate'],
                    y=df_ctx['p99_ttft'],
                    mode='lines+markers',
                    name=f'{label}',
                    line=dict(width=2, color=color),
                    marker=dict(size=6),
                    hovertemplate=f'{label}<br>Cache Rate: %{{x}}%<br>P99 TTFT: %{{y:.3f}}s<extra></extra>',
                    legendgroup=label,
                    showlegend=False
                ),
                row=3, col=1
            )

            color_idx += 1

        # Update axes for all rows
        for row in [1, 2, 3]:
            fig.update_xaxes(
                title_text="Cache Hit Rate (%)",
                row=row, col=1,
                range=[0, 100],
                dtick=10,
                gridcolor='lightgray',
                showgrid=True
            )

        fig.update_yaxes(title_text="Average TTFT (seconds)", row=1, col=1)
        fig.update_yaxes(title_text="P95 TTFT (seconds)", row=2, col=1)
        fig.update_yaxes(title_text="P99 TTFT (seconds)", row=3, col=1)

        # Update layout
        fig.update_layout(
            title=f"TTFT Comparison<br>Context: {context_size:,} tokens",
            height=900,
            hovermode='x unified',
            showlegend=True
        )

        # Save graph
        filename = output_dir / f"ttft_comparison_ctx{context_size}.html"
        fig.write_html(filename)
        logger.info(f"  Saved {filename.name}")


def generate_summary_table(
    data: Dict[str, pd.DataFrame],
    output_dir: Path
):
    """
    Generate a summary comparison table showing key metrics

    Args:
        data: Dictionary mapping label -> DataFrame
        output_dir: Output directory for table
    """
    logger.info("Generating summary comparison table")

    # Get all unique context sizes
    all_context_sizes = set()
    for df in data.values():
        all_context_sizes.update(df['context_size'].unique())

    all_context_sizes = sorted(all_context_sizes)

    # Create summary data
    summary_rows = []

    for label, df in data.items():
        for context_size in all_context_sizes:
            df_ctx = df[df['context_size'] == context_size].sort_values('cache_hit_rate')

            if len(df_ctx) == 0:
                continue

            # Get 0% and 100% cache metrics
            df_0 = df_ctx[df_ctx['cache_hit_rate'] == 0]
            df_100 = df_ctx[df_ctx['cache_hit_rate'] == 100]

            if len(df_0) > 0 and len(df_100) > 0:
                input_speedup = df_100['input_tokens_per_sec'].iloc[0] / df_0['input_tokens_per_sec'].iloc[0]
                output_speedup = df_100['output_tokens_per_sec'].iloc[0] / df_0['output_tokens_per_sec'].iloc[0]
                ttft_improvement = (df_0['avg_ttft'].iloc[0] - df_100['avg_ttft'].iloc[0]) / df_0['avg_ttft'].iloc[0] * 100

                summary_rows.append({
                    'Configuration': label,
                    'Context Size': f"{context_size:,}",
                    'Input @0% (tok/s)': f"{df_0['input_tokens_per_sec'].iloc[0]:,.0f}",
                    'Input @100% (tok/s)': f"{df_100['input_tokens_per_sec'].iloc[0]:,.0f}",
                    'Input Speedup': f"{input_speedup:.2f}x",
                    'Output @0% (tok/s)': f"{df_0['output_tokens_per_sec'].iloc[0]:,.0f}",
                    'Output @100% (tok/s)': f"{df_100['output_tokens_per_sec'].iloc[0]:,.0f}",
                    'Output Speedup': f"{output_speedup:.2f}x",
                    'TTFT Improvement': f"{ttft_improvement:.1f}%",
                    'Peak Concurrency @100%': f"{df_100['peak_concurrency'].iloc[0]}"
                })

    # Create DataFrame and save as CSV and HTML
    summary_df = pd.DataFrame(summary_rows)

    # Save as CSV
    csv_file = output_dir / "summary_comparison.csv"
    summary_df.to_csv(csv_file, index=False)
    logger.info(f"  Saved {csv_file.name}")

    # Create HTML table
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Cache Rate Tester - Combined Summary</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        tr:hover {{
            background-color: #e8f5e9;
        }}
        .metric {{
            font-family: monospace;
        }}
    </style>
</head>
<body>
    <h1>Cache Rate Tester - Combined Summary</h1>
    <p>Comparison of {len(data)} test configurations across {len(all_context_sizes)} context size(s)</p>
    {summary_df.to_html(index=False, classes='metric', escape=False)}
</body>
</html>
"""

    html_file = output_dir / "summary_comparison.html"
    with open(html_file, 'w') as f:
        f.write(html_content)
    logger.info(f"  Saved {html_file.name}")


def generate_index_html(output_dir: Path, labels: List[str]):
    """
    Generate an index.html dashboard linking to all generated graphs

    Args:
        output_dir: Output directory containing graphs
        labels: List of configuration labels
    """
    logger.info("Generating index.html dashboard")

    # Find all generated graphs
    input_graphs = sorted(output_dir.glob("input_throughput_ctx*.html"))
    output_graphs = sorted(output_dir.glob("output_throughput_ctx*.html"))
    ttft_graphs = sorted(output_dir.glob("ttft_comparison_ctx*.html"))

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Combined Cache Rate Tester Results</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
        }}
        h2 {{
            color: #555;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 5px;
        }}
        .config-list {{
            background-color: white;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .config-item {{
            padding: 5px 0;
            font-weight: bold;
            color: #4CAF50;
        }}
        .graph-section {{
            background-color: white;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .graph-link {{
            display: block;
            padding: 10px;
            margin: 5px 0;
            background-color: #4CAF50;
            color: white;
            text-decoration: none;
            border-radius: 3px;
            transition: background-color 0.3s;
        }}
        .graph-link:hover {{
            background-color: #45a049;
        }}
        ul {{
            list-style-type: none;
            padding: 0;
        }}
        li {{
            margin: 5px 0;
        }}
    </style>
</head>
<body>
    <h1>Combined Cache Rate Tester Results</h1>

    <div class="config-list">
        <h2>Configurations Compared</h2>
        <ul>
"""

    for i, label in enumerate(labels, 1):
        html_content += f"            <li class=\"config-item\">{i}. {label}</li>\n"

    html_content += """
        </ul>
    </div>

    <div class="graph-section">
        <h2>Summary</h2>
        <ul>
            <li><a class="graph-link" href="summary_comparison.html">📊 Summary Comparison Table</a></li>
        </ul>
    </div>

    <div class="graph-section">
        <h2>Input Throughput Comparisons</h2>
        <ul>
"""

    for graph in input_graphs:
        # Extract context size from filename
        ctx_str = graph.stem.replace("input_throughput_ctx", "")
        html_content += f"            <li><a class=\"graph-link\" href=\"{graph.name}\">📈 Context {ctx_str} tokens - Input Throughput</a></li>\n"

    html_content += """
        </ul>
    </div>

    <div class="graph-section">
        <h2>Output Throughput Comparisons</h2>
        <ul>
"""

    for graph in output_graphs:
        # Extract context size from filename
        ctx_str = graph.stem.replace("output_throughput_ctx", "")
        html_content += f"            <li><a class=\"graph-link\" href=\"{graph.name}\">📈 Context {ctx_str} tokens - Output Throughput</a></li>\n"

    html_content += """
        </ul>
    </div>

    <div class="graph-section">
        <h2>TTFT Comparisons</h2>
        <ul>
"""

    for graph in ttft_graphs:
        # Extract context size from filename
        ctx_str = graph.stem.replace("ttft_comparison_ctx", "")
        html_content += f"            <li><a class=\"graph-link\" href=\"{graph.name}\">⏱️ Context {ctx_str} tokens - TTFT Metrics</a></li>\n"

    html_content += """
        </ul>
    </div>
</body>
</html>
"""

    index_file = output_dir / "index.html"
    with open(index_file, 'w') as f:
        f.write(html_content)
    logger.info(f"  Saved {index_file.name}")


def main():
    """Main entry point"""
    args = parse_arguments()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Convert paths
    output_dirs = [Path(d) for d in args.output_dirs]

    # Validate output directories exist
    for output_dir in output_dirs:
        if not output_dir.exists():
            logger.error(f"Output directory does not exist: {output_dir}")
            sys.exit(1)

    # Generate labels if not provided
    if args.labels:
        if len(args.labels) != len(output_dirs):
            logger.error(f"Number of labels ({len(args.labels)}) must match number of output directories ({len(output_dirs)})")
            sys.exit(1)
        labels = args.labels
    else:
        # Use directory names as labels
        labels = [d.parent.name if d.name == 'output' else d.name for d in output_dirs]
        logger.info(f"Using auto-generated labels: {labels}")

    logger.info(f"{Colors.HEADER}{'='*80}{Colors.ENDC}")
    logger.info(f"{Colors.HEADER}{Colors.BOLD}Combined Graph Generator{Colors.ENDC}")
    logger.info(f"{Colors.HEADER}{'='*80}{Colors.ENDC}")
    logger.info(f"{Colors.OKBLUE}Combining {len(output_dirs)} test runs{Colors.ENDC}")
    logger.info(f"{Colors.OKBLUE}Output directory: {args.output}{Colors.ENDC}")
    logger.info(f"{Colors.HEADER}{'='*80}{Colors.ENDC}")

    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load all data
    logger.info("")
    logger.info(f"{Colors.PHASE}Loading data from output directories...{Colors.ENDC}")
    data = load_all_data(output_dirs, labels)

    # Generate graphs
    logger.info("")
    logger.info(f"{Colors.PHASE}Generating combined input throughput graphs...{Colors.ENDC}")
    generate_combined_throughput_graphs(data, output_path, metric="input")

    logger.info("")
    logger.info(f"{Colors.PHASE}Generating combined output throughput graphs...{Colors.ENDC}")
    generate_combined_throughput_graphs(data, output_path, metric="output")

    logger.info("")
    logger.info(f"{Colors.PHASE}Generating combined TTFT comparison graphs...{Colors.ENDC}")
    generate_combined_ttft_graph(data, output_path)

    logger.info("")
    logger.info(f"{Colors.PHASE}Generating summary comparison table...{Colors.ENDC}")
    generate_summary_table(data, output_path)

    logger.info("")
    logger.info(f"{Colors.PHASE}Generating index.html dashboard...{Colors.ENDC}")
    generate_index_html(output_path, labels)

    logger.info("")
    logger.info(f"{Colors.SUCCESS}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    logger.info(f"{Colors.SUCCESS}{Colors.BOLD}✓ Combined graphs generated successfully!{Colors.ENDC}")
    logger.info(f"{Colors.SUCCESS}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    logger.info(f"{Colors.OKGREEN}Results saved to: {output_path}{Colors.ENDC}")
    logger.info(f"{Colors.OKGREEN}Open {output_path}/index.html to view all graphs{Colors.ENDC}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
