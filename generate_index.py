#!/usr/bin/env python3
"""
Generate index.html dashboard for cache rate test results
"""

from pathlib import Path
from datetime import datetime
import pandas as pd
import json
from typing import List, Dict


def format_bytes(bytes_size: int) -> str:
    """Format bytes as human-readable string"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


def estimate_kv_cache_size(model_name: str, num_tokens: int, precision: str = 'fp8') -> int:
    """Estimate KV cache size"""
    KV_CACHE_SIZES_FP8 = {
        'llama-3.3-70b': 160_000,
        'llama-3.1-70b': 160_000,
        'qwen3-coder-30b': 60_000,
        'qwen2.5-coder-32b': 60_000,
        'qwen3-coder-480b': 124_000,
    }

    model_lower = model_name.lower()
    for known_model, size_fp8 in KV_CACHE_SIZES_FP8.items():
        if known_model in model_lower:
            size_per_token = size_fp8 if precision == 'fp8' else size_fp8 * 2
            return num_tokens * size_per_token

    # Default conservative estimate
    size_per_token = 160_000 if precision == 'fp8' else 320_000
    return num_tokens * size_per_token


def generate_index_html(output_dir: str, version: str = "1.0"):
    """Generate index.html dashboard"""
    output_path = Path(output_dir)

    # Load summary CSV
    summary_files = list(output_path.glob("summary_*.csv"))
    if not summary_files:
        print(f"No summary files found in {output_dir}")
        return

    # Get most recent summary
    summary_file = max(summary_files, key=lambda p: p.stat().st_mtime)
    df = pd.read_csv(summary_file)

    # Load progress.json if available
    progress_file = output_path / "progress.json"
    config_data = {}
    model = "Unknown"
    working_set_size = 0

    if progress_file.exists():
        with open(progress_file, 'r') as f:
            progress = json.load(f)
            config_data = progress.get('parameters', {})
            model = config_data.get('detected_model', 'Unknown')
            working_set_size = config_data.get('working_set_size', 0)

    # Calculate stats
    total_requests = int(df['total_requests'].sum())
    context_sizes = sorted(df['context_size'].unique())
    cache_hit_rates = sorted(df['cache_hit_rate'].unique())

    # Calculate total tokens processed across all tests
    # Total tokens = throughput * duration for each test, then sum
    total_input_tokens = int((df['input_tokens_per_sec'] * df['test_duration']).sum())
    total_output_tokens = int((df['output_tokens_per_sec'] * df['test_duration']).sum())

    # Find best performance
    best_input_idx = df['input_tokens_per_sec'].idxmax()
    best_output_idx = df['output_tokens_per_sec'].idxmax()
    best_ttft_idx = df['avg_ttft'].idxmin()

    best_input = df.loc[best_input_idx]
    best_output = df.loc[best_output_idx]
    best_ttft = df.loc[best_ttft_idx]

    # Estimate KV cache
    kv_cache_fp8 = estimate_kv_cache_size(model, working_set_size, 'fp8')
    kv_cache_fp16 = estimate_kv_cache_size(model, working_set_size, 'fp16')

    # Find graph files
    graph_files = {}
    for html_file in output_path.glob("*.html"):
        if html_file.name != "index.html":
            graph_files[html_file.name] = html_file.name

    # Generate HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cache Rate Test Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #95a5a6;
            padding-bottom: 8px;
        }}
        .card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-box {{
            background: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #3498db;
        }}
        .stat-label {{
            font-size: 0.9em;
            color: #7f8c8d;
            margin-bottom: 5px;
        }}
        .stat-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #2c3e50;
        }}
        .graph-link {{
            display: block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 20px;
            margin: 10px 0;
            border-radius: 5px;
            text-decoration: none;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .graph-link:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }}
        .graph-link h3 {{
            margin: 0 0 5px 0;
            font-size: 1.2em;
        }}
        .graph-link p {{
            margin: 0;
            opacity: 0.9;
            font-size: 0.9em;
        }}
        .highlight {{
            background: #e8f5e9;
            border-left: 4px solid #4caf50;
        }}
        .config-table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .config-table td {{
            padding: 8px;
            border-bottom: 1px solid #e0e0e0;
        }}
        .config-table td:first-child {{
            font-weight: bold;
            color: #555;
            width: 200px;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #7f8c8d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <h1>🚀 Cache Rate Performance Test Results</h1>

    <div class="card">
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Version:</strong> {version}</p>
    </div>

    <h2>📊 Test Configuration</h2>
    <div class="card">
        <table class="config-table">
            <tr>
                <td>API Endpoint</td>
                <td><code>{config_data.get('api_endpoint', 'N/A')}</code></td>
            </tr>
            <tr>
                <td>Detected Model</td>
                <td><strong>{model}</strong></td>
            </tr>
            <tr>
                <td>Context Sizes</td>
                <td>{', '.join(f'{size:,}' for size in context_sizes)} tokens</td>
            </tr>
            <tr>
                <td>Cache Hit Rates Tested</td>
                <td>{len(cache_hit_rates)} rates: {min(cache_hit_rates)}% to {max(cache_hit_rates)}%</td>
            </tr>
            <tr>
                <td>Working Set Size</td>
                <td>{working_set_size:,} tokens</td>
            </tr>
            <tr>
                <td>Output Tokens</td>
                <td>{config_data.get('output_tokens', 'N/A')} per request</td>
            </tr>
            <tr>
                <td>Max TTFT Threshold</td>
                <td>{config_data.get('max_ttft', 'N/A')} seconds</td>
            </tr>
            <tr>
                <td>Test Duration</td>
                <td>{config_data.get('test_duration', 'N/A')} seconds per cache rate</td>
            </tr>
            <tr>
                <td>Tokenizer</td>
                <td><code>{config_data.get('tokenizer_id', 'N/A')}</code></td>
            </tr>
        </table>
    </div>

    <h2>💾 KV Cache Estimates</h2>
    <div class="card">
        <div class="stats-grid">
            <div class="stat-box">
                <div class="stat-label">FP8 Precision</div>
                <div class="stat-value">{format_bytes(kv_cache_fp8)}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">FP16 Precision</div>
                <div class="stat-value">{format_bytes(kv_cache_fp16)}</div>
            </div>
        </div>
    </div>

    <h2>📈 Overall Statistics</h2>
    <div class="card">
        <div class="stats-grid">
            <div class="stat-box">
                <div class="stat-label">Total Requests</div>
                <div class="stat-value">{total_requests:,}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Total Input Tokens</div>
                <div class="stat-value">{total_input_tokens:,}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Total Output Tokens</div>
                <div class="stat-value">{total_output_tokens:,}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Tests Completed</div>
                <div class="stat-value">{len(df)}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Context Sizes Tested</div>
                <div class="stat-value">{len(context_sizes)}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Cache Hit Rates</div>
                <div class="stat-value">{len(cache_hit_rates)}</div>
            </div>
        </div>
    </div>

    <h2>🏆 Best Performance</h2>
    <div class="card">
        <div class="stats-grid">
            <div class="stat-box highlight">
                <div class="stat-label">Peak Input Throughput</div>
                <div class="stat-value">{best_input['input_tokens_per_sec']:,.0f} tok/s</div>
                <div class="stat-label" style="margin-top: 5px;">
                    {int(best_input['cache_hit_rate'])}% cache, {int(best_input['context_size']):,} tokens
                </div>
            </div>
            <div class="stat-box highlight">
                <div class="stat-label">Peak Output Throughput</div>
                <div class="stat-value">{best_output['output_tokens_per_sec']:,.0f} tok/s</div>
                <div class="stat-label" style="margin-top: 5px;">
                    {int(best_output['cache_hit_rate'])}% cache, {int(best_output['context_size']):,} tokens
                </div>
            </div>
            <div class="stat-box highlight">
                <div class="stat-label">Best TTFT</div>
                <div class="stat-value">{best_ttft['avg_ttft']:.3f}s</div>
                <div class="stat-label" style="margin-top: 5px;">
                    {int(best_ttft['cache_hit_rate'])}% cache, {int(best_ttft['context_size']):,} tokens
                </div>
            </div>
        </div>
    </div>

    <h2>📊 Interactive Visualizations</h2>
    <div class="card">
"""

    # Group ramp analysis graphs by context size
    ramp_graphs_by_context = {}
    comparison_graphs = []

    for graph_name in sorted(graph_files.keys()):
        if 'ramp_' in graph_name:
            # Handle both naming schemes:
            # cache_rate_tester: ramp_ctx{context}_cache{cache}.html
            # working_set_tester: ramp_ctx{context}_ws{working_set}_cache{cache}.html
            parts = graph_name.replace('ramp_ctx', '').replace('.html', '')

            # Check if it has _ws (working set tester format)
            if '_ws' in parts:
                # Format: {context}_ws{working_set}_cache{cache}
                ctx_and_ws, cache = parts.split('_cache')
                ctx, ws = ctx_and_ws.split('_ws')
                if ctx not in ramp_graphs_by_context:
                    ramp_graphs_by_context[ctx] = []
                ramp_graphs_by_context[ctx].append((f"WS:{ws}, Cache:{cache}", graph_name))
            else:
                # Format: {context}_cache{cache}
                ctx, cache = parts.split('_cache')
                if ctx not in ramp_graphs_by_context:
                    ramp_graphs_by_context[ctx] = []
                ramp_graphs_by_context[ctx].append((f"Cache:{cache}", graph_name))
        else:
            comparison_graphs.append(graph_name)

    # Add comparison and heatmap graphs first
    for graph_name in sorted(comparison_graphs):
        if 'performance_vs_cache' in graph_name:
            context = graph_name.split('_')[-1].replace('.html', '')
            html_content += f"""
        <a href="{graph_name}" class="graph-link">
            <h3>📈 Performance vs Cache Hit Rate ({context})</h3>
            <p>Input/Output throughput and TTFT across different cache hit rates</p>
        </a>
"""
        elif 'performance_vs_working_set' in graph_name:
            # Format: performance_vs_working_set_{context}_cache{cache}.html
            parts = graph_name.replace('performance_vs_working_set_', '').replace('.html', '').split('_cache')
            if len(parts) == 2:
                context, cache = parts
                html_content += f"""
        <a href="{graph_name}" class="graph-link">
            <h3>📈 Performance vs Working Set Size (Context: {context}, Cache: {cache}%)</h3>
            <p>Input/Output throughput and TTFT across different working set sizes</p>
        </a>
"""
        elif 'cache_comparison_ctx' in graph_name:
            # Format: cache_comparison_ctx{context}.html
            context = graph_name.replace('cache_comparison_ctx', '').replace('.html', '')
            html_content += f"""
        <a href="{graph_name}" class="graph-link">
            <h3>🔄 Cache Hit Rate Comparison (Context: {context})</h3>
            <p>Compare all cache hit rates across working set sizes - Input/Output/TTFT/Decode metrics</p>
        </a>
"""
        elif 'input_throughput_comparison' in graph_name:
            # Check if it has cache suffix (working_set_tester format)
            cache_suffix = ""
            if '_cache' in graph_name:
                cache = graph_name.split('_cache')[1].replace('.html', '')
                cache_suffix = f" (Cache: {cache}%)"
            html_content += f"""
        <a href="{graph_name}" class="graph-link">
            <h3>📊 Input Throughput Comparison{cache_suffix}</h3>
            <p>Compare input throughput across all context sizes</p>
        </a>
"""
        elif 'output_throughput_comparison' in graph_name:
            # Check if it has cache suffix (working_set_tester format)
            cache_suffix = ""
            if '_cache' in graph_name:
                cache = graph_name.split('_cache')[1].replace('.html', '')
                cache_suffix = f" (Cache: {cache}%)"
            html_content += f"""
        <a href="{graph_name}" class="graph-link">
            <h3>📊 Output Throughput Comparison{cache_suffix}</h3>
            <p>Compare output throughput across all context sizes</p>
        </a>
"""
        elif 'output_metrics_comparison' in graph_name:
            # Check if it has cache suffix (working_set_tester format)
            cache_suffix = ""
            if '_cache' in graph_name:
                cache = graph_name.split('_cache')[1].replace('.html', '')
                cache_suffix = f" (Cache: {cache}%)"
            html_content += f"""
        <a href="{graph_name}" class="graph-link">
            <h3>📊 Output Token Metrics Comparison{cache_suffix}</h3>
            <p>Inter-token latency and generation speed per request across context sizes</p>
        </a>
"""
        elif 'ttft_heatmap' in graph_name:
            html_content += f"""
        <a href="{graph_name}" class="graph-link">
            <h3>🔥 TTFT Heatmap</h3>
            <p>2D visualization of TTFT across context sizes and cache hit rates</p>
        </a>
"""

    # Now add grouped ramp analysis
    if ramp_graphs_by_context:
        html_content += """
        <h3 style="margin-top: 30px; color: #34495e;">Concurrency Ramp Analysis by Context Size</h3>
"""
        for ctx in sorted(ramp_graphs_by_context.keys(), key=lambda x: int(x)):
            html_content += f"""
        <details style="margin: 15px 0;">
            <summary style="cursor: pointer; padding: 10px; background: #ecf0f1; border-radius: 5px; font-weight: bold;">
                📊 Context: {ctx} tokens ({len(ramp_graphs_by_context[ctx])} tests)
            </summary>
            <div style="padding: 10px 0; padding-left: 20px;">
"""
            for label, graph_name in sorted(ramp_graphs_by_context[ctx]):
                html_content += f"""
                <a href="{graph_name}" class="graph-link" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); margin: 5px 0;">
                    <h4 style="margin: 0;">{label}</h4>
                    <p style="margin: 5px 0 0 0;">Throughput and TTFT vs concurrency level</p>
                </a>
"""
            html_content += """
            </div>
        </details>
"""

    html_content += f"""
    </div>

    <h2>📁 Data Files</h2>
    <div class="card">
        <p>Raw data is available in CSV format:</p>
        <ul>
"""

    # List CSV files
    for csv_file in sorted(output_path.glob("*.csv")):
        file_size = csv_file.stat().st_size
        html_content += f"""
            <li><code>{csv_file.name}</code> ({format_bytes(file_size)})</li>
"""

    html_content += f"""
        </ul>
    </div>

    <h2>ℹ️ Test Notes</h2>
    <div class="card">
        <ul>
            <li><strong>Token Diversity:</strong> Uses 100+ diverse tokens across 5 categories for better MoE expert activation</li>
            <li><strong>Concurrency Strategy:</strong> Ramped until TTFT exceeded {config_data.get('max_ttft', 'N/A')}s threshold</li>
            <li><strong>Retry Logic:</strong> Only retried at peak concurrency ({config_data.get('num_retries', 3)} times) to save time</li>
            <li><strong>Time Boxing:</strong> Each cache hit rate test limited to {config_data.get('test_duration', 180)} seconds</li>
        </ul>
    </div>

    <div class="footer">
        <p>Generated with Cache Rate Tester v{version}</p>
        <p>Powered by <a href="https://claude.ai/code" target="_blank">Claude Code</a></p>
    </div>
</body>
</html>
"""

    # Write index.html
    index_file = output_path / "index.html"
    with open(index_file, 'w') as f:
        f.write(html_content)

    print(f"✓ Generated index dashboard: {index_file}")


if __name__ == "__main__":
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "./output"
    version = sys.argv[2] if len(sys.argv) > 2 else "1.0"
    generate_index_html(output_dir, version)
