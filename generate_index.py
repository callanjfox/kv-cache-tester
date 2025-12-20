#!/usr/bin/env python3
"""
Unified index.html generator for all kv-cache-tester tools.

Auto-detects test type and generates appropriate dashboard:
- single_prompt: Cold start vs cached performance
- cache_rate: Performance across cache hit rates
- working_set: Performance across working set sizes
- sustained: Continuous mode with adaptive concurrency
- combined: Multi-run comparison

Usage:
    python generate_index.py <output_dir> [version]
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
import json


def format_bytes(bytes_size: int) -> str:
    """Format bytes as human-readable string"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


def format_number(n: float, decimals: int = 0) -> str:
    """Format number with commas"""
    if decimals == 0:
        return f"{int(n):,}"
    return f"{n:,.{decimals}f}"


def detect_test_type(output_path: Path) -> str:
    """Auto-detect test type from output files"""
    files = list(output_path.glob("*"))
    filenames = [f.name for f in files]

    # Check for combined graphs (from combine_graphs.py)
    if any("input_throughput_ctx" in f and "comparison" not in f for f in filenames):
        if any("summary_comparison" in f for f in filenames):
            return "combined"

    # Check for single_prompt results
    if any("single_prompt_performance" in f for f in filenames):
        return "single_prompt"

    # Check for working_set vs cache_rate based on graph names
    # These take priority over sustained detection since cache_rate/working_set
    # tools can run in sustained mode but still produce these graphs
    if any("performance_vs_working_set" in f for f in filenames):
        return "working_set"

    if any("performance_vs_cache" in f for f in filenames):
        return "cache_rate"

    # Check for sustained/continuous mode (only if no performance graphs)
    if any("sustained_periods" in f for f in filenames):
        return "sustained"

    # Check summary CSV columns as fallback
    summary_files = list(output_path.glob("summary_*.csv"))
    if summary_files:
        df = pd.read_csv(summary_files[0], nrows=1)
        cols = df.columns.tolist()
        # working_set has varying working_set_size column with different values
        if 'working_set_size' in cols:
            if df['working_set_size'].nunique() > 1 if len(df) > 1 else True:
                return "working_set"

    # Default to cache_rate
    return "cache_rate"


def load_config(output_path: Path) -> Dict[str, Any]:
    """Load configuration from progress.json, metadata.json, or test_metadata.json"""
    config = {}

    # Try progress.json first (existing format from cache_rate/working_set testers)
    progress_file = output_path / "progress.json"
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            progress = json.load(f)
            config = progress.get('parameters', {})
            config['_source'] = 'progress.json'
            # Flatten api_endpoints to api_endpoint
            if 'api_endpoints' in config and isinstance(config['api_endpoints'], list):
                config['api_endpoint'] = config['api_endpoints'][0]

    # Try test_metadata.json (has model info from cache_rate/working_set)
    test_metadata_file = output_path / "test_metadata.json"
    if test_metadata_file.exists():
        with open(test_metadata_file, 'r') as f:
            test_meta = json.load(f)
            if 'model' in test_meta and not config.get('detected_model'):
                config['detected_model'] = test_meta['model']

    # Try metadata.json (new unified format from single_prompt)
    metadata_file = output_path / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            config.update(metadata)
            config['_source'] = 'metadata.json'

    return config


def load_summary_data(output_path: Path) -> Optional[pd.DataFrame]:
    """Load summary CSV data"""
    summary_files = list(output_path.glob("summary_*.csv"))
    if not summary_files:
        return None

    # Get most recent
    summary_file = max(summary_files, key=lambda p: p.stat().st_mtime)
    return pd.read_csv(summary_file)


def get_graph_files(output_path: Path) -> Dict[str, List[Path]]:
    """Categorize graph files by type"""
    graphs = {
        'performance': [],
        'comparison': [],
        'fixed': [],
        'sustained': [],
        'heatmap': [],
        'single_prompt': [],
        'other': []
    }

    for html_file in output_path.glob("*.html"):
        if html_file.name == "index.html":
            continue

        name = html_file.name
        if 'performance_vs_cache' in name or 'performance_vs_working_set' in name:
            graphs['performance'].append(html_file)
        elif 'comparison' in name or 'throughput_ctx' in name or 'ttft_comparison' in name:
            graphs['comparison'].append(html_file)
        elif 'fixed_' in name or 'ramp_' in name:
            graphs['fixed'].append(html_file)
        elif 'sustained' in name:
            graphs['sustained'].append(html_file)
        elif 'heatmap' in name:
            graphs['heatmap'].append(html_file)
        elif 'single_prompt' in name or 'summary_table' in name:
            graphs['single_prompt'].append(html_file)
        else:
            graphs['other'].append(html_file)

    # Sort each category
    for key in graphs:
        graphs[key] = sorted(graphs[key])

    return graphs


def estimate_kv_cache_size(model_name: str, num_tokens: int, precision: str = 'fp8') -> int:
    """Estimate KV cache size based on model"""
    KV_CACHE_SIZES_FP8 = {
        'llama-3.3-70b': 160_000,
        'llama-3.1-70b': 160_000,
        'llama-3.3-8b': 32_000,
        'llama-3.1-8b': 32_000,
        'qwen2.5-coder-32b': 60_000,
    }

    model_lower = model_name.lower()
    for known_model, size_fp8 in KV_CACHE_SIZES_FP8.items():
        if known_model in model_lower:
            size_per_token = size_fp8 if precision == 'fp8' else size_fp8 * 2
            return num_tokens * size_per_token

    # Default estimate
    size_per_token = 100_000 if precision == 'fp8' else 200_000
    return num_tokens * size_per_token


# ============================================================================
# HTML Templates
# ============================================================================

CSS_STYLES = """
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .header h1 { margin: 0 0 10px 0; font-size: 2em; }
        .header p { margin: 5px 0; opacity: 0.9; }
        h2 {
            color: #2c3e50;
            margin-top: 30px;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        .card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .stat-box {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #3498db;
        }
        .stat-box.highlight { border-left-color: #27ae60; background: #e8f5e9; }
        .stat-label { font-size: 0.85em; color: #7f8c8d; margin-bottom: 5px; }
        .stat-value { font-size: 1.4em; font-weight: bold; color: #2c3e50; }
        .stat-detail { font-size: 0.8em; color: #95a5a6; margin-top: 4px; }
        .graph-link {
            display: block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 20px;
            margin: 10px 0;
            border-radius: 6px;
            text-decoration: none;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .graph-link:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }
        .graph-link h3 { margin: 0 0 5px 0; font-size: 1.1em; }
        .graph-link p { margin: 0; opacity: 0.85; font-size: 0.9em; }
        .graph-link.secondary { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
        .graph-link.tertiary { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }
        .config-table { width: 100%; border-collapse: collapse; }
        .config-table td { padding: 10px; border-bottom: 1px solid #ecf0f1; }
        .config-table td:first-child { font-weight: 600; color: #555; width: 220px; }
        details { margin: 15px 0; }
        summary {
            cursor: pointer;
            padding: 12px;
            background: #ecf0f1;
            border-radius: 6px;
            font-weight: bold;
            color: #2c3e50;
        }
        summary:hover { background: #e0e0e0; }
        details[open] summary { border-radius: 6px 6px 0 0; }
        details > div { padding: 15px; background: #fafafa; border-radius: 0 0 6px 6px; }
        .csv-list { list-style: none; padding: 0; }
        .csv-list li { padding: 8px 0; border-bottom: 1px solid #eee; }
        .csv-list a { color: #e67e22; text-decoration: none; }
        .csv-list a:hover { text-decoration: underline; }
        .footer {
            margin-top: 40px;
            padding: 20px;
            text-align: center;
            color: #7f8c8d;
            font-size: 0.9em;
            border-top: 1px solid #ddd;
        }
        .info-box {
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 15px;
            margin: 15px 0;
            border-radius: 0 6px 6px 0;
        }
"""


def generate_header(test_type: str, config: Dict, timestamp: str) -> str:
    """Generate page header based on test type"""
    titles = {
        'single_prompt': 'Single Prompt Performance Test',
        'cache_rate': 'Cache Rate Performance Test',
        'working_set': 'Working Set Performance Test',
        'sustained': 'Sustained Mode Performance Test',
        'combined': 'Combined Test Comparison'
    }

    title = titles.get(test_type, 'Performance Test Results')
    model = config.get('detected_model', config.get('model', 'Unknown'))
    api = config.get('api_endpoint', '')

    return f"""
    <div class="header">
        <h1>{title}</h1>
        <p><strong>Generated:</strong> {timestamp}</p>
        <p><strong>Model:</strong> {model}</p>
        {f'<p><strong>API:</strong> {api}</p>' if api else ''}
    </div>
"""


def generate_config_section(config: Dict, test_type: str) -> str:
    """Generate configuration table"""
    if not config:
        return ""

    rows = []

    # Common fields
    if config.get('api_endpoint'):
        rows.append(('API Endpoint', f"<code>{config['api_endpoint']}</code>"))
    if config.get('detected_model'):
        rows.append(('Model', f"<strong>{config['detected_model']}</strong>"))
    if config.get('tokenizer_id'):
        rows.append(('Tokenizer', f"<code>{config['tokenizer_id']}</code>"))

    # Context sizes
    if config.get('context_sizes'):
        sizes = config['context_sizes']
        if isinstance(sizes, list):
            rows.append(('Context Sizes', ', '.join(f'{s:,}' for s in sizes) + ' tokens'))

    # Test-specific fields
    if test_type in ('cache_rate', 'working_set', 'sustained'):
        if config.get('working_set_size'):
            rows.append(('Working Set Size', f"{config['working_set_size']:,} tokens"))
        if config.get('cache_hit_rates'):
            rates = config['cache_hit_rates']
            if isinstance(rates, list):
                rows.append(('Cache Hit Rates', f"{min(rates)}% to {max(rates)}% ({len(rates)} rates)"))

    if test_type == 'working_set':
        if config.get('min_working_set_size'):
            rows.append(('Min Working Set', f"{config['min_working_set_size']:,} tokens"))
        if config.get('max_working_set_size'):
            rows.append(('Max Working Set', f"{config['max_working_set_size']:,} tokens"))

    # Timing/threshold fields
    if config.get('output_tokens'):
        rows.append(('Output Tokens', f"{config['output_tokens']} per request"))
    if config.get('max_ttft'):
        metric = config.get('ttft_metric', 'p95').upper()
        rows.append(('TTFT Threshold', f"{config['max_ttft']}s ({metric})"))
    if config.get('test_duration'):
        rows.append(('Test Duration', f"{config['test_duration']}s per test"))
    if config.get('num_iterations'):
        rows.append(('Iterations', f"{config['num_iterations']} per context"))

    if not rows:
        return ""

    table_rows = '\n'.join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows)
    return f"""
    <h2>Test Configuration</h2>
    <div class="card">
        <table class="config-table">
            {table_rows}
        </table>
    </div>
"""


def generate_stats_section(df: Optional[pd.DataFrame], config: Dict, test_type: str) -> str:
    """Generate statistics section"""
    if df is None or df.empty:
        return ""

    stats = []

    # Total requests
    if 'total_requests' in df.columns:
        total = int(df['total_requests'].sum())
        stats.append(('Total Requests', format_number(total), None))
    elif 'num_requests' in df.columns:
        total = int(df['num_requests'].sum())
        stats.append(('Total Requests', format_number(total), None))

    # Throughput stats
    if 'input_tokens_per_sec' in df.columns:
        peak = df['input_tokens_per_sec'].max()
        stats.append(('Peak Input Throughput', f"{format_number(peak)} tok/s", None, True))

    if 'output_tokens_per_sec' in df.columns:
        peak = df['output_tokens_per_sec'].max()
        stats.append(('Peak Output Throughput', f"{format_number(peak)} tok/s", None, True))

    # TTFT stats
    if 'avg_ttft' in df.columns:
        best = df['avg_ttft'].min()
        stats.append(('Best Avg TTFT', f"{best:.3f}s", None, True))

    # Test count
    if 'context_size' in df.columns:
        n_ctx = df['context_size'].nunique()
        stats.append(('Context Sizes', str(n_ctx), None))

    if 'cache_hit_rate' in df.columns:
        n_rates = df['cache_hit_rate'].nunique()
        stats.append(('Cache Hit Rates', str(n_rates), None))

    if not stats:
        return ""

    stat_boxes = []
    for item in stats:
        label, value = item[0], item[1]
        detail = item[2] if len(item) > 2 else None
        highlight = item[3] if len(item) > 3 else False

        cls = "stat-box highlight" if highlight else "stat-box"
        detail_html = f'<div class="stat-detail">{detail}</div>' if detail else ''
        stat_boxes.append(f"""
            <div class="{cls}">
                <div class="stat-label">{label}</div>
                <div class="stat-value">{value}</div>
                {detail_html}
            </div>""")

    return f"""
    <h2>Performance Summary</h2>
    <div class="card">
        <div class="stats-grid">
            {''.join(stat_boxes)}
        </div>
    </div>
"""


def generate_kv_cache_section(config: Dict) -> str:
    """Generate KV cache estimates section"""
    model = config.get('detected_model', '')
    working_set = config.get('working_set_size', 0)

    if not working_set:
        return ""

    fp8_size = estimate_kv_cache_size(model, working_set, 'fp8')
    fp16_size = estimate_kv_cache_size(model, working_set, 'fp16')

    return f"""
    <h2>KV Cache Estimates</h2>
    <div class="card">
        <div class="stats-grid">
            <div class="stat-box">
                <div class="stat-label">FP8 Precision</div>
                <div class="stat-value">{format_bytes(fp8_size)}</div>
                <div class="stat-detail">For {format_number(working_set)} tokens</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">FP16 Precision</div>
                <div class="stat-value">{format_bytes(fp16_size)}</div>
                <div class="stat-detail">For {format_number(working_set)} tokens</div>
            </div>
        </div>
    </div>
"""


def generate_graphs_section(graphs: Dict[str, List[Path]], test_type: str) -> str:
    """Generate interactive visualizations section"""
    sections = []

    # Main performance graphs
    if graphs['performance']:
        items = []
        for g in graphs['performance']:
            name = g.stem
            if 'cache_rate' in name or 'cache' in name:
                ctx = name.split('_')[-1]
                items.append((g.name, f"Performance vs Cache Hit Rate ({ctx})",
                            "Throughput and TTFT across cache hit rates"))
            elif 'working_set' in name:
                parts = name.replace('performance_vs_working_set_', '').split('_cache')
                ctx = parts[0]
                cache = parts[1] if len(parts) > 1 else '?'
                items.append((g.name, f"Performance vs Working Set (Ctx: {ctx}, Cache: {cache}%)",
                            "Throughput and TTFT across working set sizes"))
            else:
                items.append((g.name, g.stem, "Performance metrics"))

        for href, title, desc in items:
            sections.append(f"""
        <a href="{href}" class="graph-link">
            <h3>{title}</h3>
            <p>{desc}</p>
        </a>""")

    # Single prompt graphs
    if graphs['single_prompt']:
        for g in graphs['single_prompt']:
            if 'performance' in g.name:
                sections.append(f"""
        <a href="{g.name}" class="graph-link">
            <h3>TTFT: Cold Start vs Cached</h3>
            <p>Bar chart comparing baseline and cached performance with speedup</p>
        </a>""")
            elif 'summary_table' in g.name:
                sections.append(f"""
        <a href="{g.name}" class="graph-link secondary">
            <h3>Summary Statistics Table</h3>
            <p>Detailed statistics for each context size</p>
        </a>""")

    # Comparison graphs
    if graphs['comparison']:
        for g in graphs['comparison']:
            name = g.stem
            if 'input_throughput' in name:
                ctx = name.replace('input_throughput_comparison', '').replace('input_throughput_ctx', '').strip('_')
                title = f"Input Throughput Comparison" + (f" ({ctx})" if ctx else "")
                sections.append(f"""
        <a href="{g.name}" class="graph-link secondary">
            <h3>{title}</h3>
            <p>Compare input throughput across configurations</p>
        </a>""")
            elif 'output_throughput' in name:
                ctx = name.replace('output_throughput_comparison', '').replace('output_throughput_ctx', '').strip('_')
                title = f"Output Throughput Comparison" + (f" ({ctx})" if ctx else "")
                sections.append(f"""
        <a href="{g.name}" class="graph-link secondary">
            <h3>{title}</h3>
            <p>Compare output throughput across configurations</p>
        </a>""")
            elif 'output_metrics' in name:
                sections.append(f"""
        <a href="{g.name}" class="graph-link secondary">
            <h3>Output Token Metrics</h3>
            <p>Inter-token latency and generation speed</p>
        </a>""")
            elif 'ttft' in name:
                ctx = name.replace('ttft_comparison_ctx', '').strip('_')
                title = f"TTFT Comparison" + (f" (Context: {ctx})" if ctx else "")
                sections.append(f"""
        <a href="{g.name}" class="graph-link secondary">
            <h3>{title}</h3>
            <p>Time to first token metrics comparison</p>
        </a>""")

    # Heatmaps
    if graphs['heatmap']:
        for g in graphs['heatmap']:
            sections.append(f"""
        <a href="{g.name}" class="graph-link tertiary">
            <h3>TTFT Heatmap</h3>
            <p>2D visualization across context sizes and cache rates</p>
        </a>""")

    # Sustained mode graphs
    if graphs['sustained']:
        individual = [g for g in graphs['sustained'] if 'comparison' not in g.name]
        comparisons = [g for g in graphs['sustained'] if 'comparison' in g.name]

        if individual:
            sections.append('<h3 style="margin-top: 20px;">Sustained Mode - Individual Tests</h3>')
            for g in individual:
                parts = g.stem.replace('sustained_ctx', '').replace('sustained_performance_ctx', '')
                if '_cache' in parts:
                    ctx, rest = parts.split('_cache', 1)
                    cache = rest.split('_')[0]
                    sections.append(f"""
        <a href="{g.name}" class="graph-link tertiary">
            <h3>Context {ctx} - Cache {cache}%</h3>
            <p>Performance over time with adaptive concurrency</p>
        </a>""")

        if comparisons:
            sections.append('<h3 style="margin-top: 20px;">Sustained Mode - Comparisons</h3>')
            for g in comparisons:
                cache = g.stem.replace('sustained_comparison_cache', '')
                sections.append(f"""
        <a href="{g.name}" class="graph-link tertiary">
            <h3>Cache {cache}% - All Contexts</h3>
            <p>Compare performance across context sizes</p>
        </a>""")

    # Fixed mode / Ramp graphs (collapsible by context)
    if graphs['fixed']:
        fixed_by_ctx = {}
        for g in graphs['fixed']:
            # Handle both fixed_ctx and ramp_ctx naming
            name = g.stem.replace('fixed_ctx', '').replace('ramp_ctx', '')
            parts = name.split('_')
            ctx = parts[0]
            if ctx not in fixed_by_ctx:
                fixed_by_ctx[ctx] = []
            fixed_by_ctx[ctx].append(g)

        sections.append('<h3 style="margin-top: 20px;">Fixed Concurrency Analysis</h3>')
        for ctx in sorted(fixed_by_ctx.keys(), key=lambda x: int(x)):
            fixed_graphs = fixed_by_ctx[ctx]
            sections.append(f"""
        <details>
            <summary>Context: {ctx} tokens ({len(fixed_graphs)} tests)</summary>
            <div>""")
            for g in sorted(fixed_graphs):
                name = g.stem
                if '_cache' in name:
                    cache = name.split('_cache')[1].split('_')[0]
                    label = f"Cache: {cache}%"
                elif '_ws' in name:
                    ws = name.split('_ws')[1].split('_')[0]
                    cache = name.split('_cache')[1].split('_')[0] if '_cache' in name else '?'
                    label = f"WS: {ws}, Cache: {cache}%"
                else:
                    label = g.stem
                sections.append(f"""
                <a href="{g.name}" class="graph-link secondary" style="margin: 5px 0;">
                    <h3>{label}</h3>
                    <p>Throughput and TTFT vs concurrency</p>
                </a>""")
            sections.append("""
            </div>
        </details>""")

    # Other graphs
    if graphs['other']:
        sections.append('<h3 style="margin-top: 20px;">Additional Graphs</h3>')
        for g in graphs['other']:
            sections.append(f"""
        <a href="{g.name}" class="graph-link">
            <h3>{g.stem}</h3>
            <p>Click to view</p>
        </a>""")

    if not sections:
        return ""

    return f"""
    <h2>Interactive Visualizations</h2>
    <div class="card">
        {''.join(sections)}
    </div>
"""


def generate_data_files_section(output_path: Path) -> str:
    """Generate data files section"""
    csv_files = sorted(output_path.glob("*.csv"))
    json_files = [f for f in output_path.glob("*.json") if f.name not in ('progress.json', 'metadata.json')]
    sh_files = sorted(output_path.glob("*.sh"))

    if not csv_files and not json_files and not sh_files:
        return ""

    items = []
    for f in csv_files:
        size = format_bytes(f.stat().st_size)
        items.append(f'<li><a href="{f.name}">{f.name}</a> ({size})</li>')
    for f in json_files:
        size = format_bytes(f.stat().st_size)
        items.append(f'<li><a href="{f.name}">{f.name}</a> ({size})</li>')
    for f in sh_files:
        items.append(f'<li><a href="{f.name}">{f.name}</a> (reproducible command)</li>')

    return f"""
    <h2>Data Files</h2>
    <div class="card">
        <ul class="csv-list">
            {''.join(items)}
        </ul>
    </div>
"""


def generate_info_section(test_type: str, config: Dict) -> str:
    """Generate info/notes section"""
    if test_type == 'single_prompt':
        return """
    <h2>How to Interpret</h2>
    <div class="card">
        <div class="info-box">
            <ul style="margin: 0; padding-left: 20px;">
                <li><strong>Baseline (Cold Start):</strong> First request for a prompt - no cache benefit</li>
                <li><strong>Cached (100% Hit):</strong> Same prompt sent again - uses cached KV</li>
                <li><strong>Speedup:</strong> Baseline TTFT / Cached TTFT - higher is better</li>
            </ul>
            <p style="margin-top: 10px;">A speedup of 2-10x indicates effective caching. Speedup typically increases with context size.</p>
        </div>
    </div>
"""

    if test_type == 'sustained':
        return """
    <h2>About Sustained Mode</h2>
    <div class="card">
        <div class="info-box">
            <p><strong>Sustained mode</strong> continuously adjusts concurrency based on periodic assessments:</p>
            <ul style="margin: 10px 0; padding-left: 20px;">
                <li>Each period measures TTFT, throughput, and latency</li>
                <li>If TTFT is under threshold: <strong>RAMP UP</strong> concurrency</li>
                <li>If TTFT exceeds threshold: <strong>RAMP DOWN</strong> concurrency</li>
            </ul>
            <p>Useful for understanding performance variability and stability over time.</p>
        </div>
    </div>
"""

    if test_type in ('cache_rate', 'working_set'):
        notes = []
        if config.get('max_ttft'):
            notes.append(f"Concurrency ramped until TTFT exceeded {config['max_ttft']}s threshold")
        if config.get('test_duration'):
            notes.append(f"Each test limited to {config['test_duration']} seconds")

        if notes:
            note_items = ''.join(f'<li>{n}</li>' for n in notes)
            return f"""
    <h2>Test Notes</h2>
    <div class="card">
        <ul style="margin: 0; padding-left: 20px;">
            {note_items}
        </ul>
    </div>
"""

    return ""


def generate_index_html(output_dir: str, version: str = "1.0"):
    """Generate unified index.html dashboard"""
    output_path = Path(output_dir)

    if not output_path.exists():
        print(f"Error: Directory {output_dir} does not exist")
        return

    # Auto-detect test type
    test_type = detect_test_type(output_path)
    print(f"Detected test type: {test_type}")

    # Load data
    config = load_config(output_path)
    df = load_summary_data(output_path)
    graphs = get_graph_files(output_path)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Build HTML
    html_parts = [
        f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Test Results - {timestamp}</title>
    <style>{CSS_STYLES}
    </style>
</head>
<body>""",
        generate_header(test_type, config, timestamp),
        generate_config_section(config, test_type),
        generate_stats_section(df, config, test_type),
    ]

    # Add KV cache section for relevant test types
    if test_type in ('cache_rate', 'working_set', 'sustained') and config.get('working_set_size'):
        html_parts.append(generate_kv_cache_section(config))

    html_parts.extend([
        generate_graphs_section(graphs, test_type),
        generate_data_files_section(output_path),
        generate_info_section(test_type, config),
        f"""
    <div class="footer">
        <p>Generated with KV Cache Tester v{version}</p>
    </div>
</body>
</html>"""
    ])

    # Write file
    html_content = ''.join(html_parts)
    index_file = output_path / "index.html"
    with open(index_file, 'w') as f:
        f.write(html_content)

    print(f"Generated index.html: {index_file}")


if __name__ == "__main__":
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "./output"
    version = sys.argv[2] if len(sys.argv) > 2 else "1.0"
    generate_index_html(output_dir, version)
