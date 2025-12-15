#!/usr/bin/env python3
"""
Single Prompt Performance Tester

Measures cold start (unique prompt) vs warm cache (100% cached prompt) performance
across doubling context sizes to understand:
- Time to first token (TTFT) scaling with context length
- Cache hit impact at different context sizes
- Maximum practical context length for your system

Test pattern for each context size:
1. Send unique prompt (cold start)
2. Send identical prompt again (100% cache hit)
3. Measure TTFT, TTLT, throughput for both

Version: 1.0
Date: 2025-10-24

Usage:
    python single_prompt_tester.py \\
        --api-endpoint http://localhost:8000 \\
        --min-tokens 1000 \\
        --max-tokens 128000 \\
        --output-tokens 256 \\
        --num-iterations 5 \\
        --output-dir single_prompt_results
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd

try:
    import openai
    from transformers import AutoTokenizer
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Please install: pip install openai transformers plotly pandas numpy")
    sys.exit(1)


class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    INFO = '\033[97m'
    DEBUG = '\033[90m'
    METRIC = '\033[96m'
    SUCCESS = '\033[92m'
    PHASE = '\033[95m'


class ColoredFormatter(logging.Formatter):
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
    logger = logging.getLogger(name)
    logger.setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(ColoredFormatter())
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler('single_prompt_tester.log', mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    return logger


logger = init_logger(__name__)


@dataclass
class PromptMetrics:
    """Metrics for a single prompt test"""
    context_size: int
    iteration: int
    prompt_type: str  # "unique" or "cached"
    prompt_tokens: int
    completion_tokens: int
    ttft: float
    generation_time: float
    total_time: float
    output_tokens_per_sec: float  # Per-request generation speed

    def to_dict(self) -> dict:
        return asdict(self)


class TokenizerManager:
    def __init__(self, tokenizer_id: str):
        self.tokenizer_id = tokenizer_id
        self.tokenizer = None
        self._load_tokenizer()

    def _load_tokenizer(self):
        logger.info(f"Loading tokenizer: {self.tokenizer_id}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_id,
                trust_remote_code=True
            )
            # Disable max length warnings
            if hasattr(self.tokenizer, 'model_max_length'):
                self.tokenizer.model_max_length = 1_000_000
            logger.info(f"Tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise

    def encode(self, text: str) -> List[int]:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Token indices sequence length.*")
            return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def generate_dummy_tokens(self, num_tokens: int, seed: Optional[int] = None) -> List[int]:
        """
        Generate diverse dummy tokens to activate different experts in MoE models
        Uses a mix of natural language, code, and formatting tokens
        """
        if seed is not None:
            np.random.seed(seed)

        # Diverse vocabulary to activate different MoE experts
        word_pools = {
            'natural': [
                'the', 'and', 'with', 'for', 'this', 'that', 'from', 'have', 'been', 'which',
                'their', 'said', 'each', 'about', 'many', 'some', 'these', 'would', 'other', 'into'
            ],
            'code_keywords': [
                'function', 'class', 'return', 'import', 'async', 'await', 'const', 'let', 'var',
                'def', 'if', 'else', 'while', 'for', 'try', 'catch', 'finally', 'throw', 'new'
            ],
            'code_symbols': [
                '{', '}', '(', ')', '[', ']', ';', ':', ',', '.', '=', '==', '!=', '&&', '||',
                '+', '-', '*', '/', '%', '<', '>', '<=', '>=', '=>', '->', '::', '//'
            ],
            'technical': [
                'data', 'model', 'input', 'output', 'token', 'layer', 'cache', 'memory', 'compute',
                'inference', 'batch', 'tensor', 'matrix', 'vector', 'array', 'buffer', 'queue'
            ],
            'formatting': [
                '\n', '\t', '    ', '  ', '"""', "'''", '#', '//', '/*', '*/', '<!--', '-->'
            ]
        }

        # Flatten all word pools with weights
        all_words = []
        weights = []

        # Natural language (30%)
        all_words.extend(word_pools['natural'])
        weights.extend([0.3 / len(word_pools['natural'])] * len(word_pools['natural']))

        # Code keywords (25%)
        all_words.extend(word_pools['code_keywords'])
        weights.extend([0.25 / len(word_pools['code_keywords'])] * len(word_pools['code_keywords']))

        # Code symbols (20%)
        all_words.extend(word_pools['code_symbols'])
        weights.extend([0.20 / len(word_pools['code_symbols'])] * len(word_pools['code_symbols']))

        # Technical terms (15%)
        all_words.extend(word_pools['technical'])
        weights.extend([0.15 / len(word_pools['technical'])] * len(word_pools['technical']))

        # Formatting (10%)
        all_words.extend(word_pools['formatting'])
        weights.extend([0.10 / len(word_pools['formatting'])] * len(word_pools['formatting']))

        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()

        # Generate diverse text
        chunks = []
        chunk_size = 100

        for _ in range((num_tokens * 3) // chunk_size):
            chunk_type = np.random.choice(['natural', 'code', 'mixed'], p=[0.3, 0.4, 0.3])

            if chunk_type == 'natural':
                chunk_words = np.random.choice(word_pools['natural'], size=chunk_size // 2)
                chunks.append(' '.join(chunk_words) + '.')
            elif chunk_type == 'code':
                func_name = np.random.choice(['process', 'calculate', 'transform', 'execute'])
                chunk_words = np.random.choice(word_pools['code_keywords'] + word_pools['technical'], size=chunk_size // 3)
                chunks.append(f"def {func_name}({', '.join(chunk_words[:3])}): return {' '.join(chunk_words[3:])}")
            else:
                chunk_words = np.random.choice(all_words, size=chunk_size // 2, p=weights)
                chunks.append(' '.join(chunk_words))

        text = '\n'.join(chunks)
        tokens = self.encode(text)
        return tokens[:num_tokens]


class APIClient:
    def __init__(self, api_endpoint: str, model: str):
        self.api_endpoint = api_endpoint
        self.model = model
        base_url = api_endpoint.rstrip('/')
        if not base_url.endswith('/v1'):
            base_url = base_url + '/v1'
        self.client = openai.AsyncOpenAI(api_key="EMPTY", base_url=base_url)
        logger.info(f"API Client initialized: {api_endpoint}")

    async def detect_model(self) -> str:
        try:
            models_url = self.api_endpoint.rstrip('/') + '/v1/models'
            logger.info(f"Auto-detecting model from {models_url}")

            import requests
            response = requests.get(models_url, timeout=10)
            response.raise_for_status()

            data = response.json()
            if 'data' in data and len(data['data']) > 0:
                model_name = data['data'][0]['id']
                logger.info(f"Auto-detected model: {model_name}")
                return model_name
            else:
                raise ValueError("No models found in API response")
        except Exception as e:
            logger.error(f"Failed to auto-detect model: {e}")
            raise

    async def send_request(self, prompt: str, max_tokens: int) -> Tuple[str, float, float, int, int]:
        """
        Send request and return metrics
        Returns: (response_text, ttft, generation_time, prompt_tokens, completion_tokens)
        """
        start_time = time.time()
        first_token_time = None
        response_text = ""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0,
                stream=True,
                stream_options={"include_usage": True}
            )

            async for chunk in response:
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta
                if delta.content is not None:
                    if first_token_time is None and delta.content != "":
                        first_token_time = time.time()
                    response_text += delta.content

            prompt_tokens = chunk.usage.prompt_tokens if hasattr(chunk, 'usage') else 0
            completion_tokens = chunk.usage.completion_tokens if hasattr(chunk, 'usage') else 0

            ttft = (first_token_time - start_time) if first_token_time else 0.0
            generation_time = (time.time() - first_token_time) if first_token_time else 0.0

            return response_text, ttft, generation_time, prompt_tokens, completion_tokens

        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise


async def test_single_prompt_pair(api_client: APIClient, tokenizer: TokenizerManager,
                                  context_size: int, output_tokens: int, iteration: int, base_seed: int) -> Tuple[PromptMetrics, PromptMetrics]:
    """
    Test a single unique prompt followed by cached repeat
    Returns: (unique_metrics, cached_metrics)
    """
    # Generate unique prompt with seed offset for this iteration
    prompt_seed = base_seed + iteration * 1000
    unique_tokens = tokenizer.generate_dummy_tokens(context_size, seed=prompt_seed)
    unique_prompt = tokenizer.decode(unique_tokens) + "\n\nTell me a story."

    logger.info(f"    Iteration {iteration + 1}: Testing unique prompt ({context_size:,} tokens)...")

    # Test unique prompt (cold start)
    start_time = time.time()
    response_text, ttft, gen_time, prompt_tok, completion_tok = await api_client.send_request(unique_prompt, output_tokens)
    total_time = time.time() - start_time
    output_tps = completion_tok / gen_time if gen_time > 0 else 0

    unique_metrics = PromptMetrics(
        context_size=context_size,
        iteration=iteration,
        prompt_type="unique",
        prompt_tokens=prompt_tok,
        completion_tokens=completion_tok,
        ttft=ttft,
        generation_time=gen_time,
        total_time=total_time,
        output_tokens_per_sec=output_tps
    )

    # Show snippet of response
    snippet = response_text[:100].replace('\n', ' ') if response_text else ""
    logger.info(f"      Unique: TTFT={ttft:.3f}s, Output={completion_tok} tok")
    logger.info(f"      Response: {snippet}...")

    # Test cached prompt (same prompt again - 100% cache hit)
    logger.info(f"    Iteration {iteration + 1}: Testing cached prompt (same prompt)...")

    start_time = time.time()
    response_text, ttft, gen_time, prompt_tok, completion_tok = await api_client.send_request(unique_prompt, output_tokens)
    total_time = time.time() - start_time
    output_tps = completion_tok / gen_time if gen_time > 0 else 0

    cached_metrics = PromptMetrics(
        context_size=context_size,
        iteration=iteration,
        prompt_type="cached",
        prompt_tokens=prompt_tok,
        completion_tokens=completion_tok,
        ttft=ttft,
        generation_time=gen_time,
        total_time=total_time,
        output_tokens_per_sec=output_tps
    )

    # Show snippet of response
    snippet = response_text[:100].replace('\n', ' ') if response_text else ""
    logger.info(f"      Cached: TTFT={ttft:.3f}s, Output={completion_tok} tok")
    logger.info(f"      Response: {snippet}...")
    logger.info(f"      Cache speedup: TTFT {unique_metrics.ttft/ttft:.2f}x faster")

    return unique_metrics, cached_metrics


def generate_graphs(results: List[PromptMetrics], output_dir: str):
    """Generate visualizations"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([m.to_dict() for m in results])

    # Calculate aggregated metrics (mean across iterations)
    agg_data = []
    for context_size in sorted(df['context_size'].unique()):
        for prompt_type in ['unique', 'cached']:
            data = df[(df['context_size'] == context_size) & (df['prompt_type'] == prompt_type)]

            if len(data) > 0:
                agg_data.append({
                    'context_size': context_size,
                    'prompt_type': prompt_type,
                    'avg_ttft': data['ttft'].mean(),
                    'std_ttft': data['ttft'].std(),
                    'min_ttft': data['ttft'].min(),
                    'max_ttft': data['ttft'].max(),
                    'avg_output_tps': data['output_tokens_per_sec'].mean(),
                    'num_iterations': len(data)
                })

    agg_df = pd.DataFrame(agg_data)

    # Get unique and cached data
    unique_df = agg_df[agg_df['prompt_type'] == 'unique'].sort_values('context_size')
    cached_df = agg_df[agg_df['prompt_type'] == 'cached'].sort_values('context_size')

    # Calculate speedup for each context size
    speedup_data = []
    for ctx in sorted(unique_df['context_size'].unique()):
        u_ttft = unique_df[unique_df['context_size'] == ctx]['avg_ttft'].iloc[0]
        c_ttft = cached_df[cached_df['context_size'] == ctx]['avg_ttft'].iloc[0]
        speedup = u_ttft / c_ttft if c_ttft > 0 else 0
        speedup_data.append({'context_size': ctx, 'speedup': speedup})

    speedup_df = pd.DataFrame(speedup_data)

    # Create figure with secondary y-axis for speedup
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Prepare data for grouped bar chart
    context_sizes = sorted(unique_df['context_size'].unique())
    x_labels = [f"{ctx:,}" for ctx in context_sizes]

    # Add baseline (unique) bars
    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=unique_df['avg_ttft'].values,
            name='Baseline (Cold Start)',
            marker=dict(color='#EF553B'),
            error_y=dict(type='data', array=unique_df['std_ttft'].values, visible=True),
            hovertemplate='Context: %{x} tokens<br>Baseline TTFT: %{y:.3f}s<extra></extra>'
        ),
        secondary_y=False
    )

    # Add cached bars
    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=cached_df['avg_ttft'].values,
            name='Cached (100% Hit)',
            marker=dict(color='#00CC96'),
            error_y=dict(type='data', array=cached_df['std_ttft'].values, visible=True),
            hovertemplate='Context: %{x} tokens<br>Cached TTFT: %{y:.3f}s<extra></extra>'
        ),
        secondary_y=False
    )

    # Add speedup line on secondary axis
    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=speedup_df['speedup'].values,
            name='Cache Speedup',
            mode='lines+markers',
            line=dict(color='#636EFA', width=3),
            marker=dict(size=10, symbol='diamond'),
            hovertemplate='Context: %{x} tokens<br>Speedup: %{y:.2f}x<extra></extra>'
        ),
        secondary_y=True
    )

    # Update axes
    fig.update_xaxes(title_text="Context Size (tokens)")
    fig.update_yaxes(title_text="Time to First Token (seconds)", secondary_y=False)
    fig.update_yaxes(title_text="Cache Speedup (x)", secondary_y=True)

    # Update layout
    fig.update_layout(
        title="TTFT Performance: Baseline vs Cached with Speedup",
        height=600,
        showlegend=True,
        hovermode='x unified',
        barmode='group'
    )

    filename = output_path / "single_prompt_performance.html"
    fig.write_html(filename)
    logger.info(f"Generated graph: {filename}")


def generate_summary_table(results: List[PromptMetrics], output_dir: str):
    """Generate HTML summary table"""
    output_path = Path(output_dir)
    df = pd.DataFrame([m.to_dict() for m in results])

    # Calculate aggregated metrics
    summary_data = []
    for context_size in sorted(df['context_size'].unique()):
        unique_data = df[(df['context_size'] == context_size) & (df['prompt_type'] == 'unique')]
        cached_data = df[(df['context_size'] == context_size) & (df['prompt_type'] == 'cached')]

        if len(unique_data) > 0 and len(cached_data) > 0:
            u_ttft_mean = unique_data['ttft'].mean()
            u_ttft_std = unique_data['ttft'].std()
            c_ttft_mean = cached_data['ttft'].mean()
            c_ttft_std = cached_data['ttft'].std()
            speedup = u_ttft_mean / c_ttft_mean if c_ttft_mean > 0 else 0

            summary_data.append({
                'Context Size': f"{context_size:,}",
                'Unique TTFT (mean)': f"{u_ttft_mean:.3f}s",
                'Unique TTFT (std)': f"{u_ttft_std:.3f}s",
                'Cached TTFT (mean)': f"{c_ttft_mean:.3f}s",
                'Cached TTFT (std)': f"{c_ttft_std:.3f}s",
                'Cache Speedup': f"{speedup:.2f}x",
                'Iterations': len(unique_data)
            })

    summary_df = pd.DataFrame(summary_data)

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Single Prompt Performance Summary</title>
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
            text-align: right;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }}
        th:first-child, td:first-child {{
            text-align: center;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        tr:hover {{
            background-color: #e8f5e9;
        }}
    </style>
</head>
<body>
    <h1>Single Prompt Performance Summary</h1>
    <p>Compares cold start (unique prompt) vs warm cache (100% cached) performance across context sizes</p>
    {summary_df.to_html(index=False, escape=False)}
</body>
</html>
"""

    filename = output_path / "summary_table.html"
    with open(filename, 'w') as f:
        f.write(html_content)
    logger.info(f"Generated summary table: {filename}")


async def main():
    parser = argparse.ArgumentParser(
        description="Single Prompt Performance Tester - Cold Start vs Cached",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--api-endpoint", type=str, required=True,
                       help="API endpoint (e.g., http://localhost:8000)")
    parser.add_argument("--context-sizes", type=int, nargs='+', default=None,
                       help="Specific context sizes to test (e.g., 8000 32000 64000). Overrides --min-tokens and --max-tokens.")
    parser.add_argument("--min-tokens", type=int, default=1000,
                       help="Minimum context size in tokens (default: 1000). Ignored if --context-sizes is specified.")
    parser.add_argument("--max-tokens", type=int, default=128000,
                       help="Maximum context size in tokens (default: 128000). Ignored if --context-sizes is specified.")
    parser.add_argument("--output-tokens", type=int, default=256,
                       help="Output tokens per request (default: 256)")
    parser.add_argument("--num-iterations", type=int, default=5,
                       help="Number of iterations per context size (default: 5)")
    parser.add_argument("--output-dir", type=str, default="./single_prompt_output",
                       help="Output directory (default: ./single_prompt_output)")
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-Coder-32B-Instruct",
                       help="Tokenizer model ID")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility (default: random based on timestamp)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--brief", action="store_true",
                       help="Brief output mode for agents - minimal, parseable output")

    args = parser.parse_args()

    # Brief mode setup - suppress normal logging
    brief_mode = args.brief
    if brief_mode:
        logger.setLevel(logging.WARNING)  # Only show warnings/errors

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    logger.info(f"{Colors.HEADER}{'='*80}{Colors.ENDC}")
    logger.info(f"{Colors.HEADER}{Colors.BOLD}Single Prompt Performance Tester{Colors.ENDC}")
    logger.info(f"{Colors.HEADER}{'='*80}{Colors.ENDC}")
    logger.info(f"{Colors.OKBLUE}API Endpoint: {args.api_endpoint}{Colors.ENDC}")
    logger.info(f"{Colors.OKBLUE}Iterations per size: {args.num_iterations}{Colors.ENDC}")
    logger.info(f"{Colors.HEADER}{'='*80}{Colors.ENDC}")

    # Initialize API client
    api_client = APIClient(args.api_endpoint, model="")
    model = await api_client.detect_model()
    api_client.model = model

    # Initialize tokenizer
    tokenizer = TokenizerManager(args.tokenizer)

    # Initialize random seed
    if args.seed is None:
        # Use timestamp for random seed to ensure different prompts each run
        import random
        base_seed = int(time.time() * 1000) % 1000000
        logger.info(f"{Colors.OKBLUE}Using random seed: {base_seed} (use --seed {base_seed} to reproduce){Colors.ENDC}")
    else:
        base_seed = args.seed
        logger.info(f"{Colors.OKBLUE}Using provided seed: {base_seed}{Colors.ENDC}")

    # Generate context sizes
    if args.context_sizes:
        # Use explicitly specified context sizes
        context_sizes = sorted(args.context_sizes)
        logger.info(f"{Colors.OKBLUE}Using specified context sizes: {context_sizes}{Colors.ENDC}")
    else:
        # Validate input parameters for doubling mode
        if args.min_tokens > args.max_tokens:
            logger.error(f"Invalid parameter range: min-tokens ({args.min_tokens}) must be <= max-tokens ({args.max_tokens})")
            sys.exit(1)

        # Generate context sizes (doubling)
        context_sizes = []
        size = args.min_tokens
        while size <= args.max_tokens:
            context_sizes.append(size)
            size *= 2
        logger.info(f"{Colors.OKBLUE}Context Range: {args.min_tokens:,} to {args.max_tokens:,} tokens (doubling){Colors.ENDC}")

    if len(context_sizes) == 0:
        logger.error("No context sizes to test. Specify --context-sizes or check --min-tokens and --max-tokens parameters.")
        sys.exit(1)

    logger.info(f"\n{Colors.PHASE}Testing {len(context_sizes)} context sizes: {context_sizes}{Colors.ENDC}\n")

    # Run tests
    all_results = []

    for context_size in context_sizes:
        logger.info(f"{Colors.OKCYAN}Testing context size: {context_size:,} tokens{Colors.ENDC}")

        for iteration in range(args.num_iterations):
            unique_metrics, cached_metrics = await test_single_prompt_pair(
                api_client, tokenizer, context_size, args.output_tokens, iteration, base_seed
            )
            all_results.extend([unique_metrics, cached_metrics])

        logger.info("")

    # Save results
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save detailed CSV
    df = pd.DataFrame([m.to_dict() for m in all_results])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = output_path / f"single_prompt_results_{timestamp}.csv"
    df.to_csv(csv_file, index=False)
    logger.info(f"Saved results to {csv_file}")

    # Generate graphs
    logger.info(f"\n{Colors.PHASE}Generating visualizations...{Colors.ENDC}")
    generate_graphs(all_results, args.output_dir)
    generate_summary_table(all_results, args.output_dir)

    # Save metadata for index.html generator
    metadata = {
        "test_type": "single_prompt",
        "api_endpoint": args.api_endpoint,
        "detected_model": model,
        "context_sizes": context_sizes,
        "num_iterations": args.num_iterations,
        "output_tokens": args.output_tokens,
        "seed": base_seed
    }
    metadata_file = output_path / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Generate index.html using unified generator
    logger.info(f"{Colors.PHASE}Generating index.html dashboard...{Colors.ENDC}")
    try:
        import subprocess
        subprocess.run([
            sys.executable, "generate_index.py",
            str(output_path)
        ], check=True, cwd=Path(__file__).parent)
        logger.info(f"{Colors.SUCCESS}✓ Generated index.html dashboard{Colors.ENDC}")
    except Exception as e:
        logger.warning(f"Failed to generate index.html: {e}")

    logger.info(f"\n{Colors.SUCCESS}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    logger.info(f"{Colors.SUCCESS}{Colors.BOLD}✓ Testing complete!{Colors.ENDC}")
    logger.info(f"{Colors.SUCCESS}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    logger.info(f"{Colors.OKGREEN}Results saved to: {args.output_dir}{Colors.ENDC}\n")

    # Brief mode output
    if brief_mode:
        print(f"model: {model}")
        print(f"endpoint: {args.api_endpoint}")
        print(f"seed: {base_seed}")
        print()
        print("context_size,unique_ttft,cached_ttft,speedup")

        # Calculate averages per context size
        df = pd.DataFrame([m.to_dict() for m in all_results])
        for ctx in sorted(df['context_size'].unique()):
            unique_data = df[(df['context_size'] == ctx) & (df['prompt_type'] == 'unique')]
            cached_data = df[(df['context_size'] == ctx) & (df['prompt_type'] == 'cached')]
            if len(unique_data) > 0 and len(cached_data) > 0:
                u_ttft = unique_data['ttft'].mean()
                c_ttft = cached_data['ttft'].mean()
                speedup = u_ttft / c_ttft if c_ttft > 0 else 0
                print(f"{ctx},{u_ttft:.3f},{c_ttft:.3f},{speedup:.2f}x")

        print()
        print(f"output: {args.output_dir}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
