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
    INFO = ''                # Default terminal color
    DEBUG = '\033[90m'
    METRIC = '\033[96m'      # Cyan for metrics
    SUCCESS = '\033[92m'     # Green for success
    PHASE = '\033[95m'       # Magenta for phase headers

    @classmethod
    def disable(cls):
        """Disable all colors for terminals where colors are hard to read"""
        cls.HEADER = ''
        cls.OKBLUE = ''
        cls.OKCYAN = ''
        cls.OKGREEN = ''
        cls.WARNING = ''
        cls.FAIL = ''
        cls.ENDC = ''
        cls.BOLD = ''
        cls.INFO = ''
        cls.DEBUG = ''
        cls.METRIC = ''
        cls.SUCCESS = ''
        cls.PHASE = ''


class ColoredFormatter(logging.Formatter):
    def format(self, record):
        # Read colors dynamically to support --no-color flag
        formats = {
            logging.DEBUG: Colors.DEBUG + '[%(asctime)s] DEBUG - %(message)s' + Colors.ENDC,
            logging.INFO: Colors.INFO + '[%(asctime)s] INFO - %(message)s' + Colors.ENDC,
            logging.WARNING: Colors.WARNING + '[%(asctime)s] WARNING - %(message)s' + Colors.ENDC,
            logging.ERROR: Colors.FAIL + '[%(asctime)s] ERROR - %(message)s' + Colors.ENDC,
        }
        log_fmt = formats.get(record.levelno)
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
                                  context_size: int, output_tokens: int, iteration: int, base_seed: int,
                                  concurrent_prompts: int = 1, cached_repeats: int = 1) -> Tuple[List[PromptMetrics], List[PromptMetrics]]:
    """
    Test unique prompts followed by cached repeats
    Returns: (list of unique_metrics, list of cached_metrics)
    """
    # Generate unique prompts with seed offset for this iteration
    prompts = []
    for i in range(concurrent_prompts):
        prompt_seed = base_seed + iteration * 1000 + i * 100
        unique_tokens = tokenizer.generate_dummy_tokens(context_size, seed=prompt_seed)
        unique_prompt = tokenizer.decode(unique_tokens) + "\n\nTell me a story."
        prompts.append(unique_prompt)

    prompt_label = "prompts" if concurrent_prompts > 1 else "prompt"
    logger.info(f"    Iteration {iteration + 1}: Testing {concurrent_prompts} unique {prompt_label} ({context_size:,} tokens each)...")

    # Test unique prompts (cold start) - send all simultaneously
    # Returns: (response_text, ttft, gen_time, prompt_tok, completion_tok, total_time, absolute_first_token_time)
    async def send_and_measure(prompt: str, prompt_idx: int, batch_start_ref: float) -> Tuple[str, float, float, int, int, float, float]:
        start_time = time.time()
        response_text, ttft, gen_time, prompt_tok, completion_tok = await api_client.send_request(prompt, output_tokens)
        total_time = time.time() - start_time
        # Calculate absolute first-token time relative to batch start
        absolute_first_token = (start_time - batch_start_ref) + ttft
        return response_text, ttft, gen_time, prompt_tok, completion_tok, total_time, absolute_first_token

    # Send all unique prompts simultaneously
    batch_start = time.time()
    unique_results = await asyncio.gather(*[send_and_measure(p, i, batch_start) for i, p in enumerate(prompts)])
    batch_elapsed = time.time() - batch_start

    unique_metrics_list = []
    for i, (response_text, ttft, gen_time, prompt_tok, completion_tok, total_time, abs_ttft) in enumerate(unique_results):
        output_tps = completion_tok / gen_time if gen_time > 0 else 0
        metrics = PromptMetrics(
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
        unique_metrics_list.append(metrics)

    # Log per-prompt details when concurrent
    if concurrent_prompts > 1:
        for i, (response_text, ttft, gen_time, prompt_tok, completion_tok, total_time, abs_ttft) in enumerate(unique_results):
            snippet = response_text[:80].replace('\n', ' ') if response_text else ""
            logger.info(f"        [Prompt {i+1}] TTFT={ttft:.3f}s, abs_time={abs_ttft:.3f}s, output={completion_tok} tok")
            logger.info(f"                  Response: {snippet}...")

    # Log summary for unique prompts
    avg_ttft = np.mean([m.ttft for m in unique_metrics_list])
    avg_completion = np.mean([m.completion_tokens for m in unique_metrics_list])
    if concurrent_prompts > 1:
        abs_times = [r[6] for r in unique_results]
        abs_spread = max(abs_times) - min(abs_times)
        prefill_time = max(abs_times)  # Time until all prompts got first token
        effective_prefill_per_prompt = prefill_time / concurrent_prompts
        total_input_tokens = sum([m.prompt_tokens for m in unique_metrics_list])
        logger.info(f"      Unique batch summary:")
        logger.info(f"        avg TTFT={avg_ttft:.3f}s, abs_time spread={abs_spread:.3f}s")
        logger.info(f"        prefill time={prefill_time:.3f}s (until all first tokens), effective per-prompt={effective_prefill_per_prompt:.3f}s")
        logger.info(f"        total input tokens={total_input_tokens:,}, prefill throughput={total_input_tokens/prefill_time:.0f} tok/s")
    else:
        snippet = unique_results[0][0][:100].replace('\n', ' ') if unique_results[0][0] else ""
        logger.info(f"      Unique: TTFT={avg_ttft:.3f}s, Output={int(avg_completion)} tok")
        logger.info(f"      Response: {snippet}...")

    # Test cached prompts (same prompts again - 100% cache hit) - send all simultaneously
    cached_metrics_list = []

    for repeat in range(cached_repeats):
        repeat_label = f" (repeat {repeat + 1}/{cached_repeats})" if cached_repeats > 1 else ""
        logger.info(f"    Iteration {iteration + 1}: Testing {concurrent_prompts} cached {prompt_label}{repeat_label}...")

        batch_start_cached = time.time()
        cached_results = await asyncio.gather(*[send_and_measure(p, i, batch_start_cached) for i, p in enumerate(prompts)])
        batch_elapsed_cached = time.time() - batch_start_cached

        for i, (response_text, ttft, gen_time, prompt_tok, completion_tok, total_time, abs_ttft) in enumerate(cached_results):
            output_tps = completion_tok / gen_time if gen_time > 0 else 0
            metrics = PromptMetrics(
                context_size=context_size,
                iteration=iteration,
                prompt_type=f"cached_r{repeat + 1}" if cached_repeats > 1 else "cached",
                prompt_tokens=prompt_tok,
                completion_tokens=completion_tok,
                ttft=ttft,
                generation_time=gen_time,
                total_time=total_time,
                output_tokens_per_sec=output_tps
            )
            cached_metrics_list.append(metrics)

        # Log per-prompt details when concurrent
        if concurrent_prompts > 1:
            for i, (response_text, ttft, gen_time, prompt_tok, completion_tok, total_time, abs_ttft) in enumerate(cached_results):
                snippet = response_text[:80].replace('\n', ' ') if response_text else ""
                logger.info(f"        [Prompt {i+1}] TTFT={ttft:.3f}s, abs_time={abs_ttft:.3f}s, output={completion_tok} tok")
                logger.info(f"                  Response: {snippet}...")

        # Log summary for this cached repeat
        repeat_metrics = cached_metrics_list[-concurrent_prompts:]  # Get metrics from this repeat
        avg_cached_ttft = np.mean([m.ttft for m in repeat_metrics])
        avg_cached_completion = np.mean([m.completion_tokens for m in repeat_metrics])
        if concurrent_prompts > 1:
            abs_times_cached = [r[6] for r in cached_results]
            abs_spread_cached = max(abs_times_cached) - min(abs_times_cached)
            prefill_time_cached = max(abs_times_cached)  # Time until all prompts got first token
            effective_prefill_per_prompt_cached = prefill_time_cached / concurrent_prompts
            total_input_tokens_cached = sum([m.prompt_tokens for m in repeat_metrics])
            logger.info(f"      Cached{repeat_label} batch summary:")
            logger.info(f"        avg TTFT={avg_cached_ttft:.3f}s, abs_time spread={abs_spread_cached:.3f}s")
            logger.info(f"        prefill time={prefill_time_cached:.3f}s (until all first tokens), effective per-prompt={effective_prefill_per_prompt_cached:.3f}s")
            logger.info(f"        total input tokens={total_input_tokens_cached:,}, prefill throughput={total_input_tokens_cached/prefill_time_cached:.0f} tok/s")
        else:
            snippet = cached_results[0][0][:100].replace('\n', ' ') if cached_results[0][0] else ""
            logger.info(f"      Cached{repeat_label}: TTFT={avg_cached_ttft:.3f}s, Output={int(avg_cached_completion)} tok")
            logger.info(f"      Response: {snippet}...")

    # Overall cache speedup (using all cached metrics)
    overall_avg_cached_ttft = np.mean([m.ttft for m in cached_metrics_list])
    speedup = avg_ttft / overall_avg_cached_ttft if overall_avg_cached_ttft > 0 else 0
    logger.info(f"      Cache speedup: TTFT {speedup:.2f}x faster")

    return unique_metrics_list, cached_metrics_list


def generate_graphs(results: List[PromptMetrics], output_dir: str):
    """Generate visualizations"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([m.to_dict() for m in results])

    # Calculate aggregated metrics (mean across iterations)
    # Group all cached_* types together as "cached"
    agg_data = []
    for context_size in sorted(df['context_size'].unique()):
        # Unique prompts
        unique_data = df[(df['context_size'] == context_size) & (df['prompt_type'] == 'unique')]
        if len(unique_data) > 0:
            agg_data.append({
                'context_size': context_size,
                'prompt_type': 'unique',
                'avg_ttft': unique_data['ttft'].mean(),
                'std_ttft': unique_data['ttft'].std(),
                'min_ttft': unique_data['ttft'].min(),
                'max_ttft': unique_data['ttft'].max(),
                'avg_output_tps': unique_data['output_tokens_per_sec'].mean(),
                'num_iterations': len(unique_data)
            })

        # Cached prompts (all cached_* types combined)
        cached_data = df[(df['context_size'] == context_size) & (df['prompt_type'].str.startswith('cached'))]
        if len(cached_data) > 0:
            agg_data.append({
                'context_size': context_size,
                'prompt_type': 'cached',
                'avg_ttft': cached_data['ttft'].mean(),
                'std_ttft': cached_data['ttft'].std(),
                'min_ttft': cached_data['ttft'].min(),
                'max_ttft': cached_data['ttft'].max(),
                'avg_output_tps': cached_data['output_tokens_per_sec'].mean(),
                'num_iterations': len(cached_data)
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

    # Calculate aggregated metrics (group all cached_* types together)
    summary_data = []
    for context_size in sorted(df['context_size'].unique()):
        unique_data = df[(df['context_size'] == context_size) & (df['prompt_type'] == 'unique')]
        cached_data = df[(df['context_size'] == context_size) & (df['prompt_type'].str.startswith('cached'))]

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
    parser.add_argument("--concurrent-prompts", "-n", type=int, default=1,
                       help="Number of prompts to send simultaneously (default: 1)")
    parser.add_argument("--cached-repeats", "-r", type=int, default=1,
                       help="Number of times to repeat cached prompt after unique (default: 1)")
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
    parser.add_argument("--no-color", action="store_true",
                       help="Disable colored output (useful for light terminal backgrounds)")

    args = parser.parse_args()

    # Disable colors if requested
    if args.no_color:
        Colors.disable()

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

    concurrent_prompts = args.concurrent_prompts
    cached_repeats = args.cached_repeats
    if concurrent_prompts > 1:
        logger.info(f"{Colors.OKBLUE}Concurrent prompts per test: {concurrent_prompts}{Colors.ENDC}")
    if cached_repeats > 1:
        logger.info(f"{Colors.OKBLUE}Cached repeats per iteration: {cached_repeats}{Colors.ENDC}")

    logger.info("")
    logger.info(f"{Colors.PHASE}Testing {len(context_sizes)} context sizes: {context_sizes}{Colors.ENDC}")
    logger.info("")

    # Run tests
    all_results = []

    for context_size in context_sizes:
        logger.info(f"{Colors.OKCYAN}Testing context size: {context_size:,} tokens{Colors.ENDC}")

        context_unique_metrics = []
        context_cached_metrics = []

        for iteration in range(args.num_iterations):
            unique_metrics_list, cached_metrics_list = await test_single_prompt_pair(
                api_client, tokenizer, context_size, args.output_tokens, iteration, base_seed,
                concurrent_prompts=concurrent_prompts, cached_repeats=cached_repeats
            )
            all_results.extend(unique_metrics_list)
            all_results.extend(cached_metrics_list)
            context_unique_metrics.extend(unique_metrics_list)
            context_cached_metrics.extend(cached_metrics_list)
            logger.info("")  # Blank line between iterations

        # Per-context-size summary
        avg_unique_ttft = np.mean([m.ttft for m in context_unique_metrics])
        avg_cached_ttft = np.mean([m.ttft for m in context_cached_metrics])
        speedup = avg_unique_ttft / avg_cached_ttft if avg_cached_ttft > 0 else 0
        logger.info(f"{Colors.OKGREEN}  Summary for {context_size:,} tokens:{Colors.ENDC}")
        logger.info(f"{Colors.OKGREEN}    Avg Unique TTFT: {avg_unique_ttft:.3f}s | Avg Cached TTFT: {avg_cached_ttft:.3f}s | Speedup: {speedup:.2f}x{Colors.ENDC}")
        logger.info("")

        # Brief mode per-context summary
        if brief_mode:
            print(f"[{context_size:,}] unique={avg_unique_ttft:.3f}s cached={avg_cached_ttft:.3f}s speedup={speedup:.2f}x")

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
    logger.info("")
    logger.info(f"{Colors.PHASE}Generating visualizations...{Colors.ENDC}")
    generate_graphs(all_results, args.output_dir)
    generate_summary_table(all_results, args.output_dir)

    # Save metadata for index.html generator
    metadata = {
        "test_type": "single_prompt",
        "api_endpoint": args.api_endpoint,
        "detected_model": model,
        "context_sizes": context_sizes,
        "num_iterations": args.num_iterations,
        "concurrent_prompts": concurrent_prompts,
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

    # Final summary table
    logger.info("")
    logger.info(f"{Colors.HEADER}{'='*80}{Colors.ENDC}")
    logger.info(f"{Colors.HEADER}{Colors.BOLD}Final Summary - All Context Sizes{Colors.ENDC}")
    logger.info(f"{Colors.HEADER}{'='*80}{Colors.ENDC}")
    logger.info(f"{'Context Size':>14} | {'Unique TTFT':>12} | {'Cached TTFT':>12} | {'Speedup':>8}")
    logger.info(f"{'-'*14}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}")

    df_summary = pd.DataFrame([m.to_dict() for m in all_results])
    for ctx in sorted(df_summary['context_size'].unique()):
        unique_data = df_summary[(df_summary['context_size'] == ctx) & (df_summary['prompt_type'] == 'unique')]
        cached_data = df_summary[(df_summary['context_size'] == ctx) & (df_summary['prompt_type'].str.startswith('cached'))]
        if len(unique_data) > 0 and len(cached_data) > 0:
            u_ttft = unique_data['ttft'].mean()
            c_ttft = cached_data['ttft'].mean()
            speedup = u_ttft / c_ttft if c_ttft > 0 else 0
            logger.info(f"{ctx:>14,} | {u_ttft:>11.3f}s | {c_ttft:>11.3f}s | {speedup:>7.2f}x")

    logger.info(f"{'-'*14}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}")
    logger.info("")
    logger.info(f"{Colors.SUCCESS}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    logger.info(f"{Colors.SUCCESS}{Colors.BOLD}✓ Testing complete!{Colors.ENDC}")
    logger.info(f"{Colors.SUCCESS}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    logger.info(f"{Colors.OKGREEN}Results saved to: {args.output_dir}{Colors.ENDC}")
    logger.info("")

    # Brief mode final output
    if brief_mode:
        print()
        print("=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        print(f"model: {model}")
        print(f"endpoint: {args.api_endpoint}")
        print(f"seed: {base_seed}")
        print(f"output: {args.output_dir}")
        print()
        print(f"{'Context':>10} | {'Unique TTFT':>12} | {'Cached TTFT':>12} | {'Speedup':>8}")
        print(f"{'-'*10}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}")

        # Calculate averages per context size
        df = pd.DataFrame([m.to_dict() for m in all_results])
        for ctx in sorted(df['context_size'].unique()):
            unique_data = df[(df['context_size'] == ctx) & (df['prompt_type'] == 'unique')]
            cached_data = df[(df['context_size'] == ctx) & (df['prompt_type'].str.startswith('cached'))]
            if len(unique_data) > 0 and len(cached_data) > 0:
                u_ttft = unique_data['ttft'].mean()
                c_ttft = cached_data['ttft'].mean()
                speedup = u_ttft / c_ttft if c_ttft > 0 else 0
                print(f"{ctx:>10,} | {u_ttft:>11.3f}s | {c_ttft:>11.3f}s | {speedup:>7.2f}x")

        print(f"{'-'*10}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
