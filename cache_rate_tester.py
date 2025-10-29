#!/usr/bin/env python3
"""
Cache Rate Performance Testing Tool

Measures input tokens/s, time to first token (TTFT), and output tokens/s
at varying cache hit rates (0% to 100% in 5% increments) for OpenAI-compatible
inference APIs with automatic prompt caching.

Version: 1.6
Date: 2025-01-17
"""

__version__ = "1.6"
__date__ = "2025-01-17"

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd

# Imports will be checked at runtime
try:
    import openai
    from transformers import AutoTokenizer
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Please install: pip install openai transformers plotly pandas numpy")
    sys.exit(1)


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

    # Specific colors for our use case
    INFO = '\033[97m'        # Bright white
    DEBUG = '\033[90m'       # Dark gray
    METRIC = '\033[96m'      # Cyan for metrics
    SUCCESS = '\033[92m'     # Green for success
    PHASE = '\033[95m'       # Magenta for phase headers


# Question bank to ensure long responses from the model
# These questions are designed to prompt the model to generate lengthy responses
QUESTION_BANK = [
    "Can you tell me a very long story with a happy ending?",
    "Please write a detailed explanation of how computers work, including all the technical details you can think of.",
    "Can you describe in great detail what you would do if you had a whole day free?",
    "Please write a comprehensive guide on how to learn a new language from scratch.",
    "Can you tell me an elaborate story about a journey through space?",
    "Please explain in detail the history of human civilization from ancient times to today.",
    "Can you write a long story about friendship and adventure?",
    "Please describe in detail how you would plan the perfect vacation.",
    "Can you tell me everything you know about the ocean and marine life?",
    "Please write a detailed story about overcoming challenges and achieving success.",
    "Can you explain in great detail how the human body works?",
    "Please tell me a long and interesting story about time travel.",
    "Can you describe in detail what life might be like 100 years from now?",
    "Please write a comprehensive explanation of how weather works.",
    "Can you tell me a very detailed story about exploring a mysterious island?",
]


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


# Configure logging
def init_logger(name: str, level=logging.INFO) -> logging.Logger:
    """Initialize logger with console and file handlers"""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Console handler with colors
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(ColoredFormatter())
    logger.addHandler(console_handler)

    # File handler without colors (append mode for resume support)
    file_handler = logging.FileHandler('cache_rate_tester.log', mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    return logger


logger = init_logger(__name__)


# KV cache size per token for common models (in bytes, FP8 precision)
KV_CACHE_SIZES_FP8 = {
    'llama-3.3-70b': 160_000,  # 160 KB per token
    'llama-3.1-70b': 160_000,
    'qwen3-coder-30b': 60_000,  # 60 KB per token
    'qwen2.5-coder-32b': 60_000,
    'qwen3-coder-480b': 124_000,  # 124 KB per token
}


def estimate_kv_cache_size(model_name: str, num_tokens: int, precision: str = 'fp8') -> int:
    """
    Estimate KV cache size in bytes for a given model and token count

    Args:
        model_name: Model identifier (lowercase)
        num_tokens: Number of tokens to cache
        precision: 'fp8' or 'fp16'

    Returns:
        Estimated cache size in bytes
    """
    # Try to match model name to known models
    model_lower = model_name.lower()

    for known_model, size_fp8 in KV_CACHE_SIZES_FP8.items():
        if known_model in model_lower:
            size_per_token = size_fp8 if precision == 'fp8' else size_fp8 * 2
            return num_tokens * size_per_token

    # Default conservative estimate (assume large model like 70B)
    logger.warning(f"Unknown model '{model_name}', using conservative KV cache estimate (160 KB/token FP8)")
    size_per_token = 160_000 if precision == 'fp8' else 320_000
    return num_tokens * size_per_token


def format_bytes(bytes_size: int) -> str:
    """Format bytes as human-readable string"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


def validate_working_set_size(model_name: str, working_set_size: int,
                              available_cache_gb: Optional[float] = None):
    """
    Validate working set size against estimated KV cache requirements

    Args:
        model_name: Model identifier
        working_set_size: Number of tokens in working set
        available_cache_gb: Optional hint about available cache capacity
    """
    # Estimate KV cache size
    cache_size_fp8 = estimate_kv_cache_size(model_name, working_set_size, 'fp8')
    cache_size_fp16 = estimate_kv_cache_size(model_name, working_set_size, 'fp16')

    logger.info(f"KV Cache size estimates for {working_set_size:,} tokens:")
    logger.info(f"  FP8:  {format_bytes(cache_size_fp8)}")
    logger.info(f"  FP16: {format_bytes(cache_size_fp16)}")

    # Warn if working set is very large
    if cache_size_fp8 > 100 * 1024**3:  # > 100 GB
        logger.warning(f"⚠️  Working set requires {format_bytes(cache_size_fp8)} of KV cache (FP8)")
        logger.warning(f"   This may exceed typical HBM+DRAM capacity!")

    if available_cache_gb:
        available_bytes = available_cache_gb * 1024**3
        if cache_size_fp8 > available_bytes:
            logger.error(f"❌ Working set ({format_bytes(cache_size_fp8)}) exceeds "
                        f"available cache ({format_bytes(available_bytes)})")
            logger.error(f"   Reduce --working-set-size or expect cache thrashing!")
        elif cache_size_fp8 > available_bytes * 0.8:
            logger.warning(f"⚠️  Working set uses >{80}% of available cache")
            logger.warning(f"   May experience cache evictions during testing")


@dataclass
class TestConfig:
    """Configuration for the entire test suite"""
    api_endpoint: str
    context_sizes: List[int]
    working_set_size: int
    output_tokens: int
    max_ttft: float
    ttft_metric: str
    output_dir: str
    tokenizer_id: str
    test_duration: int
    ramp_duration: int
    reinit_strategy: str
    random_selection: bool
    num_retries: int
    start_concurrency: int
    concurrency_increment: int
    max_concurrency: int
    init_concurrency: int
    cache_hit_rates: List[int]
    skip_graphs: bool
    force_restart: bool
    verbose: bool
    chunk_size: int = 256
    seed: Optional[int] = None
    kv_cache_bytes: int = 2  # 1 or 2 bytes per element for KV cache calculation
    strict_time_window: bool = False  # Only include requests completed within duration window

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return asdict(self)

    def get_test_id(self) -> str:
        """Generate unique test ID from parameters"""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:16]


@dataclass
class RequestMetrics:
    """Metrics for a single request"""
    request_id: str
    phase_id: str  # Phase identifier: "RAMP_c2", "RETRY_1", etc.
    cache_hit_rate: int
    context_size: int
    cached_tokens: int
    unique_tokens: int
    output_tokens: int
    ttft: float
    ttlt: float  # Time To Last Token (total generation time)
    generation_time: float
    total_time: float
    launch_time: float
    finish_time: float
    concurrency_level: int
    itl: float  # Inter Token Latency (average time between tokens)

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class PhaseMetadata:
    """Metadata for a single test phase (ramp level or retry run)"""
    phase_type: str  # "RAMP" or "RETRY"
    phase_id: str    # "RAMP_c2", "RAMP_c4", "RETRY_1", "RETRY_2", etc.
    cache_hit_rate: int
    context_size: int
    concurrency_level: int
    start_time: float  # Unix timestamp when phase started
    end_time: float    # Unix timestamp when phase ended
    duration: float    # Actual duration in seconds
    num_requests_launched: int  # Requests launched during this phase
    num_requests_completed: int  # Requests completed during this phase

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class AggregatedMetrics:
    """Aggregated metrics for a cache hit rate test"""
    context_size: int
    cache_hit_rate: int
    input_tokens_per_sec: float
    output_tokens_per_sec: float
    avg_ttft: float
    median_ttft: float
    p95_ttft: float
    p99_ttft: float
    avg_ttlt: float  # Average Time To Last Token
    median_ttlt: float
    p95_ttlt: float
    p99_ttlt: float
    avg_output_tokens: float  # Average output tokens/s per request (generation speed per request)
    avg_itl: float  # Average Inter Token Latency
    median_itl: float
    p95_itl: float
    p99_itl: float
    peak_concurrency: int
    total_requests: int
    test_duration: float

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return asdict(self)


class ProgressTracker:
    """Tracks and persists test progress for resume capability"""

    def __init__(self, output_dir: str, config: TestConfig):
        self.output_dir = Path(output_dir)
        self.config = config
        self.progress_file = self.output_dir / "progress.json"
        self.state = self._load_or_create_state()

    def _load_or_create_state(self) -> dict:
        """Load existing progress or create new state"""
        if self.progress_file.exists() and not self.config.force_restart:
            try:
                with open(self.progress_file, 'r') as f:
                    state = json.load(f)

                    # Check if we can resume (allow subset of context sizes)
                    old_params = state.get('parameters', {})
                    current_params = self.config.to_dict()

                    # Compare all parameters except context_sizes (allow subset)
                    can_resume = True
                    for key in current_params:
                        if key == 'context_sizes':
                            continue  # Allow different context sizes
                        if key not in old_params or old_params[key] != current_params[key]:
                            can_resume = False
                            break

                    if can_resume:
                        # Filter completed tests to only those in current context_sizes
                        current_contexts = set(self.config.context_sizes)
                        filtered_tests = []
                        for test_key in state.get('completed_tests', []):
                            context_size = int(test_key.split('_')[0])
                            if context_size in current_contexts:
                                filtered_tests.append(test_key)

                        state['completed_tests'] = filtered_tests
                        state['parameters'] = current_params  # Update to current params
                        logger.info(f"Resuming from previous progress: {len(filtered_tests)} tests completed (subset resume)")
                        return state
                    else:
                        logger.warning("Progress file found but parameters don't match. Starting fresh.")
            except Exception as e:
                logger.error(f"Failed to load progress file: {e}")

        # Create new state
        return {
            'test_id': self.config.get_test_id(),
            'parameters': self.config.to_dict(),
            'completed_tests': [],
            'last_update': datetime.now(timezone.utc).isoformat()
        }

    def is_test_completed(self, context_size: int, cache_hit_rate: int) -> bool:
        """Check if a specific test is already completed"""
        test_key = f"{context_size}_{cache_hit_rate}"
        return test_key in self.state.get('completed_tests', [])

    def mark_test_completed(self, context_size: int, cache_hit_rate: int):
        """Mark a test as completed"""
        test_key = f"{context_size}_{cache_hit_rate}"
        if test_key not in self.state['completed_tests']:
            self.state['completed_tests'].append(test_key)
        self.state['last_update'] = datetime.now(timezone.utc).isoformat()
        self._save_state()

    def _save_state(self):
        """Save current state to disk"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.progress_file, 'w') as f:
            json.dump(self.state, f, indent=2)


class TokenizerManager:
    """Manages tokenizer loading and caching"""

    def __init__(self, tokenizer_id: str):
        self.tokenizer_id = tokenizer_id
        self.tokenizer = None
        self._load_tokenizer()

    def _load_tokenizer(self):
        """Load tokenizer from Hugging Face"""
        logger.info(f"Loading tokenizer: {self.tokenizer_id}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_id,
                trust_remote_code=True
            )
            # Disable model_max_length warning for models that support longer context than tokenizer thinks
            if hasattr(self.tokenizer, 'model_max_length'):
                # Set to very large value to avoid warnings when testing long contexts
                self.tokenizer.model_max_length = 1_000_000
            logger.info(f"Tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs (suppresses max length warnings)"""
        import warnings
        # Suppress the "Token indices sequence length is longer than..." warning
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Token indices sequence length.*")
            return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def generate_dummy_tokens(self, num_tokens: int, seed: Optional[int] = None, prompt_number: Optional[int] = None) -> List[int]:
        """
        Generate diverse dummy tokens to activate different experts in MoE models
        Uses a mix of natural language, code, and formatting tokens
        """
        if seed is not None:
            np.random.seed(seed)

        # Diverse vocabulary to activate different MoE experts
        # Mix of: natural language, code syntax, formatting, technical terms
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

        # Flatten all word pools with weights to create diverse distribution
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

        # Generate diverse text that will activate different experts
        # Create chunks of different "contexts" to further diversify
        chunks = []
        chunk_size = 100  # Create variety in chunks

        for _ in range((num_tokens * 3) // chunk_size):  # Oversample for tokenization
            # Randomly select chunk type
            chunk_type = np.random.choice(['natural', 'code', 'mixed'], p=[0.3, 0.4, 0.3])

            if chunk_type == 'natural':
                # Natural language sentence
                chunk_words = np.random.choice(word_pools['natural'], size=chunk_size // 2)
                chunks.append(' '.join(chunk_words) + '.')
            elif chunk_type == 'code':
                # Code-like structure
                func_name = np.random.choice(['process', 'calculate', 'transform', 'execute'])
                chunk_words = np.random.choice(word_pools['code_keywords'] + word_pools['technical'], size=chunk_size // 3)
                chunks.append(f"def {func_name}({', '.join(chunk_words[:3])}): return {' '.join(chunk_words[3:])}")
            else:
                # Mixed content
                chunk_words = np.random.choice(all_words, size=chunk_size // 2, p=weights)
                chunks.append(' '.join(chunk_words))

        text = '\n'.join(chunks)

        # Prepend prompt number if provided for clarity in logs
        if prompt_number is not None:
            text = f"# PROMPT_{prompt_number}\n" + text

        # Encode and trim to exact length
        tokens = self.encode(text)
        return tokens[:num_tokens]


class WorkingSet:
    """Manages the working set of pre-warmed prompts"""

    def __init__(self, context_size: int, working_set_size: int,
                 tokenizer: TokenizerManager, chunk_size: int = 256, seed: Optional[int] = None):
        self.context_size = context_size
        self.working_set_size = working_set_size
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.seed = seed

        # Round up context size to nearest chunk boundary
        self.rounded_context_size = int(np.ceil(context_size / chunk_size) * chunk_size)

        # Calculate number of prompts needed
        self.num_prompts = max(1, int(np.ceil(working_set_size / self.rounded_context_size)))

        # Warn if working set is smaller than context size
        actual_working_set = self.num_prompts * self.rounded_context_size
        if working_set_size < self.rounded_context_size:
            logger.warning(f"⚠️  Working set size ({working_set_size:,} tokens) is smaller than context size ({self.rounded_context_size:,} tokens)")
            logger.warning(f"   Using minimum of 1 prompt = {actual_working_set:,} tokens (this may not be what you intended)")
            logger.warning(f"   Consider increasing --working-set-size to at least {self.rounded_context_size:,} tokens")

        if self.rounded_context_size != context_size:
            logger.info(f"Context size {context_size} rounded up to {self.rounded_context_size} (chunk size: {chunk_size})")

        prompt_word = "prompt" if self.num_prompts == 1 else "prompts"
        logger.info(f"Working set: {self.num_prompts} {prompt_word} × {self.rounded_context_size} tokens = "
                   f"{actual_working_set:,} tokens")

        self.prompts: List[List[int]] = []
        self.current_index = 0

    def generate_prompts(self):
        """Generate all working set prompts"""
        logger.info(f"Generating working set: {self.num_prompts} prompts of {self.rounded_context_size} tokens each")

        self.prompts = []
        for i in range(self.num_prompts):
            # Use different seed for each prompt to ensure uniqueness
            prompt_seed = (self.seed + i) if self.seed is not None else None
            # Add prompt number for clarity in logs
            tokens = self.tokenizer.generate_dummy_tokens(self.rounded_context_size, seed=prompt_seed, prompt_number=i)
            self.prompts.append(tokens)

            if (i + 1) % 10 == 0:
                logger.info(f"  Generated {i + 1}/{self.num_prompts} prompts")

        logger.info(f"Working set generation complete")

    def get_next_prompt(self, random_selection: bool = False) -> List[int]:
        """Get next prompt from working set"""
        if random_selection:
            import random
            return random.choice(self.prompts)
        else:
            # Cycle through deterministically
            prompt = self.prompts[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.prompts)
            return prompt


class APIClient:
    """Manages OpenAI API client and requests"""

    def __init__(self, api_endpoint: str, model: str):
        self.api_endpoint = api_endpoint
        self.model = model
        # Ensure base_url ends with /v1 for OpenAI client
        base_url = api_endpoint.rstrip('/')
        if not base_url.endswith('/v1'):
            base_url = base_url + '/v1'
        self.client = openai.AsyncOpenAI(
            api_key="EMPTY",
            base_url=base_url
        )
        logger.info(f"API Client initialized: {api_endpoint} (base_url: {base_url})")

    async def detect_model(self) -> str:
        """Auto-detect model from API"""
        try:
            models_url = self.api_endpoint.rstrip('/') + '/v1/models'
            logger.info(f"Auto-detecting model from {models_url}")

            # Use synchronous client for model detection
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
            raise ValueError(f"Could not detect model from {self.api_endpoint}/v1/models")

    async def send_request(self, prompt: str, max_tokens: int) -> Tuple[str, float, float, float, int, int]:
        """
        Send a single request and return metrics
        Returns: (response_text, ttft, ttlt, generation_time, prompt_tokens, completion_tokens)
        """
        start_time = time.time()
        first_token_time = None
        last_token_time = None
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
                    if delta.content != "":
                        last_token_time = time.time()  # Track last token time
                    response_text += delta.content

            # Get usage from the last chunk
            prompt_tokens = chunk.usage.prompt_tokens if hasattr(chunk, 'usage') else 0
            completion_tokens = chunk.usage.completion_tokens if hasattr(chunk, 'usage') else 0

            ttft = (first_token_time - start_time) if first_token_time else 0.0
            ttlt = (last_token_time - start_time) if last_token_time else 0.0
            generation_time = (time.time() - first_token_time) if first_token_time else 0.0

            return response_text, ttft, ttlt, generation_time, prompt_tokens, completion_tokens

        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Cache Rate Performance Testing Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Required arguments
    parser.add_argument("--api-endpoint", type=str, required=True,
                       help="OpenAI-compatible API endpoint (e.g., http://localhost:8125)")
    parser.add_argument("--context-sizes", type=int, nargs='+', required=True,
                       help="Context lengths to test (e.g., 8000 32000 64000)")
    parser.add_argument("--working-set-size", type=int, required=True,
                       help="Working set size in tokens (e.g., 2000000)")

    # Optional arguments
    parser.add_argument("--output-tokens", type=int, default=256,
                       help="Output tokens per request (default: 256)")
    parser.add_argument("--max-ttft", type=float, default=2.0,
                       help="TTFT threshold in seconds (default: 2.0)")
    parser.add_argument("--ttft-metric", type=str, default="p95",
                       choices=["max", "avg", "p95"],
                       help="TTFT metric to use for threshold: max (maximum), avg (average), p95 (95th percentile). Default: p95")
    parser.add_argument("--output-dir", type=str, default="./output",
                       help="Output directory (default: ./output)")
    parser.add_argument("--tokenizer", type=str,
                       default="Qwen/Qwen2.5-Coder-32B-Instruct",
                       help="Tokenizer model ID (default: Qwen/Qwen2.5-Coder-32B-Instruct)")
    parser.add_argument("--test-duration", type=int, default=300,
                       help="Max duration per cache hit rate test in seconds (default: 300)")
    parser.add_argument("--ramp-duration", type=int, default=60,
                       help="Duration per concurrency level during ramp phase in seconds (default: 60)")
    parser.add_argument("--reinit-strategy", type=str, default="once",
                       choices=["once", "per_cache_rate", "per_test"],
                       help="When to reinitialize working set cache (default: once per context size). "
                            "Choices: 'once' = once per context size, 'per_cache_rate' = before each cache hit rate, "
                            "'per_test' = before each concurrency level")
    parser.add_argument("--random-working-set-selection", action="store_true",
                       help="Use random selection instead of cycling")
    parser.add_argument("--num-retries", type=int, default=3,
                       help="Number of retries at peak concurrency (default: 3)")
    parser.add_argument("--start-concurrency", type=int, default=2,
                       help="Starting concurrent requests (default: 2)")
    parser.add_argument("--concurrency-increment", type=int, default=2,
                       help="Concurrency increment (default: 2)")
    parser.add_argument("--max-concurrency", type=int, default=1000,
                       help="Maximum concurrent requests (default: 1000)")
    parser.add_argument("--init-concurrency", type=int, default=16,
                       help="Concurrent requests for working set initialization (default: 16)")
    parser.add_argument("--cache-hit-rates", type=int, nargs='+',
                       help="Override default cache hit rates (default: 0 5 10 ... 100)")
    parser.add_argument("--skip-graphs", action="store_true",
                       help="Don't generate graphs")
    parser.add_argument("--force-restart", action="store_true",
                       help="Ignore existing progress and restart")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--chunk-size", type=int, default=256,
                       help="Cache block size in tokens for rounding (default: 256)")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility")
    parser.add_argument("--kv-cache-quantization", type=int, default=2,
                       choices=[1, 2],
                       help="KV cache bytes per element for memory estimation (default: 2). "
                            "1 = FP8 (half memory), 2 = BF16/FP16 (standard). "
                            "Note: This is for ESTIMATION only. Actual KV cache size depends on server configuration.")
    parser.add_argument("--strict-time-window", action="store_true",
                       help="Calculate throughput using only requests that launched AND completed within the ramp-duration window. "
                            "Default behavior includes all requests in throughput calculation (including cleanup time).")

    return parser.parse_args()


def create_test_config(args: argparse.Namespace) -> TestConfig:
    """Create TestConfig from command line arguments"""

    # Validate input parameters
    if args.start_concurrency > args.max_concurrency:
        raise ValueError(f"start-concurrency ({args.start_concurrency}) must be <= max-concurrency ({args.max_concurrency})")

    if args.start_concurrency < 1:
        raise ValueError(f"start-concurrency must be >= 1 (got {args.start_concurrency})")

    if args.concurrency_increment < 1:
        raise ValueError(f"concurrency-increment must be >= 1 (got {args.concurrency_increment})")

    if args.ramp_duration < 1:
        raise ValueError(f"ramp-duration must be >= 1 second (got {args.ramp_duration})")

    if args.test_duration < args.ramp_duration:
        raise ValueError(f"test-duration ({args.test_duration}) should be >= ramp-duration ({args.ramp_duration})")

    # Generate default cache hit rates if not specified
    if args.cache_hit_rates is None:
        cache_hit_rates = list(range(0, 101, 5))  # 0, 5, 10, ..., 100
    else:
        cache_hit_rates = sorted(args.cache_hit_rates)

    return TestConfig(
        api_endpoint=args.api_endpoint,
        context_sizes=sorted(args.context_sizes),
        working_set_size=args.working_set_size,
        output_tokens=args.output_tokens,
        max_ttft=args.max_ttft,
        ttft_metric=args.ttft_metric,
        output_dir=args.output_dir,
        tokenizer_id=args.tokenizer,
        test_duration=args.test_duration,
        ramp_duration=args.ramp_duration,
        reinit_strategy=args.reinit_strategy,
        random_selection=args.random_working_set_selection,
        num_retries=args.num_retries,
        start_concurrency=args.start_concurrency,
        concurrency_increment=args.concurrency_increment,
        max_concurrency=args.max_concurrency,
        init_concurrency=args.init_concurrency,
        cache_hit_rates=cache_hit_rates,
        skip_graphs=args.skip_graphs,
        force_restart=args.force_restart,
        verbose=args.verbose,
        chunk_size=args.chunk_size,
        seed=args.seed,
        kv_cache_bytes=args.kv_cache_quantization,
        strict_time_window=args.strict_time_window
    )


def calculate_kv_cache_size(context_size: int, bytes_per_element: int,
                            num_kv_heads: int = 4, num_layers: int = 48,
                            head_dim: int = 64, batch_size: int = 1) -> dict:
    """
    Calculate KV cache memory requirements

    Default parameters are for Qwen3-Coder-30B-A3B-Instruct-FP8:
    - num_kv_heads: 4 (Grouped Query Attention)
    - num_layers: 48
    - head_dim: 64

    Args:
        context_size: Sequence length in tokens
        bytes_per_element: 1 (FP8) or 2 (BF16/FP16)
        num_kv_heads: Number of KV heads (GQA)
        num_layers: Number of transformer layers
        head_dim: Dimension of each attention head
        batch_size: Batch size (default: 1)

    Returns:
        Dictionary with memory breakdown
    """

    # Per-layer KV cache size
    per_layer_bytes = (
        2 *  # K and V
        batch_size *
        num_kv_heads *
        context_size *
        head_dim *
        bytes_per_element
    )

    # Total across all layers
    total_bytes = per_layer_bytes * num_layers

    # Determine dtype string for display
    dtype_str = "FP8" if bytes_per_element == 1 else "BF16/FP16"

    return {
        "context_size": context_size,
        "dtype": dtype_str,
        "bytes_per_element": bytes_per_element,
        "per_layer_bytes": per_layer_bytes,
        "per_layer_mb": per_layer_bytes / (1024 ** 2),
        "total_bytes": total_bytes,
        "total_mb": total_bytes / (1024 ** 2),
        "total_gb": total_bytes / (1024 ** 3),
        "num_kv_heads": num_kv_heads,
        "num_layers": num_layers,
        "head_dim": head_dim
    }


async def initialize_working_set(api_client: APIClient, working_set: WorkingSet,
                                output_tokens: int, max_concurrency: int = 16):
    """
    Initialize working set by sending all prompts to API with adaptive concurrency

    Args:
        api_client: API client
        working_set: Working set to initialize
        output_tokens: Output tokens per request
        max_concurrency: Maximum concurrent initialization requests (default: 16)
    """
    num_prompts = len(working_set.prompts)
    logger.info(f"Initializing working set: sending {num_prompts} prompts to API (max concurrency: {max_concurrency})...")

    start_time = time.time()
    completed = 0
    active_tasks = []
    prompt_index = 0

    # Progress tracking
    last_progress_time = start_time

    async def init_single_prompt(i: int, prompt_tokens: List[int]):
        """Initialize a single prompt"""
        prompt_text = working_set.tokenizer.decode(prompt_tokens)
        try:
            await api_client.send_request(prompt_text, output_tokens)
            return i, True
        except Exception as e:
            logger.error(f"  Failed to initialize prompt {i + 1}: {e}")
            return i, False

    # Initialize with adaptive concurrency
    while prompt_index < num_prompts or active_tasks:
        # Launch new requests up to max_concurrency
        while len(active_tasks) < max_concurrency and prompt_index < num_prompts:
            task = asyncio.create_task(
                init_single_prompt(prompt_index, working_set.prompts[prompt_index])
            )
            active_tasks.append(task)
            prompt_index += 1

        # Wait for at least one to complete
        if active_tasks:
            done, pending = await asyncio.wait(
                active_tasks, return_when=asyncio.FIRST_COMPLETED
            )
            active_tasks = list(pending)

            # Process completed tasks
            for task in done:
                try:
                    i, success = await task
                    if success:
                        completed += 1
                except Exception as e:
                    logger.error(f"  Initialization task failed: {e}")

            # Log progress every 5 seconds or every 10 prompts
            now = time.time()
            if (now - last_progress_time >= 5.0) or (completed % 10 == 0) or (completed == num_prompts):
                elapsed = now - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                remaining = num_prompts - completed
                eta = remaining / rate if rate > 0 else 0
                logger.info(f"  Initialized {completed}/{num_prompts} prompts ({rate:.1f} prompts/s, ETA: {eta:.0f}s)")
                last_progress_time = now

    elapsed = time.time() - start_time
    final_rate = num_prompts / elapsed if elapsed > 0 else 0
    logger.info(f"Working set initialization complete: {num_prompts} prompts in {elapsed:.1f}s ({final_rate:.1f} prompts/s)")


def construct_prompt(working_set: WorkingSet, tokenizer: TokenizerManager,
                     cache_hit_rate: int, context_size: int, random_selection: bool,
                     request_seed: Optional[int] = None, question_index: int = 0) -> Tuple[str, int, int]:
    """
    Construct a prompt with the specified cache hit rate
    Returns: (prompt_text, cached_tokens, unique_tokens)
    """
    # Calculate token counts and round cached tokens to chunk boundary
    cached_tokens_raw = int(context_size * cache_hit_rate / 100)
    chunk_size = working_set.chunk_size

    # Round cached tokens down to nearest chunk boundary
    cached_tokens = int(np.floor(cached_tokens_raw / chunk_size) * chunk_size)
    unique_tokens = context_size - cached_tokens

    if cache_hit_rate == 0:
        # 0% cache: all unique tokens
        tokens = tokenizer.generate_dummy_tokens(context_size, seed=request_seed)
    elif cache_hit_rate == 100:
        # 100% cache: use complete working set prompt (already rounded)
        tokens = working_set.get_next_prompt(random_selection)
    else:
        # Mixed: cache prefix + unique suffix
        base_prompt = working_set.get_next_prompt(random_selection)
        cached_prefix = base_prompt[:cached_tokens]
        unique_suffix = tokenizer.generate_dummy_tokens(unique_tokens, seed=request_seed)
        tokens = cached_prefix + unique_suffix

    # Convert to text
    prompt_text = tokenizer.decode(tokens)

    # Append a question from the bank to encourage long responses
    # Rotate through questions using question_index
    question = QUESTION_BANK[question_index % len(QUESTION_BANK)]
    prompt_text = prompt_text + "\n\n" + question

    return prompt_text, cached_tokens, unique_tokens


async def run_single_request(api_client: APIClient, prompt: str, max_tokens: int,
                            cache_hit_rate: int, context_size: int, cached_tokens: int,
                            unique_tokens: int, concurrency_level: int,
                            request_id: str, phase_id: str, verbose: bool = False) -> RequestMetrics:
    """Run a single request and return metrics"""
    launch_time = time.time()

    try:
        response_text, ttft, ttlt, gen_time, prompt_tok, completion_tok = await api_client.send_request(
            prompt, max_tokens
        )

        finish_time = time.time()
        total_time = finish_time - launch_time

        # Calculate Inter Token Latency (ITL)
        # ITL = (TTLT - TTFT) / (output_tokens - 1)
        # This gives average time between tokens (excluding first token)
        if completion_tok > 1 and ttlt > ttft:
            itl = (ttlt - ttft) / (completion_tok - 1)
        else:
            itl = 0.0

        # Output token tracking
        token_ratio = (completion_tok / max_tokens * 100) if max_tokens > 0 else 0

        # Always warn if output tokens are significantly below expected
        if completion_tok < max_tokens * 0.95:  # Less than 95% of expected
            logger.warning(f"      [{request_id}] Output tokens below expected: {completion_tok}/{max_tokens} "
                         f"({token_ratio:.1f}%) - TTFT: {ttft:.3f}s")

        # Verbose logging for all requests
        if verbose:
            logger.debug(f"      [{request_id}] Output tokens: {completion_tok}/{max_tokens} ({token_ratio:.1f}%) - "
                       f"TTFT: {ttft:.3f}s, TTLT: {ttlt:.3f}s, ITL: {itl*1000:.2f}ms")

        return RequestMetrics(
            request_id=request_id,
            phase_id=phase_id,
            cache_hit_rate=cache_hit_rate,
            context_size=context_size,
            cached_tokens=cached_tokens,
            unique_tokens=unique_tokens,
            output_tokens=completion_tok,
            ttft=ttft,
            ttlt=ttlt,
            generation_time=gen_time,
            total_time=total_time,
            launch_time=launch_time,
            finish_time=finish_time,
            concurrency_level=concurrency_level,
            itl=itl
        )
    except Exception as e:
        logger.error(f"Request {request_id} failed: {e}")
        raise


async def run_concurrency_level(api_client: APIClient, working_set: WorkingSet,
                               tokenizer: TokenizerManager, config: TestConfig,
                               context_size: int, cache_hit_rate: int,
                               concurrency: int, duration: float, phase_id: str) -> Tuple[List[RequestMetrics], PhaseMetadata]:
    """
    Run requests at a specific concurrency level for a duration
    Returns: (results, phase_metadata)
    """
    results = []
    active_tasks = []
    request_counter = 0
    start_time = time.time()
    num_launched = 0

    logger.debug(f"    Running concurrency level {concurrency} for {duration:.1f}s (phase: {phase_id})")

    try:
        while time.time() - start_time < duration:
            # Launch requests up to concurrency limit
            while len(active_tasks) < concurrency and time.time() - start_time < duration:
                request_id = f"{phase_id}_r{request_counter}"
                request_seed = (config.seed + request_counter) if config.seed is not None else None

                # Construct prompt with rotating question
                prompt, cached_tok, unique_tok = construct_prompt(
                    working_set, tokenizer, cache_hit_rate, context_size,
                    config.random_selection, request_seed, question_index=request_counter
                )

                # Launch request
                task = asyncio.create_task(
                    run_single_request(
                        api_client, prompt, config.output_tokens, cache_hit_rate,
                        context_size, cached_tok, unique_tok, concurrency, request_id, phase_id,
                        verbose=config.verbose
                    )
                )
                active_tasks.append(task)
                request_counter += 1
                num_launched += 1

                # Small delay to avoid overwhelming the API
                await asyncio.sleep(0.01)

            # Wait for at least one task to complete
            if active_tasks:
                done, pending = await asyncio.wait(
                    active_tasks, return_when=asyncio.FIRST_COMPLETED
                )
                active_tasks = list(pending)  # Convert set back to list

                for task in done:
                    try:
                        result = await task
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Task failed: {e}")

        # Wait for remaining tasks
        if active_tasks:
            cleanup_start = time.time()
            logger.debug(f"    Waiting for {len(active_tasks)} remaining tasks...")
            remaining_results = await asyncio.gather(*active_tasks, return_exceptions=True)
            cleanup_time = time.time() - cleanup_start

            # Warn if cleanup took too long
            if cleanup_time > 15:
                logger.warning(f"    ⚠ Cleanup took {cleanup_time:.1f}s to complete {len(active_tasks)} outstanding requests")

            for result in remaining_results:
                if isinstance(result, RequestMetrics):
                    results.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"Task failed: {result}")

    except Exception as e:
        logger.error(f"Error during concurrency test: {e}")
        # Cancel remaining tasks
        for task in active_tasks:
            task.cancel()
        raise

    # Create phase metadata
    end_time = time.time()
    actual_duration = end_time - start_time

    phase_type = "RAMP" if "RAMP" in phase_id else "RETRY"
    phase_metadata = PhaseMetadata(
        phase_type=phase_type,
        phase_id=phase_id,
        cache_hit_rate=cache_hit_rate,
        context_size=context_size,
        concurrency_level=concurrency,
        start_time=start_time,
        end_time=end_time,
        duration=actual_duration,
        num_requests_launched=num_launched,
        num_requests_completed=len(results)
    )

    logger.debug(f"    Completed {len(results)} requests at concurrency {concurrency} (phase: {phase_id})")
    return results, phase_metadata


async def run_cache_hit_rate_test(config: TestConfig, api_client: APIClient,
                                  working_set: WorkingSet, tokenizer: TokenizerManager,
                                  context_size: int, cache_hit_rate: int,
                                  previous_peak: Optional[int] = None) -> Tuple[List[RequestMetrics], AggregatedMetrics, List[PhaseMetadata]]:
    """
    Run the complete cache hit rate test with concurrency ramp
    Returns: (detailed_metrics, aggregated_metrics, phase_metadata_list)

    Args:
        previous_peak: Peak concurrency from previous cache hit rate test (for adaptive starting point)
    """
    all_metrics = []
    all_phases = []  # Track all phase metadata
    peak_metrics = []

    # Track all tested concurrency levels with their performance
    tested_concurrency_levels = []  # List of {concurrency, input_tps, output_tps, p95_ttft, metrics}

    # Determine starting concurrency based on previous peak
    # Higher cache hit rates typically allow higher concurrency, so start at previous peak
    if previous_peak is not None and previous_peak > config.start_concurrency:
        # Start at the previous peak since cache performance improves with higher hit rates
        start_concurrency = previous_peak
        logger.info(f"    Using adaptive starting concurrency: {start_concurrency} (previous peak: {previous_peak})")
    else:
        start_concurrency = config.start_concurrency

    peak_concurrency = start_concurrency

    # Phase 1: Ramp up concurrency to find peak
    logger.info(f"{Colors.PHASE}    Phase 1: Finding peak concurrency (time limit: {config.test_duration}s){Colors.ENDC}")
    test_start_time = time.time()
    current_concurrency = start_concurrency
    iteration_count = 0
    last_good_concurrency = start_concurrency  # Track the last concurrency that passed

    while current_concurrency <= config.max_concurrency:
        elapsed = time.time() - test_start_time

        # Check time limit
        if elapsed >= config.test_duration:
            logger.info(f"    Time limit reached. Peak concurrency: {peak_concurrency}")
            break

        # Run at this concurrency level (shorter duration for ramp phase)
        remaining_time = config.test_duration - elapsed
        ramp_duration = min(config.ramp_duration, remaining_time)
        if ramp_duration < 10:  # Not enough time left for meaningful test
            break

        phase_id = f"RAMP_c{current_concurrency}"
        logger.info(f"    Testing concurrency {current_concurrency}... (phase: {phase_id})")

        # Reinitialize if per_test strategy
        if config.reinit_strategy == "per_test":
            await initialize_working_set(api_client, working_set, config.output_tokens, config.init_concurrency)

        metrics, phase_metadata = await run_concurrency_level(
            api_client, working_set, tokenizer, config, context_size,
            cache_hit_rate, current_concurrency, ramp_duration, phase_id
        )

        all_metrics.extend(metrics)
        all_phases.append(phase_metadata)
        iteration_count += 1

        # Calculate throughput for this concurrency level
        start_time = min(m.launch_time for m in metrics)
        end_time = max(m.finish_time for m in metrics)
        test_duration = end_time - start_time

        total_input_tokens = sum(m.cached_tokens + m.unique_tokens for m in metrics)
        total_output_tokens = sum(m.output_tokens for m in metrics)

        input_tps = total_input_tokens / test_duration if test_duration > 0 else 0
        output_tps = total_output_tokens / test_duration if test_duration > 0 else 0

        # Check if TTFT threshold exceeded
        ttfts = [m.ttft for m in metrics if m.ttft > 0]
        avg_ttft = np.mean(ttfts) if ttfts else 0
        max_ttft = np.max(ttfts) if ttfts else 0
        p95_ttft = np.percentile(ttfts, 95) if ttfts else 0

        # Choose TTFT metric based on config
        if config.ttft_metric == "max":
            measured_ttft = max_ttft
            metric_name = "Max TTFT"
        elif config.ttft_metric == "avg":
            measured_ttft = avg_ttft
            metric_name = "Avg TTFT"
        else:  # p95
            measured_ttft = p95_ttft
            metric_name = "P95 TTFT"

        # Track this concurrency level's performance
        tested_concurrency_levels.append({
            'concurrency': current_concurrency,
            'input_tps': input_tps,
            'output_tps': output_tps,
            'measured_ttft': measured_ttft,
            'p95_ttft': p95_ttft,
            'metrics': metrics
        })

        logger.info(f"      Avg TTFT: {avg_ttft:.3f}s, P95 TTFT: {p95_ttft:.3f}s, Max TTFT: {max_ttft:.3f}s ({metric_name}: {measured_ttft:.3f}s)")
        logger.info(f"      Throughput: Input={input_tps:,.0f} tok/s, Output={output_tps:,.0f} tok/s")

        if measured_ttft > config.max_ttft:
            # TTFT threshold exceeded
            ttft_overshoot = (measured_ttft - config.max_ttft) / config.max_ttft

            # Smart backoff if we exceeded on the first iteration at adaptive start
            if iteration_count == 1 and start_concurrency > config.start_concurrency:
                # We started at previous peak but it's too high for this cache hit rate
                # Back off to midpoint between start_concurrency and current
                backoff_concurrency = (config.start_concurrency + start_concurrency) // 2
                # Round to nearest increment
                backoff_concurrency = (backoff_concurrency // config.concurrency_increment) * config.concurrency_increment
                backoff_concurrency = max(config.start_concurrency, backoff_concurrency)

                logger.warning(f"      ⚠ TTFT threshold exceeded on first test at adaptive start!")
                logger.warning(f"      Backing off from {start_concurrency} to {backoff_concurrency} and retrying...")

                # Reset and try again from backoff point
                current_concurrency = backoff_concurrency
                start_concurrency = backoff_concurrency
                last_good_concurrency = backoff_concurrency
                peak_concurrency = backoff_concurrency
                iteration_count = 0  # Reset iteration count to start fresh
                continue  # Go back to while loop and test at backoff concurrency

            # Warn if we still exceeded on the first iteration after backoff
            if iteration_count == 1:
                logger.warning(f"      ⚠ TTFT threshold exceeded on first test!")
                logger.warning(f"      Consider increasing --max-ttft (current: {config.max_ttft}s, observed: {max_ttft:.3f}s)")
                peak_concurrency = current_concurrency
                peak_metrics = metrics
                break

            # Do binary search refinement if there's a gap to refine
            gap = current_concurrency - last_good_concurrency
            if gap > config.concurrency_increment:
                logger.info(f"      TTFT overshoot detected ({ttft_overshoot:.1%}), gap of {gap}. Refining with binary search...")

                # Binary search between last_good and current
                low = last_good_concurrency
                high = current_concurrency
                best_concurrency = low
                best_metrics = peak_metrics
                refinement_iterations = 0
                max_refinement_iterations = 10  # Prevent excessive refinement

                while (high - low) > config.concurrency_increment and refinement_iterations < max_refinement_iterations:
                    # Calculate midpoint, rounded to nearest increment
                    mid = low + ((high - low) // (2 * config.concurrency_increment)) * config.concurrency_increment

                    # Ensure we make progress
                    if mid == low or mid == high:
                        # Try simple midpoint if rounding caused no progress
                        mid = (low + high) // 2
                        # Round to nearest increment
                        mid = (mid // config.concurrency_increment) * config.concurrency_increment
                        # Still no progress? Break
                        if mid == low or mid == high:
                            logger.info(f"      Binary search converged (no progress possible)")
                            break

                    # Check time limit
                    if time.time() - test_start_time >= config.test_duration:
                        logger.info(f"      Time limit reached during refinement")
                        break

                    refine_phase_id = f"RAMP_REFINE_c{mid}_{refinement_iterations + 1}"
                    logger.info(f"      Refining: testing concurrency {mid} (iteration {refinement_iterations + 1}, range: {low}-{high}, phase: {refine_phase_id})...")

                    if config.reinit_strategy == "per_test":
                        await initialize_working_set(api_client, working_set, config.output_tokens, config.init_concurrency)

                    refine_metrics, refine_phase_metadata = await run_concurrency_level(
                        api_client, working_set, tokenizer, config, context_size,
                        cache_hit_rate, mid, min(config.ramp_duration, config.test_duration - (time.time() - test_start_time)),
                        refine_phase_id
                    )

                    all_metrics.extend(refine_metrics)
                    all_phases.append(refine_phase_metadata)
                    refinement_iterations += 1

                    # Calculate TTFT metric for refinement
                    refine_ttfts = [m.ttft for m in refine_metrics if m.ttft > 0]
                    if config.ttft_metric == "max":
                        refine_measured_ttft = np.max(refine_ttfts) if refine_ttfts else 0
                    elif config.ttft_metric == "avg":
                        refine_measured_ttft = np.mean(refine_ttfts) if refine_ttfts else 0
                    else:  # p95
                        refine_measured_ttft = np.percentile(refine_ttfts, 95) if refine_ttfts else 0

                    refine_p95_ttft = np.percentile(refine_ttfts, 95) if refine_ttfts else 0

                    # Calculate throughput for this refinement level
                    refine_start_time = min(m.launch_time for m in refine_metrics)
                    refine_end_time = max(m.finish_time for m in refine_metrics)
                    refine_test_duration = refine_end_time - refine_start_time
                    refine_input_tps = sum(m.cached_tokens + m.unique_tokens for m in refine_metrics) / refine_test_duration if refine_test_duration > 0 else 0
                    refine_output_tps = sum(m.output_tokens for m in refine_metrics) / refine_test_duration if refine_test_duration > 0 else 0

                    # Track this refinement level's performance
                    tested_concurrency_levels.append({
                        'concurrency': mid,
                        'input_tps': refine_input_tps,
                        'output_tps': refine_output_tps,
                        'measured_ttft': refine_measured_ttft,
                        'p95_ttft': refine_p95_ttft,
                        'metrics': refine_metrics
                    })

                    logger.info(f"        {metric_name}: {refine_measured_ttft:.3f}s, Input: {refine_input_tps:,.0f} tok/s")

                    if refine_measured_ttft <= config.max_ttft:
                        # This level is good
                        low = mid
                        best_concurrency = mid
                        best_metrics = refine_metrics
                        logger.info(f"        ✓ Under threshold, new lower bound: {low}")
                    else:
                        # Too high
                        high = mid
                        logger.info(f"        ✗ Over threshold, new upper bound: {high}")

                peak_concurrency = best_concurrency
                peak_metrics = best_metrics
                logger.info(f"      Refinement complete after {refinement_iterations} iterations. Peak concurrency: {peak_concurrency}")
            else:
                logger.info(f"      Peak found at concurrency {peak_concurrency}")

            break
        else:
            # This level is good, save as peak
            last_good_concurrency = current_concurrency
            peak_concurrency = current_concurrency
            peak_metrics = metrics

        # Adaptive increment based on TTFT headroom
        # Calculate how much headroom we have (as a percentage of threshold)
        ttft_headroom = (config.max_ttft - measured_ttft) / config.max_ttft

        if ttft_headroom > 0.7:  # More than 70% headroom - increase very aggressively
            adaptive_increment = config.concurrency_increment * 10
            logger.info(f"      Large TTFT headroom ({ttft_headroom:.1%}), using 10x increment → +{adaptive_increment}")
        elif ttft_headroom > 0.5:  # 50-70% headroom - increase aggressively
            adaptive_increment = config.concurrency_increment * 5
            logger.info(f"      Good TTFT headroom ({ttft_headroom:.1%}), using 5x increment → +{adaptive_increment}")
        elif ttft_headroom > 0.3:  # 30-50% headroom - increase moderately
            adaptive_increment = config.concurrency_increment * 3
            logger.info(f"      Moderate TTFT headroom ({ttft_headroom:.1%}), using 3x increment → +{adaptive_increment}")
        elif ttft_headroom > 0.15:  # 15-30% headroom - increase cautiously
            adaptive_increment = config.concurrency_increment * 2
            logger.info(f"      Some TTFT headroom ({ttft_headroom:.1%}), using 2x increment → +{adaptive_increment}")
        else:  # Less than 15% headroom - increase minimally
            adaptive_increment = config.concurrency_increment
            logger.info(f"      Small TTFT headroom ({ttft_headroom:.1%}), using 1x increment → +{adaptive_increment}")

        # Increment concurrency
        current_concurrency += adaptive_increment

    # After ramp phase: Select best concurrency based on throughput
    logger.info(f"{Colors.PHASE}    Analyzing ramp results to select optimal concurrency{Colors.ENDC}")

    if len(tested_concurrency_levels) == 0:
        logger.warning("    No concurrency levels tested! Using start concurrency.")
        peak_concurrency = start_concurrency
        peak_metrics = []
    else:
        # Filter to concurrency levels that passed TTFT threshold
        passed_levels = [level for level in tested_concurrency_levels if level['measured_ttft'] <= config.max_ttft]

        if len(passed_levels) > 0:
            # Select the one with highest input throughput
            best_level = max(passed_levels, key=lambda x: x['input_tps'])
            peak_concurrency = best_level['concurrency']
            peak_metrics = best_level['metrics']

            logger.info(f"    ✓ Found {len(passed_levels)} concurrency levels under TTFT threshold")
            logger.info(f"    Selected concurrency {peak_concurrency} with best throughput: {best_level['input_tps']:,.0f} input tok/s")

            # Show top 3 candidates for context
            sorted_passed = sorted(passed_levels, key=lambda x: x['input_tps'], reverse=True)
            logger.info(f"    Top candidates under threshold:")
            for i, level in enumerate(sorted_passed[:3]):
                marker = "→" if level['concurrency'] == peak_concurrency else " "
                logger.info(f"      {marker} Conc {level['concurrency']}: {level['input_tps']:,.0f} tok/s (TTFT: {level['measured_ttft']:.3f}s)")
        else:
            # No levels passed threshold - pick the one with lowest TTFT
            best_level = min(tested_concurrency_levels, key=lambda x: x['measured_ttft'])
            peak_concurrency = best_level['concurrency']
            peak_metrics = best_level['metrics']

            logger.warning(f"    ⚠️  No concurrency levels passed TTFT threshold of {config.max_ttft}s!")
            logger.warning(f"    Selecting concurrency {peak_concurrency} with lowest TTFT: {best_level['measured_ttft']:.3f}s")
            logger.warning(f"    Consider increasing --max-ttft or reducing load")

            # Show all tested levels for debugging
            sorted_all = sorted(tested_concurrency_levels, key=lambda x: x['measured_ttft'])
            logger.info(f"    All tested levels (by TTFT):")
            for i, level in enumerate(sorted_all[:5]):
                marker = "→" if level['concurrency'] == peak_concurrency else " "
                logger.info(f"      {marker} Conc {level['concurrency']}: TTFT {level['measured_ttft']:.3f}s, {level['input_tps']:,.0f} tok/s")

    # Phase 2: Retry at peak concurrency
    logger.info(f"{Colors.PHASE}    Phase 2: Retrying at peak concurrency {peak_concurrency} ({config.num_retries} times){Colors.ENDC}")

    retry_results = [peak_metrics] if peak_metrics else []  # Include the first successful run if available
    num_retries_needed = config.num_retries - (1 if peak_metrics else 0)  # -1 if we already have one run

    # Track per-retry statistics for reporting
    retry_stats = []

    # Calculate stats for the original peak metrics (from ramp) if available
    if peak_metrics:
        start_time = min(m.launch_time for m in peak_metrics)
        end_time = max(m.finish_time for m in peak_metrics)
        duration = end_time - start_time

        total_input = sum(m.cached_tokens + m.unique_tokens for m in peak_metrics)
        total_output = sum(m.output_tokens for m in peak_metrics)
        ttfts = [m.ttft for m in peak_metrics if m.ttft > 0]

        retry_stats.append({
            'run': 'RAMP',
            'input_tps': total_input / duration if duration > 0 else 0,
            'output_tps': total_output / duration if duration > 0 else 0,
            'avg_ttft': np.mean(ttfts) if ttfts else 0,
            'p95_ttft': np.percentile(ttfts, 95) if ttfts else 0,
            'num_requests': len(peak_metrics)
        })

    for retry in range(num_retries_needed):
        retry_phase_id = f"RETRY_{retry + 1}"
        logger.info(f"    Retry {retry + 1}/{num_retries_needed}... (phase: {retry_phase_id})")

        # Reinitialize if per_test strategy
        if config.reinit_strategy == "per_test":
            await initialize_working_set(api_client, working_set, config.output_tokens, config.init_concurrency)

        metrics, retry_phase_metadata = await run_concurrency_level(
            api_client, working_set, tokenizer, config, context_size,
            cache_hit_rate, peak_concurrency, config.ramp_duration, retry_phase_id
        )

        retry_results.append(metrics)
        all_metrics.extend(metrics)
        all_phases.append(retry_phase_metadata)

        # Calculate and log stats for this retry
        start_time = min(m.launch_time for m in metrics)
        end_time = max(m.finish_time for m in metrics)
        duration = end_time - start_time

        total_input = sum(m.cached_tokens + m.unique_tokens for m in metrics)
        total_output = sum(m.output_tokens for m in metrics)
        ttfts = [m.ttft for m in metrics if m.ttft > 0]

        input_tps = total_input / duration if duration > 0 else 0
        output_tps = total_output / duration if duration > 0 else 0
        avg_ttft = np.mean(ttfts) if ttfts else 0
        p95_ttft = np.percentile(ttfts, 95) if ttfts else 0

        retry_stats.append({
            'run': f'RETRY{retry + 1}',
            'input_tps': input_tps,
            'output_tps': output_tps,
            'avg_ttft': avg_ttft,
            'p95_ttft': p95_ttft,
            'num_requests': len(metrics)
        })

        logger.info(f"{Colors.METRIC}      Input: {input_tps:,.0f} tok/s, Output: {output_tps:,.0f} tok/s{Colors.ENDC}")
        logger.info(f"{Colors.METRIC}      Avg TTFT: {avg_ttft:.3f}s, P95 TTFT: {p95_ttft:.3f}s{Colors.ENDC}")

    # Show summary of all retry runs
    if len(retry_stats) > 1:
        logger.info(f"")
        logger.info(f"{Colors.PHASE}    Retry Summary (Peak Concurrency {peak_concurrency}):{Colors.ENDC}")
        logger.info(f"{Colors.PHASE}    {'Run':<10} {'Input tok/s':>15} {'Output tok/s':>15} {'Avg TTFT':>12} {'P95 TTFT':>12} {'Requests':>10}{Colors.ENDC}")
        logger.info(f"{Colors.PHASE}    {'-'*80}{Colors.ENDC}")

        for stat in retry_stats:
            logger.info(f"    {stat['run']:<10} {stat['input_tps']:>15,.0f} {stat['output_tps']:>15,.0f} "
                       f"{stat['avg_ttft']:>12.3f}s {stat['p95_ttft']:>12.3f}s {stat['num_requests']:>10}")

        # Calculate averages across all retry runs
        avg_input_tps = np.mean([s['input_tps'] for s in retry_stats])
        avg_output_tps = np.mean([s['output_tps'] for s in retry_stats])
        avg_ttft_mean = np.mean([s['avg_ttft'] for s in retry_stats])
        avg_p95_ttft = np.mean([s['p95_ttft'] for s in retry_stats])

        # Calculate standard deviation to show variability
        std_input_tps = np.std([s['input_tps'] for s in retry_stats])
        std_output_tps = np.std([s['output_tps'] for s in retry_stats])

        logger.info(f"{Colors.PHASE}    {'-'*80}{Colors.ENDC}")
        logger.info(f"{Colors.SUCCESS}    {'AVERAGE':<10} {avg_input_tps:>15,.0f} {avg_output_tps:>15,.0f} "
                   f"{avg_ttft_mean:>12.3f}s {avg_p95_ttft:>12.3f}s{Colors.ENDC}")
        logger.info(f"{Colors.METRIC}    {'STD DEV':<10} {std_input_tps:>15,.0f} {std_output_tps:>15,.0f} "
                   f"{'(±' + f'{std_input_tps/avg_input_tps*100:.1f}' + '%)':<12} {'(±' + f'{std_output_tps/avg_output_tps*100:.1f}' + '%)':<12}{Colors.ENDC}")
        logger.info(f"")

    # Calculate aggregated metrics
    aggregated = calculate_aggregated_metrics(
        all_metrics, context_size, cache_hit_rate, peak_concurrency, config
    )

    return all_metrics, aggregated, all_phases


def calculate_aggregated_metrics(metrics: List[RequestMetrics], context_size: int,
                                 cache_hit_rate: int, peak_concurrency: int,
                                 config: TestConfig) -> AggregatedMetrics:
    """
    Calculate aggregated metrics from detailed results

    Note: Filters to only include requests at peak concurrency to match the Retry Summary
    statistics shown to the user. This excludes ramp-up requests at lower concurrency levels.
    """
    if not metrics:
        raise ValueError(
            "No metrics to aggregate. This usually means:\n"
            "  1. Test duration was too short (try increasing --ramp-duration or --test-duration)\n"
            "  2. API server is not responding\n"
            "  3. All requests failed\n"
            "Current parameters: ramp_duration={}, test_duration={}".format(
                config.ramp_duration, config.test_duration
            )
        )

    # Filter to only peak concurrency requests (matching what's shown in Retry Summary)
    # This ensures the final aggregated metrics match what the user sees in the summary table
    peak_metrics = [m for m in metrics if m.concurrency_level == peak_concurrency]

    if not peak_metrics:
        logger.warning(f"No metrics found at peak concurrency {peak_concurrency}, using all metrics")
        peak_metrics = metrics
    else:
        logger.debug(f"Calculating aggregated metrics from {len(peak_metrics)} requests at peak concurrency "
                    f"(excluding {len(metrics) - len(peak_metrics)} ramp requests)")

    # Use peak_metrics instead of all metrics for calculations
    metrics = peak_metrics

    # Calculate throughput per-phase and average (matching Retry Summary calculation)
    # Group by phase_id to calculate per-phase throughput
    phase_throughputs = {'input': [], 'output': []}
    phase_ttft_stats = {'avg': [], 'median': [], 'p95': [], 'p99': []}
    phase_ttlt_stats = {'avg': [], 'median': [], 'p95': [], 'p99': []}
    phase_itl_stats = {'avg': [], 'median': [], 'p95': [], 'p99': []}

    unique_phase_ids = set(m.phase_id for m in metrics)

    for phase_id in unique_phase_ids:
        phase_metrics = [m for m in metrics if m.phase_id == phase_id]

        if not phase_metrics:
            continue

        # Calculate phase duration
        phase_start = min(m.launch_time for m in phase_metrics)
        phase_end = max(m.finish_time for m in phase_metrics)
        phase_duration = phase_end - phase_start

        if phase_duration > 0:
            # Calculate throughput for this phase
            total_input = sum(m.cached_tokens + m.unique_tokens for m in phase_metrics)
            total_output = sum(m.output_tokens for m in phase_metrics)

            phase_throughputs['input'].append(total_input / phase_duration)
            phase_throughputs['output'].append(total_output / phase_duration)

        # Calculate TTFT stats for this phase
        phase_ttfts = [m.ttft for m in phase_metrics if m.ttft > 0]
        if phase_ttfts:
            phase_ttft_stats['avg'].append(np.mean(phase_ttfts))
            phase_ttft_stats['median'].append(np.median(phase_ttfts))
            phase_ttft_stats['p95'].append(np.percentile(phase_ttfts, 95))
            phase_ttft_stats['p99'].append(np.percentile(phase_ttfts, 99))

        # Calculate TTLT stats for this phase
        phase_ttlts = [m.ttlt for m in phase_metrics if m.ttlt > 0]
        if phase_ttlts:
            phase_ttlt_stats['avg'].append(np.mean(phase_ttlts))
            phase_ttlt_stats['median'].append(np.median(phase_ttlts))
            phase_ttlt_stats['p95'].append(np.percentile(phase_ttlts, 95))
            phase_ttlt_stats['p99'].append(np.percentile(phase_ttlts, 99))

        # Calculate ITL stats for this phase
        phase_itls = [m.itl for m in phase_metrics if m.itl > 0]
        if phase_itls:
            phase_itl_stats['avg'].append(np.mean(phase_itls))
            phase_itl_stats['median'].append(np.median(phase_itls))
            phase_itl_stats['p95'].append(np.percentile(phase_itls, 95))
            phase_itl_stats['p99'].append(np.percentile(phase_itls, 99))

    # Average the per-phase statistics
    input_tokens_per_sec = np.mean(phase_throughputs['input']) if phase_throughputs['input'] else 0
    output_tokens_per_sec = np.mean(phase_throughputs['output']) if phase_throughputs['output'] else 0

    avg_ttft = np.mean(phase_ttft_stats['avg']) if phase_ttft_stats['avg'] else 0
    median_ttft = np.mean(phase_ttft_stats['median']) if phase_ttft_stats['median'] else 0
    p95_ttft = np.mean(phase_ttft_stats['p95']) if phase_ttft_stats['p95'] else 0
    p99_ttft = np.mean(phase_ttft_stats['p99']) if phase_ttft_stats['p99'] else 0

    avg_ttlt = np.mean(phase_ttlt_stats['avg']) if phase_ttlt_stats['avg'] else 0
    median_ttlt = np.mean(phase_ttlt_stats['median']) if phase_ttlt_stats['median'] else 0
    p95_ttlt = np.mean(phase_ttlt_stats['p95']) if phase_ttlt_stats['p95'] else 0
    p99_ttlt = np.mean(phase_ttlt_stats['p99']) if phase_ttlt_stats['p99'] else 0

    avg_itl = np.mean(phase_itl_stats['avg']) if phase_itl_stats['avg'] else 0
    median_itl = np.mean(phase_itl_stats['median']) if phase_itl_stats['median'] else 0
    p95_itl = np.mean(phase_itl_stats['p95']) if phase_itl_stats['p95'] else 0
    p99_itl = np.mean(phase_itl_stats['p99']) if phase_itl_stats['p99'] else 0

    # Calculate test duration (for metadata purposes, not used for throughput)
    start_time = min(m.launch_time for m in metrics)
    end_time = max(m.finish_time for m in metrics)
    test_duration = end_time - start_time

    # Calculate strict time window metrics for comparison (even if not enabled)
    # This shows what throughput would be if we only counted requests that started AND finished within ramp window
    strict_phase_throughputs = {'input': [], 'output': []}

    for phase_id in unique_phase_ids:
        phase_metrics = [m for m in metrics if m.phase_id == phase_id]

        if not phase_metrics:
            continue

        # Find the time window for this phase (assume ramp_duration)
        phase_start = min(m.launch_time for m in phase_metrics)
        duration_end = phase_start + config.ramp_duration

        # Filter to only requests that launched AND finished within the window
        windowed_metrics = [m for m in phase_metrics
                           if m.launch_time < duration_end and m.finish_time <= duration_end]

        if windowed_metrics:
            # Calculate duration based on actual request times within window
            window_start = min(m.launch_time for m in windowed_metrics)
            window_end = max(m.finish_time for m in windowed_metrics)
            window_duration = window_end - window_start

            if window_duration > 0:
                total_input = sum(m.cached_tokens + m.unique_tokens for m in windowed_metrics)
                total_output = sum(m.output_tokens for m in windowed_metrics)

                strict_phase_throughputs['input'].append(total_input / window_duration)
                strict_phase_throughputs['output'].append(total_output / window_duration)

    # Calculate strict window averages
    strict_input_tps = np.mean(strict_phase_throughputs['input']) if strict_phase_throughputs['input'] else 0
    strict_output_tps = np.mean(strict_phase_throughputs['output']) if strict_phase_throughputs['output'] else 0

    # Log comparison if values differ significantly
    if strict_input_tps > 0 and abs(strict_input_tps - input_tokens_per_sec) / input_tokens_per_sec > 0.02:
        input_diff_pct = (strict_input_tps - input_tokens_per_sec) / input_tokens_per_sec * 100
        output_diff_pct = (strict_output_tps - output_tokens_per_sec) / output_tokens_per_sec * 100 if output_tokens_per_sec > 0 else 0

        logger.info(f"  ℹ️  Strict time window comparison (requests that started AND finished within ramp window):")
        logger.info(f"      Default: Input={input_tokens_per_sec:,.0f} tok/s, Output={output_tokens_per_sec:,.0f} tok/s")
        logger.info(f"      Strict:  Input={strict_input_tps:,.0f} tok/s ({input_diff_pct:+.1f}%), Output={strict_output_tps:,.0f} tok/s ({output_diff_pct:+.1f}%)")
        if input_diff_pct > 5:
            logger.info(f"      ⚠️  Strict window shows >5% higher throughput - may indicate cleanup overhead")

    # If strict_time_window is enabled, filter to only requests that completed within duration windows
    if config.strict_time_window:
        # Group by concurrency level and filter each level's requests
        filtered_metrics = []
        for m in metrics:
            # For strict mode: only include requests that both launched AND completed within their duration window
            # We approximate the duration window by looking at the earliest launch time for this concurrency level
            # and assuming a window based on ramp_duration
            filtered_metrics.append(m)

        # Actually, we need to be smarter - we need to know the actual duration_end_time per level
        # Since we don't track that per-metric, we'll use a simpler approach:
        # Filter out requests that finished significantly after the main batch
        # by grouping by concurrency level and finding the duration window

        from collections import defaultdict
        by_concurrency = defaultdict(list)
        for m in metrics:
            by_concurrency[m.concurrency_level].append(m)

        filtered_metrics = []
        for conc_level, level_metrics in by_concurrency.items():
            # Find the time window for this concurrency level
            level_start = min(m.launch_time for m in level_metrics)
            # Assume duration is ramp_duration (most common case)
            duration_end = level_start + config.ramp_duration

            # Only include requests that launched before duration_end AND finished before duration_end
            for m in level_metrics:
                if m.launch_time < duration_end and m.finish_time <= duration_end:
                    filtered_metrics.append(m)

        if not filtered_metrics:
            # If filtering removed all metrics, log warning and use original
            logger.warning("⚠ Strict time window filtering removed all metrics, using all requests")
            filtered_metrics = metrics
        else:
            excluded_count = len(metrics) - len(filtered_metrics)
            if excluded_count > 0:
                logger.info(f"  Strict time window: excluded {excluded_count} requests that didn't complete within duration window")

        metrics = filtered_metrics

    # Calculate average output tokens per second per request
    # For each request: output_tokens / generation_time = tokens/s for that request
    # Then average across all requests
    output_tokens_per_sec_per_request = [
        m.output_tokens / m.generation_time
        for m in metrics
        if m.output_tokens > 0 and m.generation_time > 0
    ]
    avg_output_tokens_per_sec = np.mean(output_tokens_per_sec_per_request) if output_tokens_per_sec_per_request else 0

    return AggregatedMetrics(
        context_size=context_size,
        cache_hit_rate=cache_hit_rate,
        input_tokens_per_sec=input_tokens_per_sec,
        output_tokens_per_sec=output_tokens_per_sec,
        avg_ttft=avg_ttft,
        median_ttft=median_ttft,
        p95_ttft=p95_ttft,
        p99_ttft=p99_ttft,
        avg_ttlt=avg_ttlt,
        median_ttlt=median_ttlt,
        p95_ttlt=p95_ttlt,
        p99_ttlt=p99_ttlt,
        avg_output_tokens=avg_output_tokens_per_sec,
        avg_itl=avg_itl,
        median_itl=median_itl,
        p95_itl=p95_itl,
        p99_itl=p99_itl,
        peak_concurrency=peak_concurrency,
        total_requests=len(metrics),
        test_duration=test_duration
    )


def save_detailed_results(metrics: List[RequestMetrics], output_dir: str, context_size: int):
    """Save detailed per-request metrics to CSV"""
    if not metrics:
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_path / f"detailed_results_{context_size}_{timestamp}.csv"

    df = pd.DataFrame([m.to_dict() for m in metrics])
    df.to_csv(filename, index=False)
    logger.debug(f"Saved detailed results to {filename}")


def save_aggregated_results(metrics: List[AggregatedMetrics], output_dir: str):
    """Save aggregated summary metrics to CSV"""
    if not metrics:
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_path / f"summary_{timestamp}.csv"

    df = pd.DataFrame([m.to_dict() for m in metrics])
    df.to_csv(filename, index=False)
    logger.info(f"Saved summary results to {filename}")


def save_phase_metadata(phases: List[PhaseMetadata], output_dir: str):
    """Save phase metadata to CSV"""
    if not phases:
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_path / f"phase_metadata_{timestamp}.csv"

    df = pd.DataFrame([p.to_dict() for p in phases])
    df.to_csv(filename, index=False)
    logger.debug(f"Saved phase metadata to {filename}")


def load_existing_aggregated_results(output_dir: str) -> List[AggregatedMetrics]:
    """Load existing aggregated results from most recent summary CSV"""
    output_path = Path(output_dir)
    summary_files = list(output_path.glob("summary_*.csv"))

    if not summary_files:
        return []

    # Get most recent summary file
    most_recent = max(summary_files, key=lambda p: p.stat().st_mtime)

    try:
        df = pd.read_csv(most_recent)
        metrics = []
        for _, row in df.iterrows():
            metrics.append(AggregatedMetrics(
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
            ))
        logger.debug(f"Loaded {len(metrics)} aggregated results from {most_recent.name}")
        return metrics
    except Exception as e:
        logger.warning(f"Failed to load existing aggregated results: {e}")
        return []


def load_phase_metadata(output_dir: str) -> Dict[Tuple[int, int], List[PhaseMetadata]]:
    """
    Load all phase metadata from output directory
    Returns: Dict mapping (context_size, cache_hit_rate) -> List[PhaseMetadata]
    """
    output_path = Path(output_dir)
    phase_files = list(output_path.glob("phase_metadata_*.csv"))

    phases_by_test = {}  # (context_size, cache_rate) -> List[PhaseMetadata]

    for phase_file in phase_files:
        try:
            df = pd.read_csv(phase_file)
            for _, row in df.iterrows():
                context_size = int(row['context_size'])
                cache_rate = int(row['cache_hit_rate'])
                key = (context_size, cache_rate)

                phase = PhaseMetadata(
                    phase_type=row['phase_type'],
                    phase_id=row['phase_id'],
                    cache_hit_rate=cache_rate,
                    context_size=context_size,
                    concurrency_level=int(row['concurrency_level']),
                    start_time=float(row['start_time']),
                    end_time=float(row['end_time']),
                    duration=float(row['duration']),
                    num_requests_launched=int(row['num_requests_launched']),
                    num_requests_completed=int(row['num_requests_completed'])
                )

                if key not in phases_by_test:
                    phases_by_test[key] = []
                phases_by_test[key].append(phase)

        except Exception as e:
            logger.warning(f"Failed to load phase metadata from {phase_file.name}: {e}")

    logger.debug(f"Loaded phase metadata for {len(phases_by_test)} tests")
    return phases_by_test


def save_run_command(args: argparse.Namespace, output_dir: str):
    """Save the command-line arguments to a file for easy re-running"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_path / f"run_command_{timestamp}.sh"

    # Build the command string
    script_name = "cache_rate_tester.py"
    command_parts = [f"python {script_name}"]

    # Add all arguments
    command_parts.append(f"  --api-endpoint {args.api_endpoint}")
    command_parts.append(f"  --context-sizes {' '.join(map(str, args.context_sizes))}")
    command_parts.append(f"  --working-set-size {args.working_set_size}")

    # Optional arguments (only include if non-default)
    if args.output_tokens != 256:
        command_parts.append(f"  --output-tokens {args.output_tokens}")
    if args.max_ttft != 2.0:
        command_parts.append(f"  --max-ttft {args.max_ttft}")
    if args.output_dir != "./output":
        command_parts.append(f"  --output-dir {args.output_dir}")
    if args.tokenizer != "Qwen/Qwen2.5-Coder-32B-Instruct":
        command_parts.append(f"  --tokenizer {args.tokenizer}")
    if args.test_duration != 300:
        command_parts.append(f"  --test-duration {args.test_duration}")
    if args.ramp_duration != 60:
        command_parts.append(f"  --ramp-duration {args.ramp_duration}")
    if args.reinit_strategy != "once":
        command_parts.append(f"  --reinit-strategy {args.reinit_strategy}")
    if args.random_working_set_selection:
        command_parts.append(f"  --random-working-set-selection")
    if args.num_retries != 3:
        command_parts.append(f"  --num-retries {args.num_retries}")
    if args.start_concurrency != 2:
        command_parts.append(f"  --start-concurrency {args.start_concurrency}")
    if args.concurrency_increment != 2:
        command_parts.append(f"  --concurrency-increment {args.concurrency_increment}")
    if args.max_concurrency != 1000:
        command_parts.append(f"  --max-concurrency {args.max_concurrency}")
    # Only add cache-hit-rates if it's not the default
    default_rates = list(range(0, 101, 5))
    if args.cache_hit_rates != default_rates:
        command_parts.append(f"  --cache-hit-rates {' '.join(map(str, args.cache_hit_rates))}")
    if args.skip_graphs:
        command_parts.append(f"  --skip-graphs")
    if args.verbose:
        command_parts.append(f"  --verbose")
    if args.chunk_size != 256:
        command_parts.append(f"  --chunk-size {args.chunk_size}")
    if args.seed is not None:
        command_parts.append(f"  --seed {args.seed}")
    if hasattr(args, 'kv_cache_quantization') and args.kv_cache_quantization != 2:
        command_parts.append(f"  --kv-cache-quantization {args.kv_cache_quantization}")
    if hasattr(args, 'strict_time_window') and args.strict_time_window:
        command_parts.append(f"  --strict-time-window")

    command_str = " \\\n".join(command_parts)

    # Write to file
    with open(filename, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write(f"# Cache Rate Tester - Run command\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# To re-run this exact test configuration, execute:\n")
        f.write(f"#   bash {filename.name}\n")
        f.write(f"# or\n")
        f.write(f"#   bash {filename.name} --force-restart  # to restart from beginning\n")
        f.write("\n")
        f.write(command_str)
        f.write(" $@\n")  # Allow passing additional arguments

    # Make executable
    os.chmod(filename, 0o755)

    logger.info(f"Saved run command to {filename}")
    logger.info(f"To re-run this test: bash {filename.name}")


def generate_ramp_graph(detailed_metrics: List[RequestMetrics], context_size: int,
                       cache_hit_rate: int, max_ttft: float, peak_concurrency: int, output_dir: str):
    """
    Generate detailed concurrency ramp visualization for a single cache hit rate test

    Shows ONLY the ramp phase data (excludes retry runs) to make it clear why each
    concurrency level was chosen or rejected during the ramp.

    Naming scheme: ramp_ctx{context_size}_cache{cache_hit_rate}.html
    Example: ramp_ctx30000_cache50.html (30K context, 50% cache hit rate)
    """
    if not detailed_metrics:
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Filter to ONLY RAMP phases (exclude RETRY phases)
    # This shows the actual ramp behavior that determined peak concurrency
    df = pd.DataFrame([m.to_dict() for m in detailed_metrics])
    ramp_df = df[df['phase_id'].str.startswith('RAMP')].copy()

    if len(ramp_df) == 0:
        logger.warning(f"No RAMP phase data found for ramp graph (context={context_size}, cache_rate={cache_hit_rate})")
        return

    # Calculate metrics per concurrency level using ONLY ramp data
    concurrency_levels = sorted(ramp_df['concurrency_level'].unique())
    level_data = []

    for level in concurrency_levels:
        level_df = ramp_df[ramp_df['concurrency_level'] == level]

        # Calculate duration for this ramp level
        start_time = level_df['launch_time'].min()
        end_time = level_df['finish_time'].max()
        duration = end_time - start_time

        if duration > 0:
            # Calculate throughput for this ramp level
            total_input = (level_df['cached_tokens'] + level_df['unique_tokens']).sum()
            total_output = level_df['output_tokens'].sum()

            input_tps = total_input / duration
            output_tps = total_output / duration
        else:
            input_tps = 0
            output_tps = 0

        level_data.append({
            'concurrency': level,
            'input_tokens_per_sec': input_tps,
            'output_tokens_per_sec': output_tps,
            'avg_ttft': level_df['ttft'].mean(),
            'max_ttft': level_df['ttft'].max(),
            'p95_ttft': level_df['ttft'].quantile(0.95),
            'num_requests': len(level_df)
        })

    level_df = pd.DataFrame(level_data)

    # Create subplot figure with 3 rows
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            'Input Throughput vs Concurrency',
            'Output Throughput vs Concurrency',
            'TTFT Metrics vs Concurrency'
        ),
        vertical_spacing=0.12,
        specs=[[{"secondary_y": False}],
               [{"secondary_y": False}],
               [{"secondary_y": False}]]
    )

    # Row 1: Input tokens/s
    fig.add_trace(
        go.Scatter(
            x=level_df['concurrency'],
            y=level_df['input_tokens_per_sec'],
            mode='lines+markers',
            name='Input tokens/s',
            line=dict(color='blue', width=3),
            marker=dict(size=8),
            hovertemplate='Concurrency: %{x}<br>Input: %{y:,.0f} tok/s<extra></extra>'
        ),
        row=1, col=1
    )

    # Row 2: Output tokens/s
    fig.add_trace(
        go.Scatter(
            x=level_df['concurrency'],
            y=level_df['output_tokens_per_sec'],
            mode='lines+markers',
            name='Output tokens/s',
            line=dict(color='green', width=3),
            marker=dict(size=8),
            hovertemplate='Concurrency: %{x}<br>Output: %{y:,.0f} tok/s<extra></extra>'
        ),
        row=2, col=1
    )

    # Row 3: TTFT metrics (avg, max, p95)
    fig.add_trace(
        go.Scatter(
            x=level_df['concurrency'],
            y=level_df['avg_ttft'],
            mode='lines+markers',
            name='Avg TTFT',
            line=dict(color='orange', width=2),
            marker=dict(size=6),
            hovertemplate='Concurrency: %{x}<br>Avg TTFT: %{y:.3f}s<extra></extra>'
        ),
        row=3, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=level_df['concurrency'],
            y=level_df['max_ttft'],
            mode='lines+markers',
            name='Max TTFT',
            line=dict(color='red', width=2),
            marker=dict(size=6),
            hovertemplate='Concurrency: %{x}<br>Max TTFT: %{y:.3f}s<extra></extra>'
        ),
        row=3, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=level_df['concurrency'],
            y=level_df['p95_ttft'],
            mode='lines+markers',
            name='P95 TTFT',
            line=dict(color='darkred', width=2, dash='dot'),
            marker=dict(size=6),
            hovertemplate='Concurrency: %{x}<br>P95 TTFT: %{y:.3f}s<extra></extra>'
        ),
        row=3, col=1
    )

    # Add horizontal line for TTFT threshold
    fig.add_hline(
        y=max_ttft,
        line=dict(color='red', width=2, dash='dash'),
        annotation_text=f"TTFT Threshold ({max_ttft}s)",
        annotation_position="right",
        row=3, col=1
    )

    # Add vertical line for peak concurrency (across all subplots)
    for row in [1, 2, 3]:
        fig.add_vline(
            x=peak_concurrency,
            line=dict(color='purple', width=3, dash='dash'),
            annotation_text=f"Peak Concurrency ({peak_concurrency})" if row == 1 else "",
            annotation_position="top",
            row=row, col=1
        )

    # Update axes
    fig.update_xaxes(title_text="Concurrency Level", row=1, col=1)
    fig.update_xaxes(title_text="Concurrency Level", row=2, col=1)
    fig.update_xaxes(title_text="Concurrency Level", row=3, col=1)

    fig.update_yaxes(title_text="Input Tokens/s", row=1, col=1)
    fig.update_yaxes(title_text="Output Tokens/s", row=2, col=1)
    fig.update_yaxes(title_text="TTFT (seconds)", row=3, col=1)

    # Update layout
    fig.update_layout(
        title=f"Concurrency Ramp Analysis<br>Context: {context_size:,} tokens | Cache Hit Rate: {cache_hit_rate}%",
        height=900,
        showlegend=True,
        hovermode='x unified'
    )

    # Save with naming scheme: ramp_ctx{context_size}_cache{cache_hit_rate}.html
    filename = output_path / f"ramp_ctx{context_size}_cache{cache_hit_rate}.html"
    fig.write_html(filename)
    logger.debug(f"Generated ramp graph: {filename}")


def generate_graphs(metrics: List[AggregatedMetrics], output_dir: str):
    """Generate Plotly visualizations"""
    if not metrics:
        logger.warning("No metrics to visualize")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([m.to_dict() for m in metrics])

    # Load phase metadata to calculate variability using exact phase boundaries
    phase_metadata_by_test = load_phase_metadata(output_dir)

    # Load detailed results for use with phase metadata
    detailed_dfs = {}
    for context_size in df['context_size'].unique():
        csv_files = list(output_path.glob(f"detailed_results_{context_size}_*.csv"))
        if csv_files:
            # Get most recent file
            most_recent = max(csv_files, key=lambda p: p.stat().st_mtime)
            try:
                detailed_dfs[context_size] = pd.read_csv(most_recent)
                logger.debug(f"Loaded detailed results for context {context_size} from {most_recent.name}")
            except Exception as e:
                logger.warning(f"Could not load detailed results for context {context_size}: {e}")

    # Calculate throughput variability using phase metadata
    # Store as dict keyed by (context_size, cache_rate) for use across all graphs
    all_detailed_variability = {}

    for context_size in df['context_size'].unique():
        df_ctx = df[df['context_size'] == context_size].sort_values('cache_hit_rate')

        if context_size not in detailed_dfs:
            continue

        detailed_df = detailed_dfs[context_size]

        for cache_rate in df_ctx['cache_hit_rate'].unique():
            test_key = (context_size, cache_rate)

            # Get phase metadata for this test
            if test_key not in phase_metadata_by_test:
                logger.debug(f"No phase metadata found for context={context_size}, cache_rate={cache_rate}")
                continue

            phases = phase_metadata_by_test[test_key]
            peak_conc = df_ctx[df_ctx['cache_hit_rate'] == cache_rate]['peak_concurrency'].iloc[0]

            # Filter to peak concurrency phases (RAMP at peak + all RETRY phases)
            # This matches what's shown in the Retry Summary console output
            retry_phases = [p for p in phases
                          if p.concurrency_level == peak_conc and
                             (p.phase_type == "RETRY" or
                              (p.phase_type == "RAMP" and p.concurrency_level == peak_conc))]

            if len(retry_phases) == 0:
                logger.debug(f"No peak concurrency phases for context={context_size}, cache_rate={cache_rate}")
                continue

            retry_throughputs = {'input': [], 'output': []}
            retry_ttft_stats = {'avg': [], 'median': [], 'p95': [], 'p99': []}

            for phase in retry_phases:
                # Get requests that completed within this phase using phase_id
                phase_requests = detailed_df[
                    (detailed_df['phase_id'] == phase.phase_id) &
                    (detailed_df['cache_hit_rate'] == cache_rate) &
                    (detailed_df['concurrency_level'] == peak_conc)
                ]

                if len(phase_requests) > 0:
                    # Use phase duration from metadata (more accurate than calculating from timestamps)
                    duration = phase.duration

                    total_input = (phase_requests['cached_tokens'] + phase_requests['unique_tokens']).sum()
                    total_output = phase_requests['output_tokens'].sum()

                    retry_throughputs['input'].append(total_input / duration)
                    retry_throughputs['output'].append(total_output / duration)

                    # Calculate TTFT stats per-phase, then we'll average them
                    phase_ttfts = phase_requests['ttft'].values
                    if len(phase_ttfts) > 0:
                        retry_ttft_stats['avg'].append(np.mean(phase_ttfts))
                        retry_ttft_stats['median'].append(np.median(phase_ttfts))
                        retry_ttft_stats['p95'].append(np.percentile(phase_ttfts, 95))
                        retry_ttft_stats['p99'].append(np.percentile(phase_ttfts, 99))

            if len(retry_throughputs['input']) > 0:
                all_detailed_variability[(context_size, cache_rate)] = {
                    'input_min': np.min(retry_throughputs['input']),
                    'input_max': np.max(retry_throughputs['input']),
                    'input_mean': np.mean(retry_throughputs['input']),
                    'input_std': np.std(retry_throughputs['input']),
                    'output_min': np.min(retry_throughputs['output']),
                    'output_max': np.max(retry_throughputs['output']),
                    'output_mean': np.mean(retry_throughputs['output']),
                    'output_std': np.std(retry_throughputs['output']),
                    # Average the per-phase TTFT statistics (matching how we handle throughput)
                    'avg_ttft': np.mean(retry_ttft_stats['avg']) if len(retry_ttft_stats['avg']) > 0 else 0,
                    'median_ttft': np.mean(retry_ttft_stats['median']) if len(retry_ttft_stats['median']) > 0 else 0,
                    'p95_ttft': np.mean(retry_ttft_stats['p95']) if len(retry_ttft_stats['p95']) > 0 else 0,
                    'p99_ttft': np.mean(retry_ttft_stats['p99']) if len(retry_ttft_stats['p99']) > 0 else 0,
                }

    # Graph 1: Cache hit rate vs performance metrics (per context size) - now with 3 separate charts
    for context_size in df['context_size'].unique():
        df_ctx = df[df['context_size'] == context_size].sort_values('cache_hit_rate')

        # Extract detailed variability for this context size from pre-calculated dict
        detailed_variability = {}
        for cache_rate in df_ctx['cache_hit_rate'].unique():
            if (context_size, cache_rate) in all_detailed_variability:
                detailed_variability[cache_rate] = all_detailed_variability[(context_size, cache_rate)]

        # Create subplot figure with 3 rows (like ramp graph)
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                'Input Throughput vs Cache Hit Rate',
                'Output Throughput vs Cache Hit Rate',
                'TTFT Metrics vs Cache Hit Rate'
            ),
            vertical_spacing=0.12,
            specs=[[{"secondary_y": False}],
                   [{"secondary_y": False}],
                   [{"secondary_y": False}]]
        )

        # Row 1: Input tokens/s with variability bands
        # Add min/max shaded area if we have variability data
        if detailed_variability:
            cache_rates_with_var = sorted([k for k in detailed_variability.keys()])
            input_min = [detailed_variability[cr]['input_min'] for cr in cache_rates_with_var]
            input_max = [detailed_variability[cr]['input_max'] for cr in cache_rates_with_var]

            # Add upper bound
            fig.add_trace(
                go.Scatter(
                    x=cache_rates_with_var,
                    y=input_max,
                    mode='lines',
                    name='Input max',
                    line=dict(color='rgba(0, 0, 255, 0)', width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=1, col=1
            )

            # Add lower bound with fill
            fig.add_trace(
                go.Scatter(
                    x=cache_rates_with_var,
                    y=input_min,
                    mode='lines',
                    name='Input range',
                    line=dict(color='rgba(0, 0, 255, 0)', width=0),
                    fill='tonexty',
                    fillcolor='rgba(0, 0, 255, 0.15)',
                    hovertemplate='Cache Rate: %{x}%<br>Min-Max Range<extra></extra>'
                ),
                row=1, col=1
            )

        # Add main input throughput line
        # Use mean from retry runs (if available) instead of aggregated mean from all concurrency levels
        if detailed_variability:
            # Use the retry-based mean for cache rates that have variability data
            input_mean_values = []
            cache_rates_all = sorted(df_ctx['cache_hit_rate'].unique())
            for cr in cache_rates_all:
                if cr in detailed_variability:
                    input_mean_values.append(detailed_variability[cr]['input_mean'])
                else:
                    # Fallback to aggregated value if no detailed data
                    input_mean_values.append(df_ctx[df_ctx['cache_hit_rate'] == cr]['input_tokens_per_sec'].iloc[0])

            fig.add_trace(
                go.Scatter(
                    x=cache_rates_all,
                    y=input_mean_values,
                    mode='lines+markers',
                    name='Input tokens/s (avg)',
                    line=dict(color='blue', width=3),
                    marker=dict(size=8),
                    hovertemplate='Cache Rate: %{x}%<br>Input: %{y:,.0f} tok/s<extra></extra>'
                ),
                row=1, col=1
            )
        else:
            # No variability data, use aggregated metrics
            fig.add_trace(
                go.Scatter(
                    x=df_ctx['cache_hit_rate'],
                    y=df_ctx['input_tokens_per_sec'],
                    mode='lines+markers',
                    name='Input tokens/s (avg)',
                    line=dict(color='blue', width=3),
                    marker=dict(size=8),
                    hovertemplate='Cache Rate: %{x}%<br>Input: %{y:,.0f} tok/s<extra></extra>'
                ),
                row=1, col=1
            )

        # Row 2: Output tokens/s with variability bands
        # Add min/max shaded area if we have variability data
        if detailed_variability:
            output_min = [detailed_variability[cr]['output_min'] for cr in cache_rates_with_var]
            output_max = [detailed_variability[cr]['output_max'] for cr in cache_rates_with_var]

            # Add upper bound
            fig.add_trace(
                go.Scatter(
                    x=cache_rates_with_var,
                    y=output_max,
                    mode='lines',
                    name='Output max',
                    line=dict(color='rgba(0, 255, 0, 0)', width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=2, col=1
            )

            # Add lower bound with fill
            fig.add_trace(
                go.Scatter(
                    x=cache_rates_with_var,
                    y=output_min,
                    mode='lines',
                    name='Output range',
                    line=dict(color='rgba(0, 255, 0, 0)', width=0),
                    fill='tonexty',
                    fillcolor='rgba(0, 255, 0, 0.15)',
                    hovertemplate='Cache Rate: %{x}%<br>Min-Max Range<extra></extra>'
                ),
                row=2, col=1
            )

        # Add main output throughput line
        # Use mean from retry runs (if available) instead of aggregated mean from all concurrency levels
        if detailed_variability:
            # Use the retry-based mean for cache rates that have variability data
            output_mean_values = []
            cache_rates_all = sorted(df_ctx['cache_hit_rate'].unique())
            for cr in cache_rates_all:
                if cr in detailed_variability:
                    output_mean_values.append(detailed_variability[cr]['output_mean'])
                else:
                    # Fallback to aggregated value if no detailed data
                    output_mean_values.append(df_ctx[df_ctx['cache_hit_rate'] == cr]['output_tokens_per_sec'].iloc[0])

            fig.add_trace(
                go.Scatter(
                    x=cache_rates_all,
                    y=output_mean_values,
                    mode='lines+markers',
                    name='Output tokens/s (avg)',
                    line=dict(color='green', width=3),
                    marker=dict(size=8),
                    hovertemplate='Cache Rate: %{x}%<br>Output: %{y:,.0f} tok/s<extra></extra>'
                ),
                row=2, col=1
            )
        else:
            # No variability data, use aggregated metrics
            fig.add_trace(
                go.Scatter(
                    x=df_ctx['cache_hit_rate'],
                    y=df_ctx['output_tokens_per_sec'],
                    mode='lines+markers',
                    name='Output tokens/s (avg)',
                    line=dict(color='green', width=3),
                    marker=dict(size=8),
                    hovertemplate='Cache Rate: %{x}%<br>Output: %{y:,.0f} tok/s<extra></extra>'
                ),
                row=2, col=1
            )

        # Row 3: TTFT metrics (avg, median, p95, p99)
        # Use retry-based TTFT values if available, otherwise fall back to aggregated metrics
        if detailed_variability:
            # Use the retry-based TTFT for cache rates that have variability data
            cache_rates_all = sorted(df_ctx['cache_hit_rate'].unique())
            avg_ttft_values = []
            median_ttft_values = []
            p95_ttft_values = []
            p99_ttft_values = []

            for cr in cache_rates_all:
                if cr in detailed_variability:
                    avg_ttft_values.append(detailed_variability[cr]['avg_ttft'])
                    median_ttft_values.append(detailed_variability[cr]['median_ttft'])
                    p95_ttft_values.append(detailed_variability[cr]['p95_ttft'])
                    p99_ttft_values.append(detailed_variability[cr]['p99_ttft'])
                else:
                    # Fallback to aggregated values if no detailed data
                    avg_ttft_values.append(df_ctx[df_ctx['cache_hit_rate'] == cr]['avg_ttft'].iloc[0])
                    median_ttft_values.append(df_ctx[df_ctx['cache_hit_rate'] == cr]['median_ttft'].iloc[0])
                    p95_ttft_values.append(df_ctx[df_ctx['cache_hit_rate'] == cr]['p95_ttft'].iloc[0])
                    p99_ttft_values.append(df_ctx[df_ctx['cache_hit_rate'] == cr]['p99_ttft'].iloc[0])

            fig.add_trace(
                go.Scatter(
                    x=cache_rates_all,
                    y=avg_ttft_values,
                    mode='lines+markers',
                    name='Avg TTFT',
                    line=dict(color='orange', width=2),
                    marker=dict(size=6),
                    hovertemplate='Cache Rate: %{x}%<br>Avg TTFT: %{y:.3f}s<extra></extra>'
                ),
                row=3, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=cache_rates_all,
                    y=median_ttft_values,
                    mode='lines+markers',
                    name='Median TTFT',
                    line=dict(color='darkorange', width=2, dash='dot'),
                    marker=dict(size=6),
                    hovertemplate='Cache Rate: %{x}%<br>Median TTFT: %{y:.3f}s<extra></extra>'
                ),
                row=3, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=cache_rates_all,
                    y=p95_ttft_values,
                    mode='lines+markers',
                    name='P95 TTFT',
                    line=dict(color='red', width=2, dash='dot'),
                    marker=dict(size=6),
                    hovertemplate='Cache Rate: %{x}%<br>P95 TTFT: %{y:.3f}s<extra></extra>'
                ),
                row=3, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=cache_rates_all,
                    y=p99_ttft_values,
                    mode='lines+markers',
                    name='P99 TTFT',
                    line=dict(color='darkred', width=2, dash='dash'),
                    marker=dict(size=6),
                    hovertemplate='Cache Rate: %{x}%<br>P99 TTFT: %{y:.3f}s<extra></extra>'
                ),
                row=3, col=1
            )
        else:
            # No variability data, use aggregated metrics
            fig.add_trace(
                go.Scatter(
                    x=df_ctx['cache_hit_rate'],
                    y=df_ctx['avg_ttft'],
                    mode='lines+markers',
                    name='Avg TTFT',
                    line=dict(color='orange', width=2),
                    marker=dict(size=6),
                    hovertemplate='Cache Rate: %{x}%<br>Avg TTFT: %{y:.3f}s<extra></extra>'
                ),
                row=3, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=df_ctx['cache_hit_rate'],
                    y=df_ctx['median_ttft'],
                    mode='lines+markers',
                    name='Median TTFT',
                    line=dict(color='darkorange', width=2, dash='dot'),
                    marker=dict(size=6),
                    hovertemplate='Cache Rate: %{x}%<br>Median TTFT: %{y:.3f}s<extra></extra>'
                ),
                row=3, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=df_ctx['cache_hit_rate'],
                    y=df_ctx['p95_ttft'],
                    mode='lines+markers',
                    name='P95 TTFT',
                    line=dict(color='red', width=2, dash='dot'),
                    marker=dict(size=6),
                    hovertemplate='Cache Rate: %{x}%<br>P95 TTFT: %{y:.3f}s<extra></extra>'
                ),
                row=3, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=df_ctx['cache_hit_rate'],
                    y=df_ctx['p99_ttft'],
                    mode='lines+markers',
                    name='P99 TTFT',
                    line=dict(color='darkred', width=2, dash='dash'),
                    marker=dict(size=6),
                    hovertemplate='Cache Rate: %{x}%<br>P99 TTFT: %{y:.3f}s<extra></extra>'
                ),
                row=3, col=1
            )

        # Update axes
        fig.update_xaxes(
            title_text="Cache Hit Rate (%)",
            row=1, col=1,
            range=[0, 100],
            dtick=10,
            gridcolor='lightgray',
            showgrid=True
        )
        fig.update_xaxes(
            title_text="Cache Hit Rate (%)",
            row=2, col=1,
            range=[0, 100],
            dtick=10,
            gridcolor='lightgray',
            showgrid=True
        )
        fig.update_xaxes(
            title_text="Cache Hit Rate (%)",
            row=3, col=1,
            range=[0, 100],
            dtick=10,
            gridcolor='lightgray',
            showgrid=True
        )

        fig.update_yaxes(title_text="Input Tokens/s", row=1, col=1)
        fig.update_yaxes(title_text="Output Tokens/s", row=2, col=1)
        fig.update_yaxes(title_text="TTFT (seconds)", row=3, col=1)

        # Update layout
        fig.update_layout(
            title=f"Performance vs Cache Hit Rate<br>Context: {context_size:,} tokens",
            height=900,
            showlegend=True,
            hovermode='x unified'
        )

        filename = output_path / f"performance_vs_cache_rate_{context_size}.html"
        fig.write_html(filename)
        logger.info(f"Generated graph: {filename}")

    # Graph 2: Context length comparison (output tokens/s)
    # Use same per-retry calculation as performance vs cache rate graph
    # Generate even for single context size (user preference)
    if len(df['context_size'].unique()) >= 1:
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Define colors for each context size (will be reused for both solid and dashed lines)
        colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']

        for idx, context_size in enumerate(sorted(df['context_size'].unique())):
            df_ctx = df[df['context_size'] == context_size].sort_values('cache_hit_rate')
            color = colors[idx % len(colors)]

            # Use per-retry mean if available, otherwise fall back to aggregated
            y_values = []
            for _, row in df_ctx.iterrows():
                cache_rate = row['cache_hit_rate']
                if (context_size, cache_rate) in all_detailed_variability:
                    # Use per-retry mean (same as performance graph)
                    y_values.append(all_detailed_variability[(context_size, cache_rate)]['output_mean'])
                else:
                    # Fallback to aggregated value
                    y_values.append(row['output_tokens_per_sec'])

            # Primary y-axis: Absolute values (solid line)
            fig.add_trace(
                go.Scatter(
                    x=df_ctx['cache_hit_rate'],
                    y=y_values,
                    mode='lines+markers',
                    name=f'{context_size:,} tokens',
                    line=dict(width=2, color=color),
                    marker=dict(size=6),
                    hovertemplate='%{fullData.name}<br>Cache Rate: %{x}%<br>Output: %{y:,.0f} tok/s<extra></extra>'
                ),
                secondary_y=False
            )

            # Secondary y-axis: Relative to baseline (dashed line, same color)
            baseline = y_values[0] if y_values else 1  # First value (0% cache)
            relative_values = [y / baseline for y in y_values]

            fig.add_trace(
                go.Scatter(
                    x=df_ctx['cache_hit_rate'],
                    y=relative_values,
                    mode='lines+markers',
                    name=f'{context_size:,} tokens (speedup)',
                    line=dict(width=2, dash='dash', color=color),
                    marker=dict(size=6, symbol='diamond'),
                    hovertemplate='%{fullData.name}<br>Cache Rate: %{x}%<br>Speedup: %{y:.2f}x<extra></extra>'
                ),
                secondary_y=True
            )

        # Update axes
        fig.update_xaxes(
            title_text="Cache Hit Rate (%)",
            range=[0, 100],
            dtick=10,
            gridcolor='lightgray',
            showgrid=True
        )
        fig.update_yaxes(title_text="Output Tokens/s", secondary_y=False)
        fig.update_yaxes(title_text="Speedup (relative to 0% cache)", secondary_y=True)

        fig.update_layout(
            title="Output Throughput Comparison Across Context Lengths",
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

        filename = output_path / "output_throughput_comparison.html"
        fig.write_html(filename)
        logger.info(f"Generated graph: {filename}")

    # Graph 3: Context length comparison (input tokens/s)
    # Use same per-retry calculation as performance vs cache rate graph
    # Generate even for single context size (user preference)
    if len(df['context_size'].unique()) >= 1:
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Define colors for each context size (will be reused for both solid and dashed lines)
        colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']

        for idx, context_size in enumerate(sorted(df['context_size'].unique())):
            df_ctx = df[df['context_size'] == context_size].sort_values('cache_hit_rate')
            color = colors[idx % len(colors)]

            # Use per-retry mean if available, otherwise fall back to aggregated
            y_values = []
            for _, row in df_ctx.iterrows():
                cache_rate = row['cache_hit_rate']
                if (context_size, cache_rate) in all_detailed_variability:
                    # Use per-retry mean (same as performance graph)
                    y_values.append(all_detailed_variability[(context_size, cache_rate)]['input_mean'])
                else:
                    # Fallback to aggregated value
                    y_values.append(row['input_tokens_per_sec'])

            # Primary y-axis: Absolute values (solid line)
            fig.add_trace(
                go.Scatter(
                    x=df_ctx['cache_hit_rate'],
                    y=y_values,
                    mode='lines+markers',
                    name=f'{context_size:,} tokens',
                    line=dict(width=2, color=color),
                    marker=dict(size=6),
                    hovertemplate='%{fullData.name}<br>Cache Rate: %{x}%<br>Input: %{y:,.0f} tok/s<extra></extra>'
                ),
                secondary_y=False
            )

            # Secondary y-axis: Relative to baseline (dashed line, same color)
            baseline = y_values[0] if y_values else 1  # First value (0% cache)
            relative_values = [y / baseline for y in y_values]

            fig.add_trace(
                go.Scatter(
                    x=df_ctx['cache_hit_rate'],
                    y=relative_values,
                    mode='lines+markers',
                    name=f'{context_size:,} tokens (speedup)',
                    line=dict(width=2, dash='dash', color=color),
                    marker=dict(size=6, symbol='diamond'),
                    hovertemplate='%{fullData.name}<br>Cache Rate: %{x}%<br>Speedup: %{y:.2f}x<extra></extra>'
                ),
                secondary_y=True
            )

        # Update axes
        fig.update_xaxes(
            title_text="Cache Hit Rate (%)",
            range=[0, 100],
            dtick=10,
            gridcolor='lightgray',
            showgrid=True
        )
        fig.update_yaxes(title_text="Input Tokens/s", secondary_y=False)
        fig.update_yaxes(title_text="Speedup (relative to 0% cache)", secondary_y=True)

        fig.update_layout(
            title="Input Throughput Comparison Across Context Lengths",
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

        filename = output_path / "input_throughput_comparison.html"
        fig.write_html(filename)
        logger.info(f"Generated graph: {filename}")

    # Graph 4: Output token metrics comparison (ITL and avg output tokens/s per request)
    # Generate even for single context size (user preference)
    if len(df['context_size'].unique()) >= 1:
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(
                'Average Inter-Token Latency (ITL) Across Context Lengths',
                'Average Output Tokens/s per Request Across Context Lengths'
            ),
            vertical_spacing=0.15,
            specs=[[{"secondary_y": False}],
                   [{"secondary_y": False}]]
        )

        # Row 1: Inter-Token Latency (ms)
        for context_size in sorted(df['context_size'].unique()):
            df_ctx = df[df['context_size'] == context_size].sort_values('cache_hit_rate')
            fig.add_trace(
                go.Scatter(
                    x=df_ctx['cache_hit_rate'],
                    y=df_ctx['avg_itl'] * 1000,  # Convert to ms
                    mode='lines+markers',
                    name=f'{context_size:,} tokens',
                    line=dict(width=2),
                    hovertemplate='Cache Rate: %{x}%<br>ITL: %{y:.2f}ms<extra></extra>'
                ),
                row=1, col=1
            )

        # Row 2: Average output tokens/s per request
        for context_size in sorted(df['context_size'].unique()):
            df_ctx = df[df['context_size'] == context_size].sort_values('cache_hit_rate')
            fig.add_trace(
                go.Scatter(
                    x=df_ctx['cache_hit_rate'],
                    y=df_ctx['avg_output_tokens'],
                    mode='lines+markers',
                    name=f'{context_size:,} tokens',
                    line=dict(width=2),
                    showlegend=False,  # Already shown in row 1
                    hovertemplate='Cache Rate: %{x}%<br>Output: %{y:.1f} tok/s<extra></extra>'
                ),
                row=2, col=1
            )

        # Update axes
        fig.update_xaxes(
            title_text="Cache Hit Rate (%)",
            row=1, col=1,
            range=[0, 100],
            dtick=10,
            gridcolor='lightgray',
            showgrid=True
        )
        fig.update_xaxes(
            title_text="Cache Hit Rate (%)",
            row=2, col=1,
            range=[0, 100],
            dtick=10,
            gridcolor='lightgray',
            showgrid=True
        )
        fig.update_yaxes(title_text="Inter-Token Latency (ms)", row=1, col=1)
        fig.update_yaxes(title_text="Output Tokens/s per Request", row=2, col=1)

        # Update layout
        fig.update_layout(
            title="Output Token Metrics Comparison Across Context Lengths",
            height=800,
            hovermode='x unified',
            showlegend=True
        )

        filename = output_path / "output_metrics_comparison.html"
        fig.write_html(filename)
        logger.info(f"Generated graph: {filename}")

    # Graph 5: TTFT heatmap
    if len(df['context_size'].unique()) > 1:
        pivot = df.pivot(index='context_size', columns='cache_hit_rate', values='avg_ttft')

        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale='RdYlGn_r',
            text=pivot.values,
            texttemplate='%{text:.3f}s',
            textfont={"size": 10},
            colorbar=dict(title="TTFT (s)")
        ))

        fig.update_layout(
            title="Average TTFT Heatmap",
            xaxis=dict(
                title="Cache Hit Rate (%)",
                range=[-2.5, 102.5],  # Slightly wider than 0-100 for visual padding
                dtick=10,
                gridcolor='lightgray',
                showgrid=True
            ),
            yaxis_title="Context Size (tokens)",
            height=600
        )

        filename = output_path / "ttft_heatmap.html"
        fig.write_html(filename)
        logger.info(f"Generated graph: {filename}")


async def main():
    """Main entry point"""
    args = parse_arguments()

    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Create test configuration
    config = create_test_config(args)

    logger.info(f"{Colors.HEADER}{'='*80}{Colors.ENDC}")
    logger.info(f"{Colors.HEADER}{Colors.BOLD}Cache Rate Performance Testing Tool{Colors.ENDC}")
    logger.info(f"{Colors.HEADER}{'='*80}{Colors.ENDC}")
    logger.info(f"{Colors.OKBLUE}API Endpoint: {config.api_endpoint}{Colors.ENDC}")
    logger.info(f"{Colors.OKBLUE}Context Sizes: {config.context_sizes}{Colors.ENDC}")
    logger.info(f"{Colors.OKBLUE}Working Set Size: {config.working_set_size:,} tokens{Colors.ENDC}")
    logger.info(f"{Colors.OKBLUE}Output Tokens: {config.output_tokens}{Colors.ENDC}")
    logger.info(f"{Colors.OKBLUE}Max TTFT: {config.max_ttft}s{Colors.ENDC}")
    logger.info(f"{Colors.OKBLUE}Test Duration: {config.test_duration}s (ramp: {config.ramp_duration}s per level){Colors.ENDC}")
    logger.info(f"{Colors.OKBLUE}Cache Hit Rates: {len(config.cache_hit_rates)} tests ({min(config.cache_hit_rates)}% to {max(config.cache_hit_rates)}%){Colors.ENDC}")
    logger.info(f"{Colors.OKBLUE}Chunk Size: {config.chunk_size} tokens (for cache block alignment){Colors.ENDC}")
    logger.info(f"{Colors.OKBLUE}Output Directory: {config.output_dir}{Colors.ENDC}")
    logger.info(f"{Colors.HEADER}{'='*80}{Colors.ENDC}")

    # Create output directory
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save the command for easy re-running
    save_run_command(args, config.output_dir)

    # Initialize components
    logger.info("Initializing components...")

    # Initialize API client and detect model first
    api_client = APIClient(config.api_endpoint, model="")
    model = await api_client.detect_model()
    api_client.model = model

    # Validate working set size against KV cache capacity
    logger.info("")
    validate_working_set_size(model, config.working_set_size)
    logger.info("")

    # Initialize tokenizer - use detected model if --tokenizer not explicitly set
    tokenizer_id = config.tokenizer_id
    if args.tokenizer == "Qwen/Qwen2.5-Coder-32B-Instruct":  # Default value
        # User didn't specify tokenizer, try using detected model
        try:
            logger.info(f"Attempting to use detected model tokenizer: {model}")
            tokenizer = TokenizerManager(model)
        except Exception as e:
            logger.warning(f"Could not load tokenizer for detected model '{model}': {e}")
            logger.info(f"Falling back to default tokenizer: {tokenizer_id}")
            tokenizer = TokenizerManager(tokenizer_id)
    else:
        # User explicitly specified tokenizer
        tokenizer = TokenizerManager(tokenizer_id)

    # Initialize progress tracker
    progress = ProgressTracker(config.output_dir, config)

    # Calculate and display KV cache requirements
    logger.info("")
    logger.info(f"{Colors.PHASE}{'='*80}{Colors.ENDC}")
    logger.info(f"{Colors.PHASE}{Colors.BOLD}KV Cache Memory Requirements (Per Request){Colors.ENDC}")
    logger.info(f"{Colors.PHASE}{'='*80}{Colors.ENDC}")
    dtype_display = "FP8 (1 byte)" if config.kv_cache_bytes == 1 else "BF16/FP16 (2 bytes)"
    logger.info(f"{Colors.WARNING}Note: Using {dtype_display} for estimation.{Colors.ENDC}")
    logger.info(f"{Colors.WARNING}      Actual KV cache size depends on server configuration.{Colors.ENDC}")
    logger.info("")

    # Calculate KV cache for each context size
    total_working_set_cache_mb = 0
    for context_size in config.context_sizes:
        kv_info = calculate_kv_cache_size(context_size, config.kv_cache_bytes)
        logger.info(f"{Colors.OKBLUE}Context {context_size:>7,} tokens: {kv_info['total_mb']:>8.2f} MB ({kv_info['total_gb']:.3f} GB){Colors.ENDC}")

        # Calculate working set cache size for this context
        # Working set has multiple prompts, each needs KV cache
        num_prompts = max(1, int(np.ceil(config.working_set_size / context_size)))
        working_set_cache_mb = kv_info['total_mb'] * num_prompts
        total_working_set_cache_mb += working_set_cache_mb

        logger.info(f"{Colors.DEBUG}  → Working set: {num_prompts} prompts × {kv_info['total_mb']:.2f} MB = {working_set_cache_mb:.2f} MB ({working_set_cache_mb/1024:.3f} GB){Colors.ENDC}")

    logger.info("")
    logger.info(f"{Colors.METRIC}Total Working Set Cache (all contexts): {total_working_set_cache_mb:.2f} MB ({total_working_set_cache_mb/1024:.3f} GB){Colors.ENDC}")
    logger.info(f"{Colors.WARNING}⚠  Ensure your caching engine has at least {total_working_set_cache_mb/1024:.2f} GB available for working set.{Colors.ENDC}")
    logger.info(f"{Colors.PHASE}{'='*80}{Colors.ENDC}")
    logger.info("")

    logger.info("Initialization complete. Ready to start testing.")
    logger.info("")

    # Load existing aggregated results if resuming
    all_aggregated_results = load_existing_aggregated_results(config.output_dir)
    if all_aggregated_results:
        logger.info(f"Loaded {len(all_aggregated_results)} existing test results from previous run")

    # Main test loop - iterate through context sizes
    all_detailed_results = []

    for context_size in config.context_sizes:
        logger.info(f"{Colors.PHASE}{'='*80}{Colors.ENDC}")
        logger.info(f"{Colors.PHASE}{Colors.BOLD}Testing context size: {context_size:,} tokens{Colors.ENDC}")
        logger.info(f"{Colors.PHASE}{'='*80}{Colors.ENDC}")

        # Check if all tests for this context are already completed
        remaining_tests = [rate for rate in config.cache_hit_rates
                          if not progress.is_test_completed(context_size, rate)]

        if not remaining_tests:
            logger.info(f"  ✓ All tests for context {context_size:,} already completed, skipping initialization")
            logger.info("")
            continue

        logger.info(f"  Found {len(remaining_tests)} remaining tests: {remaining_tests}")

        # Initialize working set for this context size
        working_set = WorkingSet(context_size, config.working_set_size, tokenizer, config.chunk_size, config.seed)
        working_set.generate_prompts()

        # Initialize working set with API (pre-warm cache)
        if config.reinit_strategy == "once" or config.reinit_strategy == "per_cache_rate":
            await initialize_working_set(api_client, working_set, config.output_tokens, config.init_concurrency)

        # Track previous peak concurrency for adaptive starting points
        previous_peak_concurrency = None

        # Test each cache hit rate
        for cache_hit_rate in config.cache_hit_rates:
            # Check if already completed
            if progress.is_test_completed(context_size, cache_hit_rate):
                logger.info(f"  Skipping cache hit rate {cache_hit_rate}% (already completed)")
                continue

            logger.info("")
            logger.info(f"{Colors.OKCYAN}  Testing cache hit rate: {cache_hit_rate}%{Colors.ENDC}")
            logger.info(f"{Colors.OKCYAN}  {'-'*60}{Colors.ENDC}")

            # Reinitialize if strategy requires
            if config.reinit_strategy == "per_cache_rate":
                await initialize_working_set(api_client, working_set, config.output_tokens, config.init_concurrency)

            # Run the cache hit rate test
            try:
                detailed_metrics, aggregated, phase_metadata_list = await run_cache_hit_rate_test(
                    config=config,
                    api_client=api_client,
                    working_set=working_set,
                    tokenizer=tokenizer,
                    context_size=context_size,
                    cache_hit_rate=cache_hit_rate,
                    previous_peak=previous_peak_concurrency
                )

                # Save results
                all_detailed_results.extend(detailed_metrics)
                all_aggregated_results.append(aggregated)

                # Update previous peak for next cache hit rate
                previous_peak_concurrency = aggregated.peak_concurrency

                # Mark as completed
                progress.mark_test_completed(context_size, cache_hit_rate)

                # Write incremental results
                save_detailed_results(all_detailed_results, config.output_dir, context_size)
                save_aggregated_results(all_aggregated_results, config.output_dir)
                save_phase_metadata(phase_metadata_list, config.output_dir)

                # Generate ramp graph for this cache hit rate
                if not config.skip_graphs:
                    generate_ramp_graph(detailed_metrics, context_size, cache_hit_rate,
                                       config.max_ttft, aggregated.peak_concurrency, config.output_dir)

                logger.info(f"{Colors.SUCCESS}  ✓ Cache hit rate {cache_hit_rate}% complete{Colors.ENDC}")
                logger.info(f"{Colors.METRIC}    Peak concurrency: {aggregated.peak_concurrency}{Colors.ENDC}")
                logger.info(f"{Colors.METRIC}    Avg TTFT: {aggregated.avg_ttft:.3f}s{Colors.ENDC}")
                logger.info(f"{Colors.METRIC}    Avg TTLT: {aggregated.avg_ttlt:.3f}s{Colors.ENDC}")
                logger.info(f"{Colors.METRIC}    Avg ITL: {aggregated.avg_itl*1000:.2f}ms{Colors.ENDC}")
                logger.info(f"{Colors.METRIC}    Avg output tok/s per req: {aggregated.avg_output_tokens:.1f}{Colors.ENDC}")
                logger.info(f"{Colors.METRIC}    Input: {aggregated.input_tokens_per_sec:.1f} tok/s{Colors.ENDC}")
                logger.info(f"{Colors.METRIC}    Output: {aggregated.output_tokens_per_sec:.1f} tok/s{Colors.ENDC}")

            except Exception as e:
                logger.error(f"  ✗ Cache hit rate {cache_hit_rate}% failed: {e}")
                continue

        logger.info("")
        logger.info(f"{Colors.SUCCESS}✓ Context size {context_size:,} complete{Colors.ENDC}")

        # Generate graphs after each context size completes
        if not config.skip_graphs and all_aggregated_results:
            logger.info("")
            logger.info(f"{Colors.PHASE}Generating visualizations for completed tests...{Colors.ENDC}")
            generate_graphs(all_aggregated_results, config.output_dir)

            # Generate index.html dashboard
            logger.info(f"{Colors.OKBLUE}Updating index.html dashboard...{Colors.ENDC}")
            try:
                import subprocess
                subprocess.run([
                    sys.executable, "generate_index.py",
                    config.output_dir, __version__
                ], check=True, capture_output=True, text=True)
                logger.info(f"{Colors.SUCCESS}✓ Updated index.html dashboard{Colors.ENDC}")
            except Exception as e:
                logger.warning(f"Failed to generate index.html: {e}")

    # Generate final graphs if not skipped
    if not config.skip_graphs and all_aggregated_results:
        logger.info("")
        logger.info(f"{Colors.PHASE}{'='*80}{Colors.ENDC}")
        logger.info(f"{Colors.PHASE}Generating visualizations...{Colors.ENDC}")
        logger.info(f"{Colors.PHASE}{'='*80}{Colors.ENDC}")
        generate_graphs(all_aggregated_results, config.output_dir)

        # Generate index.html dashboard
        logger.info(f"{Colors.OKBLUE}Generating index.html dashboard...{Colors.ENDC}")
        try:
            import subprocess
            subprocess.run([
                sys.executable, "generate_index.py",
                config.output_dir, __version__
            ], check=True, capture_output=True, text=True)
            logger.info(f"{Colors.SUCCESS}✓ Generated index.html dashboard{Colors.ENDC}")
        except Exception as e:
            logger.warning(f"Failed to generate index.html: {e}")

    logger.info("")
    logger.info(f"{Colors.SUCCESS}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    logger.info(f"{Colors.SUCCESS}{Colors.BOLD}✓ All tests complete!{Colors.ENDC}")
    logger.info(f"{Colors.SUCCESS}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    logger.info(f"{Colors.OKGREEN}Results saved to: {config.output_dir}{Colors.ENDC}")
    logger.info(f"{Colors.OKGREEN}Total tests completed: {len(all_aggregated_results)}{Colors.ENDC}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user. Progress has been saved.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
