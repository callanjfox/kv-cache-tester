#!/usr/bin/env python3
"""
Working Set Size Performance Testing Tool

Measures input tokens/s, time to first token (TTFT), and output tokens/s
at varying working set sizes to understand memory tier performance (HBM/DRAM/SSD).
Tests are conducted at fixed cache hit rate(s) with variable working set sizes.

Version: 2.0
Date: 2025-10-28
"""

__version__ = "2.0"
__date__ = "2025-10-28"

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
# These questions are designed to prompt the model to generate lengthy, detailed responses
# Focus on technical, coding-related topics that encourage comprehensive answers
QUESTION_BANK = [
    # Algorithm analysis and explanations
    "Please provide a comprehensive analysis of the QuickSort algorithm, including its implementation in Python, time complexity analysis, space complexity, best/worst/average cases, optimizations like 3-way partitioning, comparison with other sorting algorithms, and real-world applications. Include detailed code examples and explain each step thoroughly.",

    "Explain in detail how hash tables work, including collision resolution strategies (chaining vs open addressing), load factor management, resizing mechanisms, hash function design principles, performance characteristics, and implementation details. Provide code examples demonstrating a complete hash table implementation from scratch.",

    "Write a detailed tutorial on implementing a binary search tree, including insertion, deletion, searching, tree traversal algorithms (inorder, preorder, postorder, level-order), balancing concepts, and how self-balancing trees like AVL and Red-Black trees improve performance. Include complete code implementations with explanations.",

    "Provide an in-depth explanation of dynamic programming, including the principles of optimal substructure and overlapping subproblems. Explain memoization vs tabulation approaches, and walk through detailed solutions to classic problems like longest common subsequence, knapsack problem, edit distance, and matrix chain multiplication with code and complexity analysis.",

    "Explain comprehensively how graph algorithms work, including depth-first search, breadth-first search, Dijkstra's shortest path, Bellman-Ford, Floyd-Warshall, minimum spanning trees (Kruskal's and Prim's algorithms), and topological sorting. Provide implementations and discuss time/space complexity for each.",

    # System design and architecture
    "Design a detailed architecture for a distributed caching system like Redis or Memcached. Explain data partitioning strategies, replication mechanisms, consistency models, eviction policies, persistence options, client-server protocol design, and how to handle network partitions. Include detailed diagrams and code examples.",

    "Explain in comprehensive detail how a modern web browser works, from parsing HTML/CSS/JavaScript to rendering the page. Cover the rendering engine, JavaScript engine, networking layer, security sandbox, memory management, and optimization techniques. Discuss how browsers handle concurrency and asynchronous operations.",

    "Provide a detailed explanation of how database indexing works, including B-tree and B+tree structures, clustered vs non-clustered indexes, covering indexes, index selectivity, query optimization, index maintenance overhead, and when to use different index types. Include examples of creating and using indexes effectively.",

    "Explain comprehensively how a garbage collector works in modern programming languages. Cover mark-and-sweep, generational collection, reference counting, tri-color marking, concurrent and parallel collection strategies, write barriers, and tuning parameters. Compare different GC implementations (Java, Python, Go).",

    # Coding stories and scenarios
    "Write a detailed story about a team of engineers debugging a critical production issue in a distributed system. Include their investigation process, the tools they used, how they traced the problem through multiple services, the root cause analysis, and the fix they implemented. Make it technically detailed with realistic debugging scenarios.",

    "Tell an elaborate story about designing and implementing a real-time collaborative code editor like Google Docs but for programming. Explain the technical challenges of operational transformation or CRDTs, conflict resolution, presence awareness, syntax highlighting synchronization, and how the system handles network latency and disconnections.",

    "Describe in detail the journey of building a high-performance API service from scratch, including choosing the tech stack, implementing rate limiting, caching strategies, database optimization, load balancing, monitoring and observability, CI/CD pipeline, and scaling from 100 to 10 million requests per day. Include code examples and architectural decisions.",

    # Deep technical explanations
    "Provide a comprehensive explanation of how modern CPUs execute instructions, including the instruction pipeline, branch prediction, speculative execution, out-of-order execution, register renaming, cache hierarchies (L1/L2/L3), memory barriers, and SIMD instructions. Explain how these concepts affect code performance.",

    "Explain in detail how TCP/IP networking works, from the physical layer up through application protocols. Cover packet structure, three-way handshake, flow control, congestion control, sliding window protocol, retransmission strategies, and how modern optimizations like BBR congestion control improve performance.",

    "Provide a detailed analysis of how compilers work, from lexical analysis and parsing through code generation and optimization. Explain the different phases, intermediate representations, optimization passes, register allocation, instruction selection, and how modern JIT compilers achieve high performance.",

    "Explain comprehensively how modern machine learning inference works, including model architectures (transformers, CNNs), quantization techniques, batching strategies, KV-cache optimization for autoregressive generation, attention mechanisms, and hardware acceleration using GPUs and specialized chips.",

    # Complex problem-solving
    "Walk through a detailed solution to designing a URL shortening service like bit.ly at scale. Cover the hashing strategy, database schema, handling collisions, custom short URLs, analytics tracking, rate limiting, geographic distribution, caching, and how to handle billions of URLs with low latency.",

    "Explain in detail how to implement a thread-safe LRU cache from scratch, including the data structures needed (hash map + doubly linked list), synchronization mechanisms, lock-free alternatives using atomic operations, memory management considerations, and performance optimization techniques. Include complete code with explanations.",

    "Provide a comprehensive guide to implementing a search engine, covering web crawling strategies, inverted index construction, ranking algorithms (TF-IDF, PageRank), query processing, autocomplete, distributed searching, and scaling to billions of documents. Include detailed explanations of each component.",

    "Explain how to build a real-time streaming data pipeline, covering message queue systems (Kafka, RabbitMQ), stream processing frameworks (Flink, Spark Streaming), windowing operations, state management, exactly-once semantics, backpressure handling, and monitoring. Include architecture diagrams and code examples.",

    # Additional variety
    "Write a detailed technical post-mortem of a hypothetical large-scale outage, explaining the cascade failure, how monitoring detected it, the incident response process, communication strategies, mitigation steps, root cause analysis, and the long-term architectural changes implemented to prevent recurrence.",

    "Explain comprehensively how version control systems like Git work internally, including the object model (blobs, trees, commits), DAG structure, merging strategies, rebasing, conflict resolution, pack files, and distributed workflows. Discuss advanced topics like bisect, cherry-pick, and submodules.",

    "Provide an in-depth explanation of how container orchestration systems like Kubernetes work, covering pods, services, deployments, scheduling algorithms, resource management, networking (CNI), storage (CSI), service mesh integration, and autoscaling mechanisms. Include practical deployment scenarios.",

    "Explain in detail how modern databases achieve ACID properties, covering transaction isolation levels, two-phase locking, multi-version concurrency control (MVCC), write-ahead logging, recovery mechanisms, and distributed transaction protocols like 2PC and Paxos/Raft for consensus.",

    "Write a comprehensive guide to optimizing Python code performance, covering profiling tools, algorithmic improvements, data structure selection, vectorization with NumPy, using Cython or PyPy, async/await for I/O-bound tasks, multiprocessing for CPU-bound work, and memory optimization techniques. Include before/after code examples.",
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
    file_handler = logging.FileHandler('working_set_tester.log', mode='a')
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
    api_endpoints: List[str]  # Support multiple endpoints for load balancing
    context_sizes: List[int]
    working_set_sizes: List[int]  # List of working set sizes to test
    cache_hit_rates: List[int]  # List of cache hit rates to test
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
    skip_graphs: bool
    force_restart: bool
    verbose: bool
    chunk_size: int = 256
    seed: Optional[int] = None
    kv_cache_bytes: int = 2  # 1 or 2 bytes per element for KV cache calculation
    strict_time_window: bool = False  # Only include requests completed within duration window
    mode: str = "sustained"  # "sustained" or "fixed"
    assessment_period: int = 30  # For sustained mode: duration of each assessment period
    min_tokens_per_req: Optional[float] = None  # Minimum average output tokens/s per request threshold
    init_strategy: str = "min"  # "min" or "max" - for sustained mode: initialize min or max working set
    fixed_concurrency: Optional[int] = None  # For fixed mode: single fixed concurrency level

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
    working_set_size: int
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
    working_set_size: int
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
    """Aggregated metrics for a working set size test"""
    context_size: int
    working_set_size: int  # Max working set size (or single value for fixed mode)
    cache_hit_rate: int
    model: str
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
    min_working_set_size: Optional[int] = None  # For sustained mode: min working set size

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class AssessmentPeriodMetrics:
    """Metrics for a single assessment period in sustained mode"""
    period_number: int
    context_size: int
    cache_hit_rate: int
    working_set_size: int
    start_time: float
    end_time: float
    duration: float
    concurrency_level: int
    num_requests_launched: int
    num_requests_completed: int
    total_input_tokens: int
    total_output_tokens: int
    input_tokens_per_sec: float
    output_tokens_per_sec: float
    avg_ttft: float
    median_ttft: float
    p95_ttft: float
    p99_ttft: float
    avg_ttlt: float
    median_ttlt: float
    p95_ttlt: float
    p99_ttlt: float
    avg_itl: float
    median_itl: float
    p95_itl: float
    p99_itl: float
    avg_output_tokens_per_request: float
    measured_ttft: float  # The TTFT metric used for decision (p95/avg/max)
    decision: str  # "RAMP_UP", "RAMP_DOWN", "STAY", "MAX_REACHED", "MIN_REACHED"
    next_concurrency: int  # Concurrency for next period

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

    def is_test_completed(self, context_size: int, working_set_size: int, cache_hit_rate: int) -> bool:
        """Check if a specific test is already completed"""
        test_key = f"{context_size}_{working_set_size}_{cache_hit_rate}"
        return test_key in self.state.get('completed_tests', [])

    def mark_test_completed(self, context_size: int, working_set_size: int, cache_hit_rate: int):
        """Mark a test as completed"""
        test_key = f"{context_size}_{working_set_size}_{cache_hit_rate}"
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
        self.num_prompts_in_rotation: int = 0  # For init=max mode: tracks how many prompts are active

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
        # Determine which prompts are active (for init=max mode with gradual rotation)
        num_active = self.num_prompts_in_rotation if self.num_prompts_in_rotation > 0 else len(self.prompts)
        active_prompts = self.prompts[:num_active]

        if random_selection:
            import random
            return random.choice(active_prompts)
        else:
            # Cycle through deterministically (only through active prompts)
            prompt = active_prompts[self.current_index % len(active_prompts)]
            self.current_index = (self.current_index + 1) % len(active_prompts)
            return prompt


class APIClient:
    """Manages OpenAI API client and requests with load balancing support"""

    def __init__(self, api_endpoints: List[str], model: str):
        self.api_endpoints = api_endpoints
        self.model = model
        self.clients = []
        self.current_index = 0

        # Create a client for each endpoint
        for endpoint in api_endpoints:
            # Ensure base_url ends with /v1 for OpenAI client
            base_url = endpoint.rstrip('/')
            if not base_url.endswith('/v1'):
                base_url = base_url + '/v1'
            client = openai.AsyncOpenAI(
                api_key="EMPTY",
                base_url=base_url
            )
            self.clients.append({'endpoint': endpoint, 'base_url': base_url, 'client': client})

        if len(api_endpoints) == 1:
            logger.info(f"API Client initialized: {api_endpoints[0]} (base_url: {self.clients[0]['base_url']})")
        else:
            logger.info(f"API Client initialized with {len(api_endpoints)} endpoints:")
            for i, client_info in enumerate(self.clients):
                logger.info(f"  [{i+1}] {client_info['endpoint']} (base_url: {client_info['base_url']})")

    def get_next_client(self):
        """Get next client using round-robin load balancing"""
        client_info = self.clients[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.clients)
        return client_info

    async def detect_model(self) -> str:
        """Auto-detect model from API (uses first endpoint)"""
        try:
            first_endpoint = self.api_endpoints[0]
            models_url = first_endpoint.rstrip('/') + '/v1/models'
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
            raise ValueError(f"Could not detect model from {self.api_endpoints[0]}/v1/models")

    async def send_request(self, prompt: str, max_tokens: int) -> Tuple[str, float, float, float, int, int]:
        """
        Send a single request and return metrics
        Uses round-robin load balancing across multiple endpoints if configured.
        Returns: (response_text, ttft, ttlt, generation_time, prompt_tokens, completion_tokens)
        """
        start_time = time.time()
        first_token_time = None
        last_token_time = None
        response_text = ""

        # Get next client using round-robin load balancing
        client_info = self.get_next_client()

        try:
            response = await client_info['client'].chat.completions.create(
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
        description="Working Set Size Performance Testing Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Required arguments
    parser.add_argument("--api-endpoint", type=str, nargs='+', required=True,
                       help="OpenAI-compatible API endpoint(s). Can specify multiple for load balancing (e.g., http://localhost:8125 http://localhost:8126)")
    parser.add_argument("--context-sizes", type=int, nargs='+', required=True,
                       help="Context lengths to test (e.g., 8000 32000 64000)")
    parser.add_argument("--min-working-set-size", type=int, required=True,
                       help="Minimum working set size in tokens (e.g., 100000)")
    parser.add_argument("--max-working-set-size", type=int, required=True,
                       help="Maximum working set size in tokens (e.g., 5000000)")
    parser.add_argument("--working-set-increments", type=int, required=True,
                       help="Number of working set size steps to test (e.g., 10)")

    # Optional arguments
    parser.add_argument("--output-tokens", type=int, default=256,
                       help="Output tokens per request (default: 256)")
    parser.add_argument("--max-ttft", type=float, default=None,
                       help="TTFT threshold in seconds (e.g., 2.0). Required for sustained mode (or --min-tokens-per-req). "
                            "Limits Time To First Token to ensure good prefill performance.")
    parser.add_argument("--ttft-metric", type=str, default="p95",
                       choices=["max", "avg", "p95"],
                       help="TTFT metric to use for threshold: max (maximum), avg (average), p95 (95th percentile). Default: p95")
    parser.add_argument("--min-tokens-per-req", type=float, default=None,
                       help="Minimum average output tokens/s per request (e.g., 30). Optional if --max-ttft is specified. "
                            "Ensures good generation speed per request for user experience. "
                            "Concurrency level is rejected if average output tokens/s per request falls below this value.")
    parser.add_argument("--output-dir", type=str, default="./output",
                       help="Output directory (default: ./output)")
    parser.add_argument("--tokenizer", type=str,
                       default="Qwen/Qwen2.5-Coder-32B-Instruct",
                       help="Tokenizer model ID (default: Qwen/Qwen2.5-Coder-32B-Instruct)")
    parser.add_argument("--test-duration", type=int, default=300,
                       help="Max duration per cache hit rate test in seconds (default: 300)")
    parser.add_argument("--ramp-duration", type=int, default=60,
                       help="Duration per concurrency level during ramp phase in seconds (default: 60)")
    parser.add_argument("--cache-hit-rates", type=int, nargs='+', default=[100],
                       help="Cache hit rates to test (default: [100])")
    parser.add_argument("--reinit-strategy", type=str, default="per_working_set",
                       choices=["once", "per_working_set", "per_test"],
                       help="When to reinitialize working set cache (default: per_working_set). "
                            "Choices: 'once' = once per context size, 'per_working_set' = before each working set size, "
                            "'per_test' = before each concurrency level")
    parser.add_argument("--random-working-set-selection", action="store_true",
                       help="Use random selection instead of cycling")
    parser.add_argument("--num-retries", type=int, default=0,
                       help="Number of retries at peak concurrency (default: 0)")
    parser.add_argument("--start-concurrency", type=int, default=2,
                       help="Starting concurrent requests (default: 2)")
    parser.add_argument("--concurrency-increment", type=int, default=2,
                       help="Concurrency increment (default: 2)")
    parser.add_argument("--max-concurrency", type=int, default=1000,
                       help="Maximum concurrent requests (default: 1000)")
    parser.add_argument("--init-concurrency", type=int, default=16,
                       help="Concurrent requests for working set initialization (default: 16)")
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
    parser.add_argument("--mode", type=str, default="sustained",
                       choices=["sustained", "fixed"],
                       help="Test mode (default: sustained). "
                            "'sustained' = sustained load with adaptive concurrency and working set growth, "
                            "'fixed' = fixed concurrency with working set growth (same assessment periods, no concurrency adjustment)")
    parser.add_argument("--fixed-concurrency", type=int,
                       help="Fixed concurrency level for fixed mode. Required for fixed mode.")
    parser.add_argument("--assessment-period", type=int, default=30,
                       help="Assessment period duration in seconds (default: 30). Used for both sustained and fixed modes.")
    parser.add_argument("--init-strategy", type=str, default="min",
                       choices=["min", "max"],
                       help="Initialization strategy for sustained mode (default: min). "
                            "'min' = initialize only min working set, add new prompts at growth (cache misses), "
                            "'max' = initialize all prompts upfront, add pre-warmed prompts at growth (no cache misses)")
    parser.add_argument("--brief", action="store_true",
                       help="Brief output mode for agents - minimal, parseable output")

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

    # Validate min_tokens_per_req
    if args.min_tokens_per_req is not None and args.min_tokens_per_req <= 0:
        raise ValueError(f"--min-tokens-per-req must be > 0 (got {args.min_tokens_per_req})")

    # Validate max_ttft
    if args.max_ttft is not None and args.max_ttft <= 0:
        raise ValueError(f"--max-ttft must be > 0 (got {args.max_ttft})")

    # Mode-specific validation
    if args.mode == "sustained":
        # Sustained mode requires at least one threshold for concurrency adjustment
        if args.max_ttft is None and args.min_tokens_per_req is None:
            raise ValueError("At least one performance threshold is required for sustained mode: --max-ttft OR --min-tokens-per-req (or both)")
        if args.assessment_period < 1:
            raise ValueError(f"assessment-period must be >= 1 second (got {args.assessment_period})")
    elif args.mode == "fixed":
        # Fixed mode requires a single concurrency level
        if args.fixed_concurrency is None:
            raise ValueError("--fixed-concurrency is required when --mode=fixed")
        if args.fixed_concurrency < 1:
            raise ValueError(f"--fixed-concurrency must be >= 1 (got {args.fixed_concurrency})")
        if args.assessment_period < 1:
            raise ValueError(f"assessment-period must be >= 1 second (got {args.assessment_period})")

    # Generate working set sizes (linear interpolation)
    min_size = args.min_working_set_size
    max_size = args.max_working_set_size
    num_steps = args.working_set_increments

    if min_size <= 0 or max_size <= 0:
        raise ValueError(f"Working set sizes must be positive (min: {min_size}, max: {max_size})")

    if num_steps < 1:
        raise ValueError(f"working-set-increments must be >= 1 (got {num_steps})")

    # Create evenly spaced working set sizes
    if num_steps == 1:
        working_set_sizes = [min_size]
    else:
        working_set_sizes = [int(min_size + (max_size - min_size) * i / (num_steps - 1))
                            for i in range(num_steps)]

    cache_hit_rates = sorted(args.cache_hit_rates)

    # Handle api_endpoint as list
    api_endpoints = args.api_endpoint if isinstance(args.api_endpoint, list) else [args.api_endpoint]

    return TestConfig(
        api_endpoints=api_endpoints,
        context_sizes=sorted(args.context_sizes),
        working_set_sizes=working_set_sizes,
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
        strict_time_window=args.strict_time_window,
        mode=args.mode,
        assessment_period=args.assessment_period,
        min_tokens_per_req=args.min_tokens_per_req,
        init_strategy=args.init_strategy,
        fixed_concurrency=args.fixed_concurrency
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
        output_tokens: Output tokens per request (typically 1 for fast initialization)
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
                            working_set_size: int, cache_hit_rate: int, context_size: int, cached_tokens: int,
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
            working_set_size=working_set_size,
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
                               context_size: int, working_set_size: int, cache_hit_rate: int,
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
                        api_client, prompt, config.output_tokens, working_set_size, cache_hit_rate,
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
        working_set_size=working_set_size,
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



def calculate_period_metrics_simple(all_requests, period_start, period_end):
    """
    Calculate period metrics using simplified completion-time-based approach.

    Unlike cache_rate_tester.py's streaming approach, this filters requests that
    COMPLETED during the assessment period (using finish_time).

    Args:
        all_requests: List of all RequestMetrics collected so far
        period_start: Start timestamp of this period
        period_end: End timestamp of this period

    Returns:
        dict with:
            - completed_requests: List of RequestMetrics that finished in this period
            - input_tokens: Total input tokens from completed requests
            - output_tokens: Total output tokens from completed requests
    """
    # Filter to requests that completed during this period
    # Use finish_time to determine if request completed in this window
    completed_requests = [
        r for r in all_requests
        if period_start <= r.finish_time <= period_end
    ]

    # Calculate totals from completed requests
    total_input = sum(r.cached_tokens + r.unique_tokens for r in completed_requests)
    total_output = sum(r.output_tokens for r in completed_requests)

    return {
        'completed_requests': completed_requests,
        'input_tokens': total_input,
        'output_tokens': total_output
    }


async def run_continuous_mode(config: TestConfig, api_client: APIClient,
                              working_set: WorkingSet, tokenizer: TokenizerManager,
                              context_size: int, working_set_size: int,
                              cache_hit_rate: int, model: str) -> List[AssessmentPeriodMetrics]:
    """
    Run sustained load testing mode - continuously adjust concurrency based on periodic measurements

    Key differences from cache_rate_tester.py:
    1. Uses simplified completion-time-based metrics (no streaming token tracking)
    2. Includes working_set_size parameter
    3. Supports working set growth during the test
    4. Uses existing run_single_request() without streaming modifications

    Returns: List of assessment period metrics
    """
    # Check if we're in fixed or sustained mode
    is_fixed_mode = (config.mode == "fixed")

    if is_fixed_mode:
        logger.info(f"{Colors.PHASE}  Fixed Concurrency Mode: Running at concurrency {config.fixed_concurrency} for {config.test_duration}s{Colors.ENDC}")
        current_concurrency = config.fixed_concurrency
    else:
        logger.info(f"{Colors.PHASE}  Sustained Mode: Assessing every {config.assessment_period}s for {config.test_duration}s{Colors.ENDC}")
        current_concurrency = config.start_concurrency

    all_periods = []
    period_number = 0
    test_start_time = time.time()

    # Track all requests across all periods (for final detailed CSV)
    all_requests = []

    # Track throughput history for plateau detection
    throughput_history = []  # List of (period_number, input_tps, output_tps)
    peak_input_tps = 0
    peak_output_tps = 0
    peak_period = 0

    # Working set growth tracking - scheduled at section boundaries
    # For init=max mode, use num_prompts_in_rotation as initial (not total prompts)
    if working_set.num_prompts_in_rotation > 0:
        initial_num_prompts = working_set.num_prompts_in_rotation
    else:
        initial_num_prompts = len(working_set.prompts)

    target_num_prompts = max(1, int(np.ceil(working_set_size / working_set.rounded_context_size)))
    total_prompts_to_add = target_num_prompts - initial_num_prompts

    # Calculate scheduled growth times
    # Divide test into (increments + 1) sections, add at section boundaries (skip first and last)
    # This leaves buffer time at start and end
    num_sections = len(config.working_set_sizes) + 1  # Use working_set_increments concept
    section_duration = config.test_duration / num_sections
    num_growth_events = num_sections - 1  # Number of boundaries (skip adding at T=0 and T=end)

    # Schedule growth times at section boundaries (excluding T=0)
    scheduled_growth_times = [section_duration * i for i in range(1, num_sections)]

    # Calculate prompts to add at each growth event
    if num_growth_events > 0 and total_prompts_to_add > 0:
        base_prompts_per_event = total_prompts_to_add // num_growth_events
        extra_prompts = total_prompts_to_add % num_growth_events
        # Distribute prompts: first 'extra_prompts' events get base+1, rest get base
        prompts_per_event = [base_prompts_per_event + (1 if i < extra_prompts else 0)
                            for i in range(num_growth_events)]
    else:
        prompts_per_event = []
        scheduled_growth_times = []

    logger.info(f"  Working set growth schedule:")
    logger.info(f"    Initial: {initial_num_prompts} prompts ({initial_num_prompts * working_set.rounded_context_size:,} tokens)")
    logger.info(f"    Target: {target_num_prompts} prompts ({working_set_size:,} tokens)")
    logger.info(f"    Growth events: {num_growth_events} (at section boundaries)")
    if scheduled_growth_times:
        for i, (time_sec, num_prompts) in enumerate(zip(scheduled_growth_times, prompts_per_event)):
            logger.info(f"      T={time_sec:.0f}s: Add {num_prompts} prompt(s)")

    next_growth_index = 0  # Track which growth event is next

    while time.time() - test_start_time < config.test_duration:
        period_number += 1
        period_start = time.time()

        # Calculate remaining time
        elapsed = period_start - test_start_time
        remaining = config.test_duration - elapsed
        period_duration = min(config.assessment_period, remaining)

        if period_duration < 5:  # Not enough time for meaningful assessment
            logger.info(f"  Period {period_number}: Skipping (only {period_duration:.1f}s remaining)")
            break

        logger.info(f"")
        logger.info(f"{Colors.OKCYAN}  Period {period_number}: Running at concurrency {current_concurrency} for {period_duration:.1f}s{Colors.ENDC}")

        # Run requests at current concurrency for the assessment period
        active_tasks = []
        request_counter = 0
        num_launched = 0

        # Track how many requests we've collected so far (to calculate new requests this period)
        requests_before_period = len(all_requests)

        try:
            while time.time() - period_start < period_duration:
                # Check if we should grow working set based on scheduled times
                current_time = time.time()
                elapsed = current_time - test_start_time

                # Check if we've passed the next scheduled growth time
                if (next_growth_index < len(scheduled_growth_times) and
                    elapsed >= scheduled_growth_times[next_growth_index]):

                    prompts_to_add = prompts_per_event[next_growth_index]

                    if prompts_to_add > 0:
                        if config.init_strategy == "max":
                            # init=max: Prompts already generated and initialized
                            # Just increase the number in rotation
                            old_num_in_rotation = working_set.num_prompts_in_rotation
                            working_set.num_prompts_in_rotation = min(
                                len(working_set.prompts),
                                working_set.num_prompts_in_rotation + prompts_to_add
                            )
                            actual_added = working_set.num_prompts_in_rotation - old_num_in_rotation
                            new_ws_size = working_set.num_prompts_in_rotation * working_set.rounded_context_size

                            logger.info(f"")
                            logger.info(f"{Colors.OKGREEN}  📈 Working Set Growth: Activated {actual_added} pre-warmed prompt(s){Colors.ENDC}")
                            logger.info(f"{Colors.OKGREEN}     Now: {working_set.num_prompts_in_rotation}/{len(working_set.prompts)} prompts in rotation ({new_ws_size:,} tokens){Colors.ENDC}")
                            logger.info(f"")
                        else:
                            # init=min: Generate new prompts on-the-fly (NOT initialized - cache misses!)
                            current_num_prompts = len(working_set.prompts)
                            for i in range(prompts_to_add):
                                prompt_idx = current_num_prompts + i
                                prompt_seed = (working_set.seed + prompt_idx) if working_set.seed is not None else None
                                tokens = tokenizer.generate_dummy_tokens(
                                    working_set.rounded_context_size,
                                    seed=prompt_seed,
                                    prompt_number=prompt_idx
                                )
                                working_set.prompts.append(tokens)

                            # Update tracking
                            new_ws_size = len(working_set.prompts) * working_set.rounded_context_size

                            # Log growth with OKGREEN color
                            logger.info(f"")
                            logger.info(f"{Colors.OKGREEN}  📈 Working Set Growth: Added {prompts_to_add} new prompt(s) (cache misses){Colors.ENDC}")
                            logger.info(f"{Colors.OKGREEN}     Now: {len(working_set.prompts)} prompts ({new_ws_size:,} tokens){Colors.ENDC}")
                            logger.info(f"")

                    # Mark this growth event as complete
                    next_growth_index += 1

                # Launch requests up to concurrency limit
                while len(active_tasks) < current_concurrency and time.time() - period_start < period_duration:
                    request_id = f"P{period_number}_c{current_concurrency}_r{request_counter}"
                    phase_id = f"SUSTAINED_P{period_number}"
                    request_seed = (config.seed + request_counter) if config.seed is not None else None

                    # Construct prompt
                    prompt, cached_tok, unique_tok = construct_prompt(
                        working_set, tokenizer, cache_hit_rate, context_size,
                        config.random_selection, request_seed, question_index=request_counter
                    )

                    # Launch request using existing run_single_request (no streaming params)
                    task = asyncio.create_task(
                        run_single_request(
                            api_client, prompt, config.output_tokens, working_set_size, cache_hit_rate,
                            context_size, cached_tok, unique_tok, current_concurrency, request_id, phase_id,
                            verbose=config.verbose
                        )
                    )
                    active_tasks.append(task)
                    request_counter += 1
                    num_launched += 1

                    # Small delay
                    await asyncio.sleep(0.01)

                # Wait for at least one task to complete
                if active_tasks:
                    done, pending = await asyncio.wait(
                        active_tasks, return_when=asyncio.FIRST_COMPLETED
                    )
                    active_tasks = list(pending)

                    for task in done:
                        try:
                            result = await task
                            all_requests.append(result)
                        except Exception as e:
                            logger.error(f"Task failed: {e}")

            # Period duration ended - wait for remaining tasks to complete
            period_end_time = period_start + period_duration

            if active_tasks:
                logger.debug(f"    Period {period_number}: Waiting for {len(active_tasks)} remaining tasks...")
                remaining_results = await asyncio.gather(*active_tasks, return_exceptions=True)

                for result in remaining_results:
                    if isinstance(result, RequestMetrics):
                        # Add to all_requests for detailed CSV
                        all_requests.append(result)
                    elif isinstance(result, Exception):
                        logger.error(f"Task failed: {result}")

        except Exception as e:
            logger.error(f"Error during period {period_number}: {e}")
            # Cancel remaining tasks
            for task in active_tasks:
                task.cancel()
            raise

        # SIMPLIFIED METRICS: Calculate based on request completion times
        # Filter to requests that completed during this period
        period_metrics = calculate_period_metrics_simple(all_requests, period_start, period_end_time)

        completed_requests = period_metrics['completed_requests']
        total_input = period_metrics['input_tokens']
        total_output = period_metrics['output_tokens']
        num_completed = len(completed_requests)

        logger.debug(f"    Period {period_number}: {num_launched} launched, {num_completed} completed ({len(all_requests) - requests_before_period} new requests total)")

        # Calculate metrics for this period
        if num_completed == 0:
            # No requests completed - concurrency is too high, need to ramp down
            if current_concurrency <= config.start_concurrency:
                logger.warning(f"    Period {period_number}: No requests completed! Already at minimum concurrency {current_concurrency}")
                decision = "MIN_REACHED"
                next_concurrency = current_concurrency
            else:
                # Ramp down aggressively when no completions
                next_concurrency = max(config.start_concurrency, current_concurrency - config.concurrency_increment * 2)
                decision = "RAMP_DOWN"
                logger.warning(f"    Period {period_number}: No requests completed! RAMP DOWN: {current_concurrency} -> {next_concurrency}")

            # Create empty period record
            # Use num_prompts_in_rotation if set (init=max mode), otherwise use total prompts
            active_prompts = working_set.num_prompts_in_rotation if working_set.num_prompts_in_rotation > 0 else len(working_set.prompts)
            empty_ws_size = active_prompts * working_set.rounded_context_size
            period_record = AssessmentPeriodMetrics(
                period_number=period_number,
                context_size=context_size,
                cache_hit_rate=cache_hit_rate,
                working_set_size=empty_ws_size,  # Current active WS size
                start_time=period_start,
                end_time=time.time(),
                duration=period_duration,
                concurrency_level=current_concurrency,
                num_requests_launched=num_launched,
                num_requests_completed=0,
                total_input_tokens=0,
                total_output_tokens=0,
                input_tokens_per_sec=0,
                output_tokens_per_sec=0,
                avg_ttft=0,
                median_ttft=0,
                p95_ttft=0,
                p99_ttft=0,
                avg_ttlt=0,
                median_ttlt=0,
                p95_ttlt=0,
                p99_ttlt=0,
                avg_itl=0,
                median_itl=0,
                p95_itl=0,
                p99_itl=0,
                avg_output_tokens_per_request=0,
                measured_ttft=0,
                decision=decision,
                next_concurrency=next_concurrency
            )
            all_periods.append(period_record)
            current_concurrency = next_concurrency
            continue

        # Calculate period throughput based on actual period duration
        actual_period_duration = time.time() - period_start

        input_tps = total_input / actual_period_duration if actual_period_duration > 0 else 0
        output_tps = total_output / actual_period_duration if actual_period_duration > 0 else 0

        # Calculate TTFT stats from completed requests
        ttfts = [m.ttft for m in completed_requests if m.ttft > 0]
        avg_ttft = np.mean(ttfts) if ttfts else 0
        median_ttft = np.median(ttfts) if ttfts else 0
        p95_ttft = np.percentile(ttfts, 95) if ttfts else 0
        p99_ttft = np.percentile(ttfts, 99) if ttfts else 0
        max_ttft = np.max(ttfts) if ttfts else 0

        # Calculate TTLT stats from completed requests
        ttlts = [m.ttlt for m in completed_requests if m.ttlt > 0]
        avg_ttlt = np.mean(ttlts) if ttlts else 0
        median_ttlt = np.median(ttlts) if ttlts else 0
        p95_ttlt = np.percentile(ttlts, 95) if ttlts else 0
        p99_ttlt = np.percentile(ttlts, 99) if ttlts else 0

        # Calculate ITL stats from completed requests
        itls = [m.itl for m in completed_requests if m.itl > 0]
        avg_itl = np.mean(itls) if itls else 0
        median_itl = np.median(itls) if itls else 0
        p95_itl = np.percentile(itls, 95) if itls else 0
        p99_itl = np.percentile(itls, 99) if itls else 0

        # Calculate avg output tokens/s per request from completed requests
        output_rates = [m.output_tokens / m.generation_time
                       for m in completed_requests
                       if m.generation_time > 0 and m.output_tokens > 0]
        avg_output_per_request = np.mean(output_rates) if output_rates else 0

        # Use average tokens per request (matching cache_rate_tester behavior)
        measured_tokens_per_req = avg_output_per_request
        tokens_metric_name = "Avg Tokens/Req"

        # Choose measured TTFT based on config
        if config.ttft_metric == "max":
            measured_ttft = max_ttft
            ttft_metric_name = "Max TTFT"
        elif config.ttft_metric == "avg":
            measured_ttft = avg_ttft
            ttft_metric_name = "Avg TTFT"
        else:  # p95
            measured_ttft = p95_ttft
            ttft_metric_name = "P95 TTFT"

        # Decide: RAMP_UP, RAMP_DOWN, or STAY (only for sustained mode)
        decision = "FIXED" if is_fixed_mode else "STAY"
        next_concurrency = current_concurrency

        # Skip concurrency adjustment for fixed mode
        if is_fixed_mode:
            # In fixed mode, just log the metrics without adjustment
            # Still track peak for reporting
            if input_tps > peak_input_tps:
                peak_input_tps = input_tps
                peak_output_tps = output_tps
                peak_period = period_number
            logger.info(f"    {ttft_metric_name}: {measured_ttft:.3f}s | {tokens_metric_name}: {measured_tokens_per_req:.1f} tok/s")
        else:
            # Check if thresholds exceeded (sustained mode only)
            ttft_exceeded = (config.max_ttft is not None) and (measured_ttft > config.max_ttft)
            tokens_per_req_exceeded = (config.min_tokens_per_req is not None) and (measured_tokens_per_req < config.min_tokens_per_req)

            if ttft_exceeded or tokens_per_req_exceeded:
                # Over threshold - need to ramp down
                exceeded_reasons = []
                if ttft_exceeded:
                    exceeded_reasons.append(f"{ttft_metric_name}: {measured_ttft:.3f}s > {config.max_ttft}s")
                if tokens_per_req_exceeded:
                    exceeded_reasons.append(f"{tokens_metric_name}: {measured_tokens_per_req:.1f} tok/s < {config.min_tokens_per_req} tok/s")

                if current_concurrency <= config.start_concurrency:
                    decision = "MIN_REACHED"
                    next_concurrency = current_concurrency
                    logger.warning(f"    Performance threshold(s) exceeded BUT already at minimum concurrency {current_concurrency}")
                    for reason in exceeded_reasons:
                        logger.warning(f"      - {reason}")
                else:
                    # Adaptive ramp down based on severity (conservative: max 2x increment)
                    # Calculate how badly thresholds are exceeded
                    ttft_overshoot = 0
                    tokens_shortfall = 0

                    if ttft_exceeded:
                        ttft_overshoot = (measured_ttft - config.max_ttft) / config.max_ttft
                    if tokens_per_req_exceeded:
                        tokens_shortfall = (config.min_tokens_per_req - measured_tokens_per_req) / config.min_tokens_per_req if config.min_tokens_per_req > 0 else 0

                    # Use the worst violation to determine ramp down severity
                    max_violation = max(ttft_overshoot, tokens_shortfall)

                    # Adaptive ramp down (conservative: 1x or 2x increment only)
                    if max_violation > 0.5:  # >50% violation - ramp down by 2x
                        down_increment = config.concurrency_increment * 2
                    else:  # <=50% violation - ramp down by 1x
                        down_increment = config.concurrency_increment

                    next_concurrency = max(config.start_concurrency, current_concurrency - down_increment)
                    decision = "RAMP_DOWN"
                    logger.info(f"    Performance threshold(s) exceeded -> RAMP DOWN: {current_concurrency} -> {next_concurrency} (-{current_concurrency - next_concurrency})")
                    for reason in exceeded_reasons:
                        logger.info(f"      - {reason}")
            else:
                # Under threshold - ramp up to find optimal concurrency
                if current_concurrency >= config.max_concurrency:
                    decision = "MAX_REACHED"
                    next_concurrency = current_concurrency
                    logger.info(f"    Performance thresholds OK BUT already at max concurrency {current_concurrency}")
                else:
                    # Update peak tracking (for informational purposes)
                    if input_tps > peak_input_tps:
                        peak_input_tps = input_tps
                        peak_output_tps = output_tps
                        peak_period = period_number

                    # Thresholds are OK - ramp up to increase throughput
                    # Calculate headroom based on thresholds
                    ttft_headroom = (config.max_ttft - measured_ttft) / config.max_ttft if config.max_ttft else 1.0

                    # Also consider tokens/req headroom if configured
                    tokens_headroom = 1.0
                    if config.min_tokens_per_req and config.min_tokens_per_req > 0:
                        tokens_headroom = (measured_tokens_per_req - config.min_tokens_per_req) / config.min_tokens_per_req

                    # Use the minimum headroom (most constrained)
                    min_headroom = min(ttft_headroom, tokens_headroom)

                    # Use headroom to determine increment (1x to 10x)
                    if min_headroom > 0.7:
                        adaptive_increment = config.concurrency_increment * 10
                    elif min_headroom > 0.5:
                        adaptive_increment = config.concurrency_increment * 5
                    elif min_headroom > 0.3:
                        adaptive_increment = config.concurrency_increment * 3
                    elif min_headroom > 0.15:
                        adaptive_increment = config.concurrency_increment * 2
                    else:
                        adaptive_increment = config.concurrency_increment

                    next_concurrency = min(config.max_concurrency, current_concurrency + adaptive_increment)
                    decision = "RAMP_UP"

                    # Show peak info if we have it
                    peak_info = f" (peak: {peak_input_tps:,.0f} tok/s @ P{peak_period})" if peak_input_tps > 0 else ""
                    logger.info(f"    Performance thresholds OK (headroom: {min_headroom:.1%}) -> RAMP UP: {current_concurrency} -> {next_concurrency} (+{next_concurrency - current_concurrency}){peak_info}")

        # Print period summary
        logger.info(f"{Colors.METRIC}    Completed: {num_completed}, Launched: {num_launched}{Colors.ENDC}")
        logger.info(f"{Colors.METRIC}    Input: {input_tps:,.0f} tok/s | Output: {output_tps:,.0f} tok/s{Colors.ENDC}")
        logger.info(f"{Colors.METRIC}    Avg TTFT: {avg_ttft:.3f}s | P95 TTFT: {p95_ttft:.3f}s | P99 TTFT: {p99_ttft:.3f}s{Colors.ENDC}")
        logger.info(f"{Colors.METRIC}    Avg ITL: {avg_itl*1000:.2f}ms | Avg Tokens/Req: {avg_output_per_request:.1f} tok/s{Colors.ENDC}")

        # Create period record (include current working set size)
        # Use num_prompts_in_rotation if set (init=max mode), otherwise use total prompts
        active_prompts = working_set.num_prompts_in_rotation if working_set.num_prompts_in_rotation > 0 else len(working_set.prompts)
        current_ws_size = active_prompts * working_set.rounded_context_size
        period_record = AssessmentPeriodMetrics(
            period_number=period_number,
            context_size=context_size,
            cache_hit_rate=cache_hit_rate,
            working_set_size=current_ws_size,  # Track actual active WS size at this point
            start_time=period_start,
            end_time=time.time(),
            duration=actual_period_duration,
            concurrency_level=current_concurrency,
            num_requests_launched=num_launched,
            num_requests_completed=num_completed,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            input_tokens_per_sec=input_tps,
            output_tokens_per_sec=output_tps,
            avg_ttft=avg_ttft,
            median_ttft=median_ttft,
            p95_ttft=p95_ttft,
            p99_ttft=p99_ttft,
            avg_ttlt=avg_ttlt,
            median_ttlt=median_ttlt,
            p95_ttlt=p95_ttlt,
            p99_ttlt=p99_ttlt,
            avg_itl=avg_itl,
            median_itl=median_itl,
            p95_itl=p95_itl,
            p99_itl=p99_itl,
            avg_output_tokens_per_request=avg_output_per_request,
            measured_ttft=measured_ttft,
            decision=decision,
            next_concurrency=next_concurrency
        )

        all_periods.append(period_record)

        # Track throughput for next period's plateau detection
        throughput_history.append((period_number, input_tps, output_tps))

        # Update concurrency for next period
        current_concurrency = next_concurrency

    # Summary
    total_elapsed = time.time() - test_start_time
    final_ws_size = len(working_set.prompts) * working_set.rounded_context_size
    prompts_added = len(working_set.prompts) - initial_num_prompts

    logger.info(f"")
    logger.info(f"{Colors.SUCCESS}  Continuous mode complete: {period_number} periods in {total_elapsed:.1f}s{Colors.ENDC}")
    logger.info(f"{Colors.METRIC}    Total requests: {len(all_requests)}{Colors.ENDC}")
    logger.info(f"{Colors.METRIC}    Peak throughput: {peak_input_tps:,.0f} input tok/s at period {peak_period}{Colors.ENDC}")
    logger.info(f"{Colors.METRIC}    Working set growth: {initial_num_prompts} -> {len(working_set.prompts)} prompts (+{prompts_added}){Colors.ENDC}")
    logger.info(f"{Colors.METRIC}    Final working set size: {final_ws_size:,} tokens{Colors.ENDC}")

    # Save all requests to detailed CSV
    if all_requests:
        save_detailed_results(all_requests, config.output_dir, context_size)

    return all_periods


def calculate_aggregated_metrics(metrics: List[RequestMetrics], context_size: int,
                                 working_set_size: int, cache_hit_rate: int, peak_concurrency: int,
                                 config: TestConfig, model: str) -> AggregatedMetrics:
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

    # Apply strict time window filter BEFORE calculating throughput if enabled
    if config.strict_time_window:
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

    # Log comparison if values differ significantly (only when strict mode is NOT enabled)
    # When strict mode is enabled, we've already filtered and calculated with that data
    if not config.strict_time_window and strict_input_tps > 0 and abs(strict_input_tps - input_tokens_per_sec) / input_tokens_per_sec > 0.02:
        input_diff_pct = (strict_input_tps - input_tokens_per_sec) / input_tokens_per_sec * 100
        output_diff_pct = (strict_output_tps - output_tokens_per_sec) / output_tokens_per_sec * 100 if output_tokens_per_sec > 0 else 0

        logger.info(f"  ℹ️  Strict time window comparison (requests that started AND finished within ramp window):")
        logger.info(f"      Default: Input={input_tokens_per_sec:,.0f} tok/s, Output={output_tokens_per_sec:,.0f} tok/s")
        logger.info(f"      Strict:  Input={strict_input_tps:,.0f} tok/s ({input_diff_pct:+.1f}%), Output={strict_output_tps:,.0f} tok/s ({output_diff_pct:+.1f}%)")
        if input_diff_pct > 5:
            logger.info(f"      ⚠️  Strict window shows >5% higher throughput - may indicate cleanup overhead")

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
        working_set_size=working_set_size,
        cache_hit_rate=cache_hit_rate,
        model=model,
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


def save_continuous_results(periods: List[AssessmentPeriodMetrics], output_dir: str,
                           context_size: int, working_set_size: int, cache_hit_rate: int):
    """Save continuous mode assessment period metrics to CSV"""
    if not periods:
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_path / f"sustained_periods_ctx{context_size}_ws{working_set_size}_cache{cache_hit_rate}_{timestamp}.csv"

    df = pd.DataFrame([p.to_dict() for p in periods])
    df.to_csv(filename, index=False)
    logger.info(f"Saved sustained mode results to {filename}")

    return filename  # Return filename for graph generation


def generate_sustained_mode_graphs(csv_path: Path, output_dir: str, context_size: int,
                                   working_set_size: int, cache_hit_rate: int, config: TestConfig):
    """Generate graphs from sustained mode CSV results"""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        logger.warning("Plotly not installed - skipping graph generation")
        return

    df = pd.read_csv(csv_path)

    # Calculate summary statistics for annotation
    total_input_tokens = df['total_input_tokens'].sum()
    total_output_tokens = df['total_output_tokens'].sum()
    avg_input_tps = df['input_tokens_per_sec'].mean()
    avg_output_tps = df['output_tokens_per_sec'].mean()
    avg_ttft = df['avg_ttft'].mean()
    avg_output_per_req = df['avg_output_tokens_per_request'].mean()
    total_requests = df['num_requests_completed'].sum()

    # Create figure with multiple subplots
    fig = make_subplots(
        rows=6, cols=1,
        subplot_titles=(
            f'Input Throughput (Context: {context_size:,}, Cache Rate: {cache_hit_rate}%)',
            'Output Throughput',
            'Concurrency Level Adjustments',
            'TTFT Metrics',
            'Output Tokens per Request',
            'Working Set Size Growth'
        ),
        vertical_spacing=0.06
    )

    # Calculate time in minutes from start
    df['time_min'] = (df['end_time'] - df['start_time'].iloc[0]) / 60

    # Row 1: Input Throughput
    fig.add_trace(
        go.Scatter(x=df['time_min'], y=df['input_tokens_per_sec'],
                  mode='lines+markers', name='Input tok/s',
                  line=dict(color='blue')),
        row=1, col=1
    )

    # Row 2: Output Throughput
    fig.add_trace(
        go.Scatter(x=df['time_min'], y=df['output_tokens_per_sec'],
                  mode='lines+markers', name='Output tok/s',
                  line=dict(color='green')),
        row=2, col=1
    )

    # Row 3: Concurrency
    fig.add_trace(
        go.Scatter(x=df['time_min'], y=df['concurrency_level'],
                  mode='lines+markers', name='Concurrency',
                  line=dict(color='purple')),
        row=3, col=1
    )

    # Row 4: TTFT
    fig.add_trace(
        go.Scatter(x=df['time_min'], y=df['avg_ttft'],
                  mode='lines+markers', name='Avg TTFT',
                  line=dict(color='orange')),
        row=4, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['time_min'], y=df['p95_ttft'],
                  mode='lines+markers', name='P95 TTFT',
                  line=dict(color='red')),
        row=4, col=1
    )

    # Add max TTFT threshold line if configured
    if config.max_ttft:
        fig.add_hline(y=config.max_ttft, line_dash="dash", line_color="red",
                     annotation_text=f"Max TTFT Threshold ({config.max_ttft}s)",
                     row=4, col=1)

    # Row 5: Output Tokens per Request
    fig.add_trace(
        go.Scatter(x=df['time_min'], y=df['avg_output_tokens_per_request'],
                  mode='lines+markers', name='Avg Tokens/Req',
                  line=dict(color='teal')),
        row=5, col=1
    )

    # Add min tokens per request threshold line if configured
    if config.min_tokens_per_req:
        fig.add_hline(y=config.min_tokens_per_req, line_dash="dash", line_color="red",
                     annotation_text=f"Min Threshold ({config.min_tokens_per_req} tok/s)",
                     row=5, col=1)

    # Row 6: Working Set Size
    fig.add_trace(
        go.Scatter(x=df['time_min'], y=df['working_set_size'],
                  mode='lines+markers', name='Working Set Size',
                  line=dict(color='brown'), fill='tozeroy'),
        row=6, col=1
    )

    # Update axes labels
    fig.update_xaxes(title_text="Time (minutes)", row=6, col=1)
    fig.update_yaxes(title_text="Tokens/sec", row=1, col=1)
    fig.update_yaxes(title_text="Tokens/sec", row=2, col=1)
    fig.update_yaxes(title_text="Concurrency", row=3, col=1)
    fig.update_yaxes(title_text="Seconds", row=4, col=1)
    fig.update_yaxes(title_text="Tokens/sec", row=5, col=1)
    fig.update_yaxes(title_text="Tokens", row=6, col=1)

    # Create statistics summary text
    stats_text = (
        f"<b>Summary Statistics:</b><br>"
        f"Total Input Tokens: {total_input_tokens:,} | "
        f"Total Output Tokens: {total_output_tokens:,} | "
        f"Total Requests: {total_requests:,}<br>"
        f"Avg Input: {avg_input_tps:,.0f} tok/s | "
        f"Avg Output: {avg_output_tps:,.0f} tok/s | "
        f"Avg TTFT: {avg_ttft:.3f}s | "
        f"Avg Tokens/Req: {avg_output_per_req:.1f} tok/s"
    )

    # Update layout with top margin for stats annotation
    fig.update_layout(
        height=1700,  # Increased height for stats annotation
        showlegend=True,
        margin=dict(t=120, b=50, l=80, r=80),  # Add top margin for annotation
        title={
            'text': f"Sustained Mode Performance - Context {context_size:,}, Cache {cache_hit_rate}%<br><sub>{stats_text}</sub>",
            'y': 0.98,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=16)
        }
    )

    # Save graph with timestamp from CSV filename
    output_path = Path(output_dir)
    # Extract timestamp from CSV filename (e.g., sustained_periods_ctx30000_ws5000000_cache95_20251114_191042.csv)
    # Timestamp is the last two parts: date_time
    parts = csv_path.stem.split('_')
    if len(parts) >= 2 and parts[-1].isdigit() and parts[-2].isdigit():
        # Has full timestamp: YYYYMMDD_HHMMSS
        timestamp = f"{parts[-2]}_{parts[-1]}"
    elif len(parts) >= 1 and parts[-1].isdigit():
        # Has only time part
        timestamp = parts[-1]
    else:
        # No timestamp, generate one
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    graph_filename = output_path / f"sustained_performance_ctx{context_size}_cache{cache_hit_rate}_{timestamp}.html"
    fig.write_html(str(graph_filename))
    logger.info(f"Generated sustained mode graph: {graph_filename}")

    return graph_filename


def generate_sustained_index_html(csv_path: Path, output_dir: str, context_size: int,
                                  working_set_size: int, cache_hit_rate: int, config: TestConfig):
    """Generate index.html dashboard for sustained mode results"""
    output_path = Path(output_dir)

    # Find all sustained periods CSV files for this context size
    all_csv_files = sorted(output_path.glob(f"sustained_periods_ctx{context_size}_*.csv"),
                          key=lambda p: p.stat().st_mtime, reverse=True)

    # Find all sustained performance graphs (with timestamps)
    all_graph_files = sorted(output_path.glob(f"sustained_performance_ctx{context_size}_cache*_*.html"),
                            key=lambda p: p.stat().st_mtime, reverse=True)

    # Also include graphs without timestamps (legacy format) for backward compatibility
    legacy_graphs = list(output_path.glob(f"sustained_performance_ctx{context_size}_cache*.html"))
    legacy_graphs = [g for g in legacy_graphs if g not in all_graph_files]  # Avoid duplicates
    all_graph_files.extend(sorted(legacy_graphs, key=lambda p: p.stat().st_mtime, reverse=True))

    # Use the most recent CSV for summary stats (the one we just generated)
    df = pd.read_csv(csv_path)

    # Calculate summary statistics from most recent run
    total_periods = len(df)
    total_duration = df['duration'].sum() / 60  # minutes
    peak_input_tps = df['input_tokens_per_sec'].max()
    peak_output_tps = df['output_tokens_per_sec'].max()
    avg_input_tps = df['input_tokens_per_sec'].mean()
    avg_output_tps = df['output_tokens_per_sec'].mean()
    peak_concurrency = df['concurrency_level'].max()
    min_concurrency = df['concurrency_level'].min()
    total_requests = df['num_requests_completed'].sum()
    final_working_set = df['working_set_size'].iloc[-1]
    initial_working_set = df['working_set_size'].iloc[0]

    # Count decisions
    ramp_ups = len(df[df['decision'].str.contains('RAMP_UP', na=False)])
    ramp_downs = len(df[df['decision'].str.contains('RAMP_DOWN', na=False)])

    # Build visualizations list
    graphs_html = ""
    if not all_graph_files:
        graphs_html = '    <p>No performance graphs found.</p>\n'
    else:
        for i, graph_file in enumerate(all_graph_files):
            # Extract info from filename (format: sustained_performance_ctx30000_cache95_20251114_191042.html)
            stem = graph_file.stem  # e.g., sustained_performance_ctx30000_cache95_191042

            # Extract cache rate
            if 'cache' in stem:
                cache_part = stem.split('cache')[1]  # e.g., 95_191042 or 95
                cache_rate = cache_part.split('_')[0]  # e.g., 95
            else:
                cache_rate = "?"

            # Check if has timestamp (last part is numeric and looks like HHMMSS)
            parts = stem.split('_')
            if len(parts) >= 2 and parts[-1].isdigit() and len(parts[-1]) == 6:
                # Has timestamp: YYYYMMDD_HHMMSS format
                date_part = parts[-2]
                time_part = parts[-1]
                # Format timestamp nicely: YYYY-MM-DD HH:MM:SS
                timestamp_str = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]} {time_part[:2]}:{time_part[2:4]}:{time_part[4:6]}"
                graphs_html += f'    <a href="{graph_file.name}" class="graph-link">📊 Cache {cache_rate}% - {timestamp_str}</a>\n'
            else:
                # No timestamp (legacy format or just cache rate)
                graphs_html += f'    <a href="{graph_file.name}" class="graph-link">📊 Cache {cache_rate}%</a>\n'

    # Build CSV files list
    csv_list_html = ""
    for csv_file in all_csv_files:
        timestamp_str = csv_file.stem.split('_')[-1] if '_' in csv_file.stem else csv_file.name
        csv_list_html += f'        <li><a href="{csv_file.name}">Sustained Periods CSV - {timestamp_str}</a></li>\n'

    # Find detailed results CSV files
    detailed_csv_files = sorted(output_path.glob(f"detailed_results_{context_size}_*.csv"),
                               key=lambda p: p.stat().st_mtime, reverse=True)
    for detailed_file in detailed_csv_files:
        timestamp_str = detailed_file.stem.split('_')[-1] if '_' in detailed_file.stem else detailed_file.name
        csv_list_html += f'        <li><a href="{detailed_file.name}">Detailed Results CSV - {timestamp_str}</a></li>\n'

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sustained Mode Results - Context {context_size:,}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        h1 {{ margin: 0; font-size: 2em; }}
        h2 {{ color: #333; margin-top: 30px; }}
        .timestamp {{ opacity: 0.9; margin-top: 10px; }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stat-label {{
            color: #666;
            font-size: 0.9em;
            margin-bottom: 5px;
        }}
        .stat-value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #333;
        }}
        .config-section {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .config-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .config-item {{
            padding: 10px;
            background: #f9f9f9;
            border-radius: 4px;
        }}
        .graph-link {{
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 15px 30px;
            text-decoration: none;
            border-radius: 5px;
            margin: 10px 10px 10px 0;
            font-weight: bold;
        }}
        .graph-link:hover {{
            background: #5568d3;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Sustained Mode Performance Test Results</h1>
        <div class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    </div>

    <h2>Performance Summary</h2>
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-label">Total Duration</div>
            <div class="stat-value">{total_duration:.1f} min</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Total Periods</div>
            <div class="stat-value">{total_periods}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Total Requests</div>
            <div class="stat-value">{total_requests:,}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Peak Input Throughput</div>
            <div class="stat-value">{peak_input_tps:,.0f} tok/s</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Peak Output Throughput</div>
            <div class="stat-value">{peak_output_tps:,.0f} tok/s</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Avg Input Throughput</div>
            <div class="stat-value">{avg_input_tps:,.0f} tok/s</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Avg Output Throughput</div>
            <div class="stat-value">{avg_output_tps:,.0f} tok/s</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Peak Concurrency</div>
            <div class="stat-value">{peak_concurrency}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Concurrency Range</div>
            <div class="stat-value">{min_concurrency}-{peak_concurrency}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Ramp Ups / Downs</div>
            <div class="stat-value">{ramp_ups} / {ramp_downs}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Initial Working Set</div>
            <div class="stat-value">{initial_working_set:,} tok</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Final Working Set</div>
            <div class="stat-value">{final_working_set:,} tok</div>
        </div>
    </div>

    <h2>Visualizations</h2>
{graphs_html}

    <h2>Test Configuration (Most Recent Run)</h2>
    <div class="config-section">
        <div class="config-grid">
            <div class="config-item"><strong>Context Size:</strong> {context_size:,} tokens</div>
            <div class="config-item"><strong>Cache Hit Rate:</strong> {cache_hit_rate}%</div>
            <div class="config-item"><strong>Assessment Period:</strong> {config.assessment_period}s</div>
            <div class="config-item"><strong>Output Tokens:</strong> {config.output_tokens}</div>
            <div class="config-item"><strong>Max TTFT:</strong> {config.max_ttft}s ({config.ttft_metric})</div>
            <div class="config-item"><strong>Min Tokens/Req:</strong> {config.min_tokens_per_req if config.min_tokens_per_req else 'Not set'}</div>
            <div class="config-item"><strong>Start Concurrency:</strong> {config.start_concurrency}</div>
            <div class="config-item"><strong>Concurrency Increment:</strong> {config.concurrency_increment}</div>
            <div class="config-item"><strong>Max Concurrency:</strong> {config.max_concurrency}</div>
        </div>
    </div>

    <h2>Data Files (All Runs)</h2>
    <div class="config-section">
        <ul>
{csv_list_html}
        </ul>
    </div>
</body>
</html>
"""

    output_path = Path(output_dir)
    index_path = output_path / "index.html"
    with open(index_path, 'w') as f:
        f.write(html_content)

    logger.info(f"Generated index.html: {index_path}")
    return index_path


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
                working_set_size=int(row['working_set_size']),
                cache_hit_rate=int(row['cache_hit_rate']),
                model=str(row.get('model', 'unknown')),  # Default to 'unknown' for old CSV files
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


def load_phase_metadata(output_dir: str) -> Dict[Tuple[int, int, int], List[PhaseMetadata]]:
    """
    Load all phase metadata from output directory
    Returns: Dict mapping (context_size, working_set_size, cache_hit_rate) -> List[PhaseMetadata]
    """
    output_path = Path(output_dir)
    phase_files = list(output_path.glob("phase_metadata_*.csv"))

    phases_by_test = {}  # (context_size, working_set_size, cache_rate) -> List[PhaseMetadata]

    for phase_file in phase_files:
        try:
            df = pd.read_csv(phase_file)
            for _, row in df.iterrows():
                context_size = int(row['context_size'])
                working_set_size = int(row['working_set_size'])
                cache_rate = int(row['cache_hit_rate'])
                key = (context_size, working_set_size, cache_rate)

                phase = PhaseMetadata(
                    phase_type=row['phase_type'],
                    phase_id=row['phase_id'],
                    working_set_size=working_set_size,
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
    script_name = "working_set_tester.py"
    command_parts = [f"python {script_name}"]

    # Add all arguments
    # Handle multiple endpoints
    endpoints = args.api_endpoint if isinstance(args.api_endpoint, list) else [args.api_endpoint]
    command_parts.append(f"  --api-endpoint {' '.join(endpoints)}")
    command_parts.append(f"  --context-sizes {' '.join(map(str, args.context_sizes))}")
    command_parts.append(f"  --min-working-set-size {args.min_working_set_size}")
    command_parts.append(f"  --max-working-set-size {args.max_working_set_size}")
    command_parts.append(f"  --working-set-increments {args.working_set_increments}")

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
    if args.cache_hit_rates != [100]:
        command_parts.append(f"  --cache-hit-rates {' '.join(map(str, args.cache_hit_rates))}")
    if args.reinit_strategy != "per_working_set":
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
        f.write(f"# Working Set Tester - Run command\n")
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
                       working_set_size: int, cache_hit_rate: int, max_ttft: float, peak_concurrency: int, output_dir: str,
                       mode: str = "fixed"):
    """
    Generate detailed concurrency visualization for a single working set size test

    Shows ONLY the run phase data (excludes retry runs) to show performance at each
    concurrency level.

    Naming scheme:
    - Fixed mode: fixed_ctx{context_size}_ws{working_set_size}_cache{cache_hit_rate}.html
    - Other modes: ramp_ctx{context_size}_ws{working_set_size}_cache{cache_hit_rate}.html
    """
    if not detailed_metrics:
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Filter to ONLY run phases (exclude RETRY phases)
    # For sustained mode: RAMP_c{concurrency}
    # For fixed mode: FIXED_c{concurrency}_run{n}
    df = pd.DataFrame([m.to_dict() for m in detailed_metrics])

    # Include RAMP phases and FIXED run phases, exclude RETRY phases
    ramp_df = df[
        (df['phase_id'].str.startswith('RAMP')) |
        (df['phase_id'].str.contains('_run') & ~df['phase_id'].str.contains('RETRY'))
    ].copy()

    if len(ramp_df) == 0:
        logger.warning(f"No run phase data found for graph (context={context_size}, ws={working_set_size}, cache_rate={cache_hit_rate})")
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

    # Add horizontal line for TTFT threshold (if specified)
    if max_ttft is not None:
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
    mode_title = "Fixed Concurrency" if mode == "fixed" else "Concurrency Ramp"
    fig.update_layout(
        title=f"{mode_title} Analysis<br>Context: {context_size:,} tokens | Working Set: {working_set_size:,} tokens | Cache Hit Rate: {cache_hit_rate}%",
        height=900,
        showlegend=True,
        hovermode='x unified'
    )

    # Save with naming scheme based on mode
    prefix = "fixed" if mode == "fixed" else "ramp"
    filename = output_path / f"{prefix}_ctx{context_size}_ws{working_set_size}_cache{cache_hit_rate}.html"
    fig.write_html(filename)
    logger.info(f"Generated {mode} mode graph: {filename}")


def generate_graphs(metrics: List[AggregatedMetrics], output_dir: str, config: TestConfig):
    """Generate Plotly visualizations"""
    if not metrics:
        logger.warning("No metrics to visualize")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([m.to_dict() for m in metrics])

    # Deduplicate: keep only the most recent entry for each (context_size, working_set_size, cache_hit_rate)
    # This handles cases where tests were resumed and same configs run multiple times
    df = df.drop_duplicates(subset=['context_size', 'working_set_size', 'cache_hit_rate'], keep='last')

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
    # Store as dict keyed by (context_size, working_set_size, cache_rate) for use across all graphs
    all_detailed_variability = {}

    for context_size in df['context_size'].unique():
        if context_size not in detailed_dfs:
            continue

        detailed_df = detailed_dfs[context_size]

        for cache_rate in df['cache_hit_rate'].unique():
            df_subset = df[(df['context_size'] == context_size) & (df['cache_hit_rate'] == cache_rate)]

            for working_set_size in df_subset['working_set_size'].unique():
                test_key = (context_size, working_set_size, cache_rate)

                # Get phase metadata for this test
                if test_key not in phase_metadata_by_test:
                    logger.debug(f"No phase metadata found for context={context_size}, ws={working_set_size}, cache_rate={cache_rate}")
                    continue

                phases = phase_metadata_by_test[test_key]
                peak_conc = df_subset[df_subset['working_set_size'] == working_set_size]['peak_concurrency'].iloc[0]

                # Filter to peak concurrency phases (RAMP at peak + all RETRY phases)
                # This matches what's shown in the Retry Summary console output
                retry_phases = [p for p in phases
                              if p.concurrency_level == peak_conc and
                                 (p.phase_type == "RETRY" or
                                  (p.phase_type == "RAMP" and p.concurrency_level == peak_conc))]

                if len(retry_phases) == 0:
                    logger.debug(f"No peak concurrency phases for context={context_size}, ws={working_set_size}, cache_rate={cache_rate}")
                    continue

                retry_throughputs = {'input': [], 'output': []}
                retry_ttft_stats = {'avg': [], 'median': [], 'p95': [], 'p99': []}

                for phase in retry_phases:
                    # Get requests that completed within this phase using phase_id
                    phase_requests = detailed_df[
                        (detailed_df['phase_id'] == phase.phase_id) &
                        (detailed_df['working_set_size'] == working_set_size) &
                        (detailed_df['cache_hit_rate'] == cache_rate) &
                        (detailed_df['concurrency_level'] == peak_conc)
                    ]

                    # Apply strict time window filter if enabled
                    if config.strict_time_window and len(phase_requests) > 0:
                        # Filter to only requests that launched AND finished within the phase duration
                        phase_start = phase_requests['launch_time'].min()
                        duration_end = phase_start + phase.duration
                        phase_requests = phase_requests[
                            (phase_requests['launch_time'] < duration_end) &
                            (phase_requests['finish_time'] <= duration_end)
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
                    all_detailed_variability[(context_size, working_set_size, cache_rate)] = {
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

    # Graph 1: Working set size vs performance metrics (per context size and cache hit rate) - now with 3 separate charts
    for context_size in df['context_size'].unique():
        for cache_rate in df['cache_hit_rate'].unique():
            df_subset = df[(df['context_size'] == context_size) & (df['cache_hit_rate'] == cache_rate)].sort_values('working_set_size')

            if len(df_subset) == 0:
                continue

            # Extract detailed variability for this context size and cache rate from pre-calculated dict
            detailed_variability = {}
            for ws_size in df_subset['working_set_size'].unique():
                if (context_size, ws_size, cache_rate) in all_detailed_variability:
                    detailed_variability[ws_size] = all_detailed_variability[(context_size, ws_size, cache_rate)]

            # Create subplot figure with 3 rows (like ramp graph)
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=(
                    'Input Throughput vs Working Set Size',
                    'Output Throughput vs Working Set Size',
                    'TTFT Metrics vs Working Set Size'
                ),
                vertical_spacing=0.12,
                specs=[[{"secondary_y": False}],
                       [{"secondary_y": False}],
                       [{"secondary_y": False}]]
            )

            # Row 1: Input tokens/s with variability bands
            # Add min/max shaded area if we have variability data
            if detailed_variability:
                ws_sizes_with_var = sorted([k for k in detailed_variability.keys()])
                input_min = [detailed_variability[ws]['input_min'] for ws in ws_sizes_with_var]
                input_max = [detailed_variability[ws]['input_max'] for ws in ws_sizes_with_var]

                # Add upper bound
                fig.add_trace(
                    go.Scatter(
                        x=ws_sizes_with_var,
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
                        x=ws_sizes_with_var,
                        y=input_min,
                        mode='lines',
                        name='Input range',
                        line=dict(color='rgba(0, 0, 255, 0)', width=0),
                        fill='tonexty',
                        fillcolor='rgba(0, 0, 255, 0.15)',
                        hovertemplate='Working Set: %{x:,} tokens<br>Min-Max Range<extra></extra>'
                    ),
                    row=1, col=1
                )

            # Add main input throughput line
            # Use mean from retry runs (if available) instead of aggregated mean from all concurrency levels
            if detailed_variability:
                # Use the retry-based mean for working set sizes that have variability data
                input_mean_values = []
                ws_sizes_all = sorted(df_subset['working_set_size'].unique())
                for ws in ws_sizes_all:
                    if ws in detailed_variability:
                        input_mean_values.append(detailed_variability[ws]['input_mean'])
                    else:
                        # Fallback to aggregated value if no detailed data
                        input_mean_values.append(df_subset[df_subset['working_set_size'] == ws]['input_tokens_per_sec'].iloc[0])

                fig.add_trace(
                    go.Scatter(
                        x=ws_sizes_all,
                        y=input_mean_values,
                        mode='lines+markers',
                        name='Input tokens/s (avg)',
                        line=dict(color='blue', width=3),
                        marker=dict(size=8),
                        hovertemplate='Working Set: %{x:,} tokens<br>Input: %{y:,.0f} tok/s<extra></extra>'
                    ),
                    row=1, col=1
                )
            else:
                # No variability data, use aggregated metrics
                fig.add_trace(
                    go.Scatter(
                        x=df_subset['working_set_size'],
                        y=df_subset['input_tokens_per_sec'],
                        mode='lines+markers',
                        name='Input tokens/s (avg)',
                        line=dict(color='blue', width=3),
                        marker=dict(size=8),
                        hovertemplate='Working Set: %{x:,} tokens<br>Input: %{y:,.0f} tok/s<extra></extra>'
                    ),
                    row=1, col=1
                )

            # Row 2: Output tokens/s with variability bands
            # Add min/max shaded area if we have variability data
            if detailed_variability:
                ws_sizes_with_var = sorted([k for k in detailed_variability.keys()])
                output_min = [detailed_variability[ws]['output_min'] for ws in ws_sizes_with_var]
                output_max = [detailed_variability[ws]['output_max'] for ws in ws_sizes_with_var]

                # Add upper bound
                fig.add_trace(
                    go.Scatter(
                        x=ws_sizes_with_var,
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
                        x=ws_sizes_with_var,
                        y=output_min,
                        mode='lines',
                        name='Output range',
                        line=dict(color='rgba(0, 255, 0, 0)', width=0),
                        fill='tonexty',
                        fillcolor='rgba(0, 255, 0, 0.15)',
                        hovertemplate='Working Set: %{x:,} tokens<br>Min-Max Range<extra></extra>'
                    ),
                    row=2, col=1
                )

            # Add main output throughput line
            # Use mean from retry runs (if available) instead of aggregated mean from all concurrency levels
            if detailed_variability:
                # Use the retry-based mean for working set sizes that have variability data
                output_mean_values = []
                ws_sizes_all = sorted(df_subset['working_set_size'].unique())
                for ws in ws_sizes_all:
                    if ws in detailed_variability:
                        output_mean_values.append(detailed_variability[ws]['output_mean'])
                    else:
                        # Fallback to aggregated value if no detailed data
                        output_mean_values.append(df_subset[df_subset['working_set_size'] == ws]['output_tokens_per_sec'].iloc[0])

                fig.add_trace(
                    go.Scatter(
                        x=ws_sizes_all,
                        y=output_mean_values,
                        mode='lines+markers',
                        name='Output tokens/s (avg)',
                        line=dict(color='green', width=3),
                        marker=dict(size=8),
                        hovertemplate='Working Set: %{x:,} tokens<br>Output: %{y:,.0f} tok/s<extra></extra>'
                    ),
                    row=2, col=1
                )
            else:
                # No variability data, use aggregated metrics
                fig.add_trace(
                    go.Scatter(
                        x=df_subset['working_set_size'],
                        y=df_subset['output_tokens_per_sec'],
                        mode='lines+markers',
                        name='Output tokens/s (avg)',
                        line=dict(color='green', width=3),
                        marker=dict(size=8),
                        hovertemplate='Working Set: %{x:,} tokens<br>Output: %{y:,.0f} tok/s<extra></extra>'
                    ),
                    row=2, col=1
                )

            # Row 3: TTFT metrics (avg, median, p95, p99)
            # Use retry-based TTFT values if available, otherwise fall back to aggregated metrics
            if detailed_variability:
                # Use the retry-based TTFT for working set sizes that have variability data
                ws_sizes_all = sorted(df_subset['working_set_size'].unique())
                avg_ttft_values = []
                median_ttft_values = []
                p95_ttft_values = []
                p99_ttft_values = []

                for ws in ws_sizes_all:
                    if ws in detailed_variability:
                        avg_ttft_values.append(detailed_variability[ws]['avg_ttft'])
                        median_ttft_values.append(detailed_variability[ws]['median_ttft'])
                        p95_ttft_values.append(detailed_variability[ws]['p95_ttft'])
                        p99_ttft_values.append(detailed_variability[ws]['p99_ttft'])
                    else:
                        # Fallback to aggregated values if no detailed data
                        avg_ttft_values.append(df_subset[df_subset['working_set_size'] == ws]['avg_ttft'].iloc[0])
                        median_ttft_values.append(df_subset[df_subset['working_set_size'] == ws]['median_ttft'].iloc[0])
                        p95_ttft_values.append(df_subset[df_subset['working_set_size'] == ws]['p95_ttft'].iloc[0])
                        p99_ttft_values.append(df_subset[df_subset['working_set_size'] == ws]['p99_ttft'].iloc[0])

                fig.add_trace(
                    go.Scatter(
                        x=ws_sizes_all,
                        y=avg_ttft_values,
                        mode='lines+markers',
                        name='Avg TTFT',
                        line=dict(color='orange', width=2),
                        marker=dict(size=6),
                        hovertemplate='Working Set: %{x:,} tokens<br>Avg TTFT: %{y:.3f}s<extra></extra>'
                    ),
                    row=3, col=1
                )

                fig.add_trace(
                    go.Scatter(
                        x=ws_sizes_all,
                        y=median_ttft_values,
                        mode='lines+markers',
                        name='Median TTFT',
                        line=dict(color='darkorange', width=2, dash='dot'),
                        marker=dict(size=6),
                        hovertemplate='Working Set: %{x:,} tokens<br>Median TTFT: %{y:.3f}s<extra></extra>'
                    ),
                    row=3, col=1
                )

                fig.add_trace(
                    go.Scatter(
                        x=ws_sizes_all,
                        y=p95_ttft_values,
                        mode='lines+markers',
                        name='P95 TTFT',
                        line=dict(color='red', width=2, dash='dot'),
                        marker=dict(size=6),
                        hovertemplate='Working Set: %{x:,} tokens<br>P95 TTFT: %{y:.3f}s<extra></extra>'
                    ),
                    row=3, col=1
                )

                fig.add_trace(
                    go.Scatter(
                        x=ws_sizes_all,
                        y=p99_ttft_values,
                        mode='lines+markers',
                        name='P99 TTFT',
                        line=dict(color='darkred', width=2, dash='dash'),
                        marker=dict(size=6),
                        hovertemplate='Working Set: %{x:,} tokens<br>P99 TTFT: %{y:.3f}s<extra></extra>'
                    ),
                    row=3, col=1
                )
            else:
                # No variability data, use aggregated metrics
                fig.add_trace(
                    go.Scatter(
                        x=df_subset['working_set_size'],
                        y=df_subset['avg_ttft'],
                        mode='lines+markers',
                        name='Avg TTFT',
                        line=dict(color='orange', width=2),
                        marker=dict(size=6),
                        hovertemplate='Working Set: %{x:,} tokens<br>Avg TTFT: %{y:.3f}s<extra></extra>'
                    ),
                    row=3, col=1
                )

                fig.add_trace(
                    go.Scatter(
                        x=df_subset['working_set_size'],
                        y=df_subset['median_ttft'],
                        mode='lines+markers',
                        name='Median TTFT',
                        line=dict(color='darkorange', width=2, dash='dot'),
                        marker=dict(size=6),
                        hovertemplate='Working Set: %{x:,} tokens<br>Median TTFT: %{y:.3f}s<extra></extra>'
                    ),
                    row=3, col=1
                )

                fig.add_trace(
                    go.Scatter(
                        x=df_subset['working_set_size'],
                        y=df_subset['p95_ttft'],
                        mode='lines+markers',
                        name='P95 TTFT',
                        line=dict(color='red', width=2, dash='dot'),
                        marker=dict(size=6),
                        hovertemplate='Working Set: %{x:,} tokens<br>P95 TTFT: %{y:.3f}s<extra></extra>'
                    ),
                    row=3, col=1
                )

                fig.add_trace(
                    go.Scatter(
                        x=df_subset['working_set_size'],
                        y=df_subset['p99_ttft'],
                        mode='lines+markers',
                        name='P99 TTFT',
                        line=dict(color='darkred', width=2, dash='dash'),
                        marker=dict(size=6),
                        hovertemplate='Working Set: %{x:,} tokens<br>P99 TTFT: %{y:.3f}s<extra></extra>'
                    ),
                    row=3, col=1
                )

            # Update axes - use auto-scaling for working set size (tokens)
            fig.update_xaxes(
                title_text="Working Set Size (tokens)",
                row=1, col=1,
                gridcolor='lightgray',
                showgrid=True
            )
            fig.update_xaxes(
                title_text="Working Set Size (tokens)",
                row=2, col=1,
                gridcolor='lightgray',
                showgrid=True
            )
            fig.update_xaxes(
                title_text="Working Set Size (tokens)",
                row=3, col=1,
                gridcolor='lightgray',
                showgrid=True
            )

            fig.update_yaxes(title_text="Input Tokens/s", row=1, col=1)
            fig.update_yaxes(title_text="Output Tokens/s", row=2, col=1)
            fig.update_yaxes(title_text="TTFT (seconds)", row=3, col=1)

            # Update layout
            fig.update_layout(
                title=f"Performance vs Working Set Size<br>Context: {context_size:,} tokens | Cache Hit Rate: {cache_rate}%",
                height=900,
                showlegend=True,
                hovermode='x unified'
            )

            filename = output_path / f"performance_vs_working_set_{context_size}_cache{cache_rate}.html"
            fig.write_html(filename)
            logger.info(f"Generated graph: {filename}")

    # Graph 2: Output throughput comparison across working set sizes (per cache hit rate)
    # Use same per-retry calculation as performance vs working set graph
    # Generate even for single cache hit rate (user preference)
    if len(df['cache_hit_rate'].unique()) >= 1:
        for cache_rate in sorted(df['cache_hit_rate'].unique()):
            # Create figure with secondary y-axis
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            # Define colors for each context size (will be reused for both solid and dashed lines)
            colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']

            for idx, context_size in enumerate(sorted(df['context_size'].unique())):
                df_subset = df[(df['context_size'] == context_size) & (df['cache_hit_rate'] == cache_rate)].sort_values('working_set_size')

                if len(df_subset) == 0:
                    continue

                color = colors[idx % len(colors)]

                # Use per-retry mean if available, otherwise fall back to aggregated
                y_values = []
                ws_sizes = sorted(df_subset['working_set_size'].unique())
                for ws in ws_sizes:
                    if (context_size, ws, cache_rate) in all_detailed_variability:
                        # Use per-retry mean (same as performance graph)
                        y_values.append(all_detailed_variability[(context_size, ws, cache_rate)]['output_mean'])
                    else:
                        # Fallback to aggregated value
                        y_values.append(df_subset[df_subset['working_set_size'] == ws]['output_tokens_per_sec'].iloc[0])

                # Primary y-axis: Absolute values (solid line)
                fig.add_trace(
                    go.Scatter(
                        x=ws_sizes,
                        y=y_values,
                        mode='lines+markers',
                        name=f'{context_size:,} tokens',
                        line=dict(width=2, color=color),
                        marker=dict(size=6),
                        hovertemplate='%{fullData.name}<br>Working Set: %{x:,} tokens<br>Output: %{y:,.0f} tok/s<extra></extra>'
                    ),
                    secondary_y=False
                )

                # Secondary y-axis: Relative to baseline (dashed line, same color)
                baseline = y_values[0] if y_values else 1  # First value (smallest working set)
                relative_values = [y / baseline for y in y_values]

                fig.add_trace(
                    go.Scatter(
                        x=ws_sizes,
                        y=relative_values,
                        mode='lines+markers',
                        name=f'{context_size:,} tokens (speedup)',
                        line=dict(width=2, dash='dash', color=color),
                        marker=dict(size=6, symbol='diamond'),
                        hovertemplate='%{fullData.name}<br>Working Set: %{x:,} tokens<br>Speedup: %{y:.2f}x<extra></extra>'
                    ),
                    secondary_y=True
                )

            # Update axes
            fig.update_xaxes(
                title_text="Working Set Size (tokens)",
                gridcolor='lightgray',
                showgrid=True
            )
            fig.update_yaxes(title_text="Output Tokens/s", secondary_y=False)
            fig.update_yaxes(title_text="Speedup (relative to smallest working set)", secondary_y=True)

            fig.update_layout(
                title=f"Output Throughput Comparison Across Working Set Sizes<br>Cache Hit Rate: {cache_rate}%",
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

            filename = output_path / f"output_throughput_comparison_cache{cache_rate}.html"
            fig.write_html(filename)
            logger.info(f"Generated graph: {filename}")

    # Graph 3: Input throughput comparison across working set sizes (per cache hit rate)
    # Use same per-retry calculation as performance vs working set graph
    # Generate even for single cache hit rate (user preference)
    if len(df['cache_hit_rate'].unique()) >= 1:
        for cache_rate in sorted(df['cache_hit_rate'].unique()):
            # Create figure with secondary y-axis
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            # Define colors for each context size (will be reused for both solid and dashed lines)
            colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']

            for idx, context_size in enumerate(sorted(df['context_size'].unique())):
                df_subset = df[(df['context_size'] == context_size) & (df['cache_hit_rate'] == cache_rate)].sort_values('working_set_size')

                if len(df_subset) == 0:
                    continue

                color = colors[idx % len(colors)]

                # Use per-retry mean if available, otherwise fall back to aggregated
                y_values = []
                ws_sizes = sorted(df_subset['working_set_size'].unique())
                for ws in ws_sizes:
                    if (context_size, ws, cache_rate) in all_detailed_variability:
                        # Use per-retry mean (same as performance graph)
                        y_values.append(all_detailed_variability[(context_size, ws, cache_rate)]['input_mean'])
                    else:
                        # Fallback to aggregated value
                        y_values.append(df_subset[df_subset['working_set_size'] == ws]['input_tokens_per_sec'].iloc[0])

                # Primary y-axis: Absolute values (solid line)
                fig.add_trace(
                    go.Scatter(
                        x=ws_sizes,
                        y=y_values,
                        mode='lines+markers',
                        name=f'{context_size:,} tokens',
                        line=dict(width=2, color=color),
                        marker=dict(size=6),
                        hovertemplate='%{fullData.name}<br>Working Set: %{x:,} tokens<br>Input: %{y:,.0f} tok/s<extra></extra>'
                    ),
                    secondary_y=False
                )

                # Secondary y-axis: Relative to baseline (dashed line, same color)
                baseline = y_values[0] if y_values else 1  # First value (smallest working set)
                relative_values = [y / baseline for y in y_values]

                fig.add_trace(
                    go.Scatter(
                        x=ws_sizes,
                        y=relative_values,
                        mode='lines+markers',
                        name=f'{context_size:,} tokens (speedup)',
                        line=dict(width=2, dash='dash', color=color),
                        marker=dict(size=6, symbol='diamond'),
                        hovertemplate='%{fullData.name}<br>Working Set: %{x:,} tokens<br>Speedup: %{y:.2f}x<extra></extra>'
                    ),
                    secondary_y=True
                )

            # Update axes
            fig.update_xaxes(
                title_text="Working Set Size (tokens)",
                gridcolor='lightgray',
                showgrid=True
            )
            fig.update_yaxes(title_text="Input Tokens/s", secondary_y=False)
            fig.update_yaxes(title_text="Speedup (relative to smallest working set)", secondary_y=True)

            fig.update_layout(
                title=f"Input Throughput Comparison Across Working Set Sizes<br>Cache Hit Rate: {cache_rate}%",
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

            filename = output_path / f"input_throughput_comparison_cache{cache_rate}.html"
            fig.write_html(filename)
            logger.info(f"Generated graph: {filename}")

    # Graph 4: Output token metrics comparison (ITL and avg output tokens/s per request) across working set sizes
    # Generate even for single cache hit rate (user preference)
    if len(df['cache_hit_rate'].unique()) >= 1:
        for cache_rate in sorted(df['cache_hit_rate'].unique()):
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=(
                    'Average Inter-Token Latency (ITL) Across Working Set Sizes',
                    'Average Output Tokens/s per Request Across Working Set Sizes'
                ),
                vertical_spacing=0.15,
                specs=[[{"secondary_y": False}],
                       [{"secondary_y": False}]]
            )

            # Row 1: Inter-Token Latency (ms)
            for context_size in sorted(df['context_size'].unique()):
                df_subset = df[(df['context_size'] == context_size) & (df['cache_hit_rate'] == cache_rate)].sort_values('working_set_size')

                if len(df_subset) == 0:
                    continue

                fig.add_trace(
                    go.Scatter(
                        x=df_subset['working_set_size'],
                        y=df_subset['avg_itl'] * 1000,  # Convert to ms
                        mode='lines+markers',
                        name=f'{context_size:,} tokens',
                        line=dict(width=2),
                        hovertemplate='Working Set: %{x:,} tokens<br>ITL: %{y:.2f}ms<extra></extra>'
                    ),
                    row=1, col=1
                )

            # Row 2: Average output tokens/s per request
            for context_size in sorted(df['context_size'].unique()):
                df_subset = df[(df['context_size'] == context_size) & (df['cache_hit_rate'] == cache_rate)].sort_values('working_set_size')

                if len(df_subset) == 0:
                    continue

                fig.add_trace(
                    go.Scatter(
                        x=df_subset['working_set_size'],
                        y=df_subset['avg_output_tokens'],
                        mode='lines+markers',
                        name=f'{context_size:,} tokens',
                        line=dict(width=2),
                        showlegend=False,  # Already shown in row 1
                        hovertemplate='Working Set: %{x:,} tokens<br>Output: %{y:.1f} tok/s<extra></extra>'
                    ),
                    row=2, col=1
                )

            # Update axes
            fig.update_xaxes(
                title_text="Working Set Size (tokens)",
                row=1, col=1,
                gridcolor='lightgray',
                showgrid=True
            )
            fig.update_xaxes(
                title_text="Working Set Size (tokens)",
                row=2, col=1,
                gridcolor='lightgray',
                showgrid=True
            )
            fig.update_yaxes(title_text="Inter-Token Latency (ms)", row=1, col=1)
            fig.update_yaxes(title_text="Output Tokens/s per Request", row=2, col=1)

            # Update layout
            fig.update_layout(
                title=f"Output Token Metrics Comparison Across Working Set Sizes<br>Cache Hit Rate: {cache_rate}%",
                height=800,
                hovermode='x unified',
                showlegend=True
            )

            filename = output_path / f"output_metrics_comparison_cache{cache_rate}.html"
            fig.write_html(filename)
            logger.info(f"Generated graph: {filename}")

    # Graph 5: TTFT heatmap - REMOVED
    # This graph doesn't make sense for 3-dimensional data (context_size, working_set_size, cache_hit_rate)
    # Would need multiple heatmaps or 3D visualization which is less useful

    # Graph 6: Cache Hit Rate Comparison (one per context size)
    # Shows all cache hit rates as separate series vs working set size
    for context_size in df['context_size'].unique():
        df_ctx = df[df['context_size'] == context_size]

        # Create 4-row subplot for comprehensive metrics
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=(
                'Input Throughput vs Working Set Size',
                'Output Throughput vs Working Set Size',
                'Average TTFT vs Working Set Size',
                'Average Output Tokens per Request vs Working Set Size'
            ),
            vertical_spacing=0.08,
            specs=[[{"secondary_y": False}],
                   [{"secondary_y": False}],
                   [{"secondary_y": False}],
                   [{"secondary_y": False}]]
        )

        # Define colors for cache hit rates
        cache_colors = {
            0: '#EF553B',    # Red
            25: '#FFA15A',   # Orange
            50: '#AB63FA',   # Purple
            75: '#00CC96',   # Green
            90: '#636EFA',   # Blue
            100: '#19D3F3',  # Cyan
        }

        # Default color generator for other rates
        def get_color(rate):
            return cache_colors.get(rate, f'hsl({rate * 3.6}, 70%, 50%)')

        # For each cache hit rate, add a series
        for cache_rate in sorted(df_ctx['cache_hit_rate'].unique()):
            df_rate = df_ctx[df_ctx['cache_hit_rate'] == cache_rate].sort_values('working_set_size')
            color = get_color(cache_rate)

            # Row 1: Input throughput
            fig.add_trace(
                go.Scatter(
                    x=df_rate['working_set_size'],
                    y=df_rate['input_tokens_per_sec'],
                    mode='lines+markers',
                    name=f'{cache_rate}% cache',
                    line=dict(width=2, color=color),
                    marker=dict(size=6),
                    legendgroup=f'cache_{cache_rate}',
                    hovertemplate=f'Cache: {cache_rate}%<br>WS: %{{x:,}} tokens<br>Input: %{{y:,.0f}} tok/s<extra></extra>'
                ),
                row=1, col=1
            )

            # Row 2: Output throughput
            fig.add_trace(
                go.Scatter(
                    x=df_rate['working_set_size'],
                    y=df_rate['output_tokens_per_sec'],
                    mode='lines+markers',
                    name=f'{cache_rate}% cache',
                    line=dict(width=2, color=color),
                    marker=dict(size=6),
                    legendgroup=f'cache_{cache_rate}',
                    showlegend=False,
                    hovertemplate=f'Cache: {cache_rate}%<br>WS: %{{x:,}} tokens<br>Output: %{{y:,.0f}} tok/s<extra></extra>'
                ),
                row=2, col=1
            )

            # Row 3: TTFT
            fig.add_trace(
                go.Scatter(
                    x=df_rate['working_set_size'],
                    y=df_rate['avg_ttft'],
                    mode='lines+markers',
                    name=f'{cache_rate}% cache',
                    line=dict(width=2, color=color),
                    marker=dict(size=6),
                    legendgroup=f'cache_{cache_rate}',
                    showlegend=False,
                    hovertemplate=f'Cache: {cache_rate}%<br>WS: %{{x:,}} tokens<br>TTFT: %{{y:.3f}}s<extra></extra>'
                ),
                row=3, col=1
            )

            # Row 4: Avg output tokens per request
            fig.add_trace(
                go.Scatter(
                    x=df_rate['working_set_size'],
                    y=df_rate['avg_output_tokens'],
                    mode='lines+markers',
                    name=f'{cache_rate}% cache',
                    line=dict(width=2, color=color),
                    marker=dict(size=6),
                    legendgroup=f'cache_{cache_rate}',
                    showlegend=False,
                    hovertemplate=f'Cache: {cache_rate}%<br>WS: %{{x:,}} tokens<br>Output: %{{y:.1f}} tok/s<extra></extra>'
                ),
                row=4, col=1
            )

        # Update axes
        fig.update_xaxes(title_text="Working Set Size (tokens)", row=1, col=1)
        fig.update_xaxes(title_text="Working Set Size (tokens)", row=2, col=1)
        fig.update_xaxes(title_text="Working Set Size (tokens)", row=3, col=1)
        fig.update_xaxes(title_text="Working Set Size (tokens)", row=4, col=1)

        fig.update_yaxes(title_text="Input Tokens/s", row=1, col=1)
        fig.update_yaxes(title_text="Output Tokens/s", row=2, col=1)
        fig.update_yaxes(title_text="TTFT (seconds)", row=3, col=1)
        fig.update_yaxes(title_text="Output Tokens/s per Request", row=4, col=1)

        # Update layout
        cache_rates_str = ", ".join([f"{int(r)}%" for r in sorted(df_ctx['cache_hit_rate'].unique())])
        fig.update_layout(
            title=f"Cache Hit Rate Comparison Across Working Set Sizes<br>Context: {context_size:,} tokens | Cache Rates: {cache_rates_str}",
            height=1200,
            showlegend=True,
            hovermode='x unified'
        )

        filename = output_path / f"cache_comparison_ctx{context_size}.html"
        fig.write_html(filename)
        logger.info(f"Generated graph: {filename}")


async def main():
    """Main entry point"""
    args = parse_arguments()

    # Brief mode setup - suppress normal logging
    brief_mode = args.brief
    if brief_mode:
        logger.setLevel(logging.WARNING)  # Only show warnings/errors

    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Create test configuration
    config = create_test_config(args)

    logger.info(f"{Colors.HEADER}{'='*80}{Colors.ENDC}")
    logger.info(f"{Colors.HEADER}{Colors.BOLD}Working Set Size Performance Testing Tool{Colors.ENDC}")
    logger.info(f"{Colors.HEADER}{'='*80}{Colors.ENDC}")
    if len(config.api_endpoints) == 1:
        logger.info(f"{Colors.OKBLUE}API Endpoint: {config.api_endpoints[0]}{Colors.ENDC}")
    else:
        logger.info(f"{Colors.OKBLUE}API Endpoints ({len(config.api_endpoints)} for load balancing): {', '.join(config.api_endpoints)}{Colors.ENDC}")
    logger.info(f"{Colors.OKBLUE}Context Sizes: {config.context_sizes}{Colors.ENDC}")
    logger.info(f"{Colors.OKBLUE}Working Set Sizes: {len(config.working_set_sizes)} tests from {min(config.working_set_sizes):,} to {max(config.working_set_sizes):,} tokens{Colors.ENDC}")
    logger.info(f"{Colors.OKBLUE}Cache Hit Rates: {config.cache_hit_rates}{Colors.ENDC}")
    logger.info(f"{Colors.OKBLUE}Output Tokens: {config.output_tokens}{Colors.ENDC}")
    logger.info(f"{Colors.OKBLUE}Max TTFT: {config.max_ttft}s{Colors.ENDC}")
    logger.info(f"{Colors.OKBLUE}Test Duration: {config.test_duration}s (ramp: {config.ramp_duration}s per level){Colors.ENDC}")
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
    api_client = APIClient(config.api_endpoints, model="")
    model = await api_client.detect_model()
    api_client.model = model

    # Validate maximum working set size against KV cache capacity
    logger.info("")
    max_working_set = max(config.working_set_sizes)
    validate_working_set_size(model, max_working_set)
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
    for context_size in config.context_sizes:
        kv_info = calculate_kv_cache_size(context_size, config.kv_cache_bytes)
        logger.info(f"{Colors.OKBLUE}Context {context_size:>7,} tokens: {kv_info['total_mb']:>8.2f} MB ({kv_info['total_gb']:.3f} GB) per request{Colors.ENDC}")

        # Show working set cache range
        max_ws = max(config.working_set_sizes)
        min_ws = min(config.working_set_sizes)
        max_prompts = max(1, int(np.ceil(max_ws / context_size)))
        min_prompts = max(1, int(np.ceil(min_ws / context_size)))

        max_cache_mb = kv_info['total_mb'] * max_prompts
        min_cache_mb = kv_info['total_mb'] * min_prompts

        logger.info(f"{Colors.DEBUG}  → Working set range: {min_ws:,} to {max_ws:,} tokens{Colors.ENDC}")
        logger.info(f"{Colors.DEBUG}     Min: {min_prompts} prompts × {kv_info['total_mb']:.2f} MB = {min_cache_mb:.2f} MB ({min_cache_mb/1024:.3f} GB){Colors.ENDC}")
        logger.info(f"{Colors.DEBUG}     Max: {max_prompts} prompts × {kv_info['total_mb']:.2f} MB = {max_cache_mb:.2f} MB ({max_cache_mb/1024:.3f} GB){Colors.ENDC}")

    logger.info(f"{Colors.PHASE}{'='*80}{Colors.ENDC}")
    logger.info("")

    logger.info("Initialization complete. Ready to start testing.")
    logger.info("")

    # Load existing aggregated results if resuming
    all_aggregated_results = load_existing_aggregated_results(config.output_dir)
    if all_aggregated_results:
        logger.info(f"Loaded {len(all_aggregated_results)} existing test results from previous run")

        # Filter to ONLY include tests marked as completed in progress.json
        # This removes partial results from crashed/interrupted runs
        original_count = len(all_aggregated_results)
        all_aggregated_results = [
            m for m in all_aggregated_results
            if progress.is_test_completed(m.context_size, m.working_set_size, m.cache_hit_rate)
        ]

        removed_count = original_count - len(all_aggregated_results)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} partial/incomplete results (not marked completed in progress.json)")
            logger.info(f"Keeping {len(all_aggregated_results)} completed results")
        else:
            logger.info(f"All {len(all_aggregated_results)} loaded results are marked as completed")

    # Main test loop - iterate through context sizes, then cache rates, then working set sizes
    all_detailed_results = []

    for context_size in config.context_sizes:
        logger.info(f"{Colors.PHASE}{'='*80}{Colors.ENDC}")
        logger.info(f"{Colors.PHASE}{Colors.BOLD}Testing context size: {context_size:,} tokens{Colors.ENDC}")
        logger.info(f"{Colors.PHASE}{'='*80}{Colors.ENDC}")

        # Test each cache hit rate
        for cache_hit_rate in config.cache_hit_rates:
            logger.info("")
            logger.info(f"{Colors.HEADER}  Testing cache hit rate: {cache_hit_rate}%{Colors.ENDC}")
            logger.info(f"{Colors.HEADER}  {'='*70}{Colors.ENDC}")

            # Both modes: only test the max working_set_size (growth happens inside test)
            working_set_sizes_to_test = [config.working_set_sizes[-1]]

            # Test each working set size
            for working_set_size in working_set_sizes_to_test:
                # Check if already completed
                if progress.is_test_completed(context_size, working_set_size, cache_hit_rate):
                    logger.info(f"    Skipping working set size {working_set_size:,} (already completed)")
                    continue

                logger.info("")
                logger.info(f"{Colors.OKCYAN}    Testing working set size: {working_set_size:,} tokens{Colors.ENDC}")
                logger.info(f"{Colors.OKCYAN}    {'-'*60}{Colors.ENDC}")

                # Create working set for this size
                working_set = WorkingSet(context_size, working_set_size, tokenizer, config.chunk_size, config.seed)

                # Generate prompts based on init_strategy (applies to both modes)
                min_ws_size = config.working_set_sizes[0]  # Smallest working set size
                initial_num_prompts = max(1, int(np.ceil(min_ws_size / working_set.rounded_context_size)))
                mode_name = "Fixed" if config.mode == "fixed" else "Sustained"

                if config.init_strategy == "max":
                    # Generate ALL prompts upfront (up to max working set)
                    logger.info(f"{mode_name} mode (init=max): Generating ALL {working_set.num_prompts} prompts ({working_set_size:,} tokens)")
                    logger.info(f"  Will initialize all prompts, then add to rotation during growth")
                    working_set.generate_prompts()
                    # Set initial rotation to min working set size
                    working_set.num_prompts_in_rotation = initial_num_prompts
                    logger.info(f"  Starting with {initial_num_prompts} prompts in rotation (will grow during test)")
                else:  # "min"
                    # Generate only min working set prompts initially
                    logger.info(f"{mode_name} mode (init=min): Generating initial {initial_num_prompts} prompts ({min_ws_size:,} tokens)")
                    logger.info(f"  Will generate new prompts during growth (cache misses)")

                    # Generate only initial prompts
                    working_set.prompts = []
                    for i in range(initial_num_prompts):
                        prompt_seed = (working_set.seed + i) if working_set.seed is not None else None
                        tokens = tokenizer.generate_dummy_tokens(working_set.rounded_context_size, seed=prompt_seed, prompt_number=i)
                        working_set.prompts.append(tokens)
                    logger.info(f"  Generated {len(working_set.prompts)} initial prompts")

                # Initialize working set with API (pre-warm cache)
                # For sustained mode with init=min: only initial prompts
                # For sustained mode with init=max: all prompts
                if config.reinit_strategy == "once" or config.reinit_strategy == "per_working_set":
                    await initialize_working_set(api_client, working_set, 1, config.init_concurrency)  # Use 1 token for fast init

                # Run the test - both modes use run_continuous_mode (fixed mode just skips concurrency adjustment)
                try:
                    period_metrics = await run_continuous_mode(
                        config=config,
                        api_client=api_client,
                        working_set=working_set,
                        tokenizer=tokenizer,
                        context_size=context_size,
                        working_set_size=working_set_size,
                        cache_hit_rate=cache_hit_rate,
                        model=model
                    )

                    # Save results and generate graphs
                    if period_metrics:
                        csv_path = save_continuous_results(period_metrics, config.output_dir, context_size, working_set_size, cache_hit_rate)
                        if csv_path:
                            generate_sustained_mode_graphs(csv_path, config.output_dir, context_size, working_set_size, cache_hit_rate, config)
                            generate_sustained_index_html(csv_path, config.output_dir, context_size, working_set_size, cache_hit_rate, config)

                    # Mark as completed
                    progress.mark_test_completed(context_size, working_set_size, cache_hit_rate)

                except Exception as e:
                    logger.error(f"    ✗ Working set size {working_set_size:,} failed: {e}")
                    continue

            logger.info("")
            logger.info(f"{Colors.SUCCESS}  ✓ Cache hit rate {cache_hit_rate}% complete{Colors.ENDC}")

        logger.info("")
        logger.info(f"{Colors.SUCCESS}✓ Context size {context_size:,} complete{Colors.ENDC}")

        # Generate graphs after each context size completes
        if not config.skip_graphs and all_aggregated_results:
            logger.info("")
            logger.info(f"{Colors.PHASE}Generating visualizations for completed tests...{Colors.ENDC}")
            generate_graphs(all_aggregated_results, config.output_dir, config)

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
        generate_graphs(all_aggregated_results, config.output_dir, config)

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

    # Aggregate results from period CSV files for the summary (both modes use same format)
    sustained_aggregated_results = []
    period_files = list(Path(config.output_dir).glob("sustained_periods_*.csv"))
    if period_files:

        for period_file in period_files:
            try:
                df_periods = pd.read_csv(period_file)
                if len(df_periods) > 0:
                    # Get context_size and cache_hit_rate from CSV data
                    ctx_size = int(df_periods['context_size'].iloc[0])
                    cache_rate = int(df_periods['cache_hit_rate'].iloc[0])

                    # Get min and max working_set_size from the data (sustained mode grows over time)
                    min_ws_size = int(df_periods['working_set_size'].min())
                    max_ws_size = int(df_periods['working_set_size'].max())

                    # Create aggregated metrics from period data
                    aggregated = AggregatedMetrics(
                        context_size=ctx_size,
                        working_set_size=max_ws_size,
                        cache_hit_rate=cache_rate,
                        model=model,
                        input_tokens_per_sec=df_periods['input_tokens_per_sec'].mean(),
                        output_tokens_per_sec=df_periods['output_tokens_per_sec'].mean(),
                        avg_ttft=df_periods['avg_ttft'].mean(),
                        median_ttft=df_periods['median_ttft'].mean(),
                        p95_ttft=df_periods['p95_ttft'].mean(),
                        p99_ttft=df_periods['p99_ttft'].mean(),
                        avg_ttlt=df_periods['avg_ttlt'].mean(),
                        median_ttlt=df_periods['median_ttlt'].mean(),
                        p95_ttlt=df_periods['p95_ttlt'].mean(),
                        p99_ttlt=df_periods['p99_ttlt'].mean(),
                        avg_output_tokens=df_periods['avg_output_tokens_per_request'].mean(),
                        avg_itl=df_periods['avg_itl'].mean(),
                        median_itl=df_periods['median_itl'].mean(),
                        p95_itl=df_periods['p95_itl'].mean(),
                        p99_itl=df_periods['p99_itl'].mean(),
                        peak_concurrency=int(df_periods['concurrency_level'].max()),
                        total_requests=int(df_periods['num_requests_completed'].sum()),
                        test_duration=df_periods['duration'].sum(),
                        min_working_set_size=min_ws_size
                    )
                    sustained_aggregated_results.append(aggregated)
            except Exception as e:
                logger.warning(f"Failed to parse sustained period file {period_file}: {e}")

    # Combine fixed and sustained mode results for summary
    all_results_for_summary = all_aggregated_results + sustained_aggregated_results

    logger.info("")
    logger.info(f"{Colors.SUCCESS}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    logger.info(f"{Colors.SUCCESS}{Colors.BOLD}✓ All tests complete!{Colors.ENDC}")
    logger.info(f"{Colors.SUCCESS}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    logger.info(f"{Colors.OKGREEN}Results saved to: {config.output_dir}{Colors.ENDC}")
    logger.info(f"{Colors.OKGREEN}Total tests completed: {len(all_results_for_summary)}{Colors.ENDC}")

    # Print final summary table
    if all_results_for_summary:
        logger.info("")
        logger.info(f"{Colors.PHASE}{'='*130}{Colors.ENDC}")
        logger.info(f"{Colors.PHASE}{Colors.BOLD}Final Summary - All Test Results{Colors.ENDC}")
        logger.info(f"{Colors.PHASE}{'='*130}{Colors.ENDC}")

        # Header
        logger.info(f"{Colors.PHASE}{'Context':>10} {'WorkingSet':>12} {'Cache%':>8} {'Requests':>10} {'Input Tok':>12} {'Output Tok':>12} {'Input/s':>12} {'Output/s':>12} {'Avg TTFT':>10} {'Conc':>6}{Colors.ENDC}")
        logger.info(f"{Colors.PHASE}{'-'*130}{Colors.ENDC}")

        # Calculate grand totals
        grand_total_input = 0
        grand_total_output = 0
        grand_total_requests = 0

        # Sort by context size, then working set size, then cache rate
        sorted_results = sorted(all_results_for_summary, key=lambda x: (x.context_size, x.working_set_size, x.cache_hit_rate))

        for m in sorted_results:
            # Estimate total tokens from throughput and duration
            est_input_tokens = int(m.input_tokens_per_sec * m.test_duration)
            est_output_tokens = int(m.output_tokens_per_sec * m.test_duration)

            grand_total_input += est_input_tokens
            grand_total_output += est_output_tokens
            grand_total_requests += m.total_requests

            # Format working set size - show range for sustained mode (min-max)
            if m.min_working_set_size is not None and m.min_working_set_size != m.working_set_size:
                # Format as "minK-maxK" or "minM-maxM" for readability
                def format_ws(val):
                    if val >= 1_000_000:
                        return f"{val/1_000_000:.1f}M"
                    elif val >= 1_000:
                        return f"{val/1_000:.0f}K"
                    else:
                        return str(val)
                ws_display = f"{format_ws(m.min_working_set_size)}-{format_ws(m.working_set_size)}"
            else:
                ws_display = f"{m.working_set_size:,}"

            logger.info(f"{m.context_size:>10,} {ws_display:>12} {m.cache_hit_rate:>7}% {m.total_requests:>10,} "
                       f"{est_input_tokens/1e6:>11.2f}M {est_output_tokens/1e6:>11.2f}M "
                       f"{m.input_tokens_per_sec:>11,.0f} {m.output_tokens_per_sec:>11,.0f} "
                       f"{m.avg_ttft:>9.3f}s {m.peak_concurrency:>6}")

        # Grand totals
        logger.info(f"{Colors.PHASE}{'-'*130}{Colors.ENDC}")
        logger.info(f"{Colors.SUCCESS}{'TOTAL':>10} {'':>12} {'':>8} {grand_total_requests:>10,} "
                   f"{grand_total_input/1e6:>11.2f}M {grand_total_output/1e6:>11.2f}M{Colors.ENDC}")
        logger.info(f"{Colors.PHASE}{'='*130}{Colors.ENDC}")

    # Brief mode output
    if brief_mode and all_results_for_summary:
        print(f"model: {model}")
        print(f"endpoint: {config.api_endpoints[0]}")
        print(f"cache_rate: {config.cache_hit_rates[0]}")
        print()
        print("context_size,working_set,requests,input_tokens,output_tokens,input_tps,output_tps,avg_ttft,p95_ttft,concurrency")

        # Calculate totals for brief mode
        brief_total_requests = 0
        brief_total_input = 0
        brief_total_output = 0

        for m in sorted(all_results_for_summary, key=lambda x: (x.context_size, x.working_set_size)):
            est_input = int(m.input_tokens_per_sec * m.test_duration)
            est_output = int(m.output_tokens_per_sec * m.test_duration)
            brief_total_requests += m.total_requests
            brief_total_input += est_input
            brief_total_output += est_output
            print(f"{m.context_size},{m.working_set_size},{m.total_requests},{est_input},{est_output},{m.input_tokens_per_sec:.0f},{m.output_tokens_per_sec:.0f},{m.avg_ttft:.3f},{m.p95_ttft:.3f},{m.peak_concurrency}")

        print()
        print(f"total_requests: {brief_total_requests}")
        print(f"total_input_tokens: {brief_total_input}")
        print(f"total_output_tokens: {brief_total_output}")
        print()
        print(f"output: {config.output_dir}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user. Progress has been saved.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
