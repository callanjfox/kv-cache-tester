#!/usr/bin/env python3
"""
Trace Replay Performance Testing Tool

Replays real agentic coding traces to benchmark LLM inference performance
with realistic cache hit patterns, timing, and message structures.

Version: 1.0
Date: 2025-01-21
"""

__version__ = "1.0"
__date__ = "2025-01-21"

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import random
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Literal, Set
from collections import defaultdict, deque
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


# =============================================================================
# ANSI Colors for Terminal Output
# =============================================================================

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
    INFO = ''                # Default terminal color
    DEBUG = '\033[90m'
    METRIC = '\033[96m'
    SUCCESS = '\033[92m'
    PHASE = '\033[95m'
    USER = '\033[94m'

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
        cls.UNDERLINE = ''
        cls.INFO = ''
        cls.DEBUG = ''
        cls.METRIC = ''
        cls.SUCCESS = ''
        cls.PHASE = ''
        cls.USER = ''


# =============================================================================
# Question Bank - Prompts to encourage detailed responses
# =============================================================================

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


# =============================================================================
# Logging Setup
# =============================================================================

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""

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
    """Initialize logger with console handler"""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(ColoredFormatter())
        logger.addHandler(console_handler)

    return logger


logger = init_logger("trace_replay")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TraceStats:
    """Statistics about loaded traces"""
    total_traces: int
    filtered_traces: int
    total_requests: int
    avg_requests_per_trace: float
    avg_cache_hit_rate: float
    min_input_tokens: int
    max_input_tokens: int
    total_input_tokens: int
    traces_with_tool_use: int
    max_shared_prefix_tokens: int = 0  # Max(tool_tokens + system_tokens) across traces


@dataclass
class TestConfig:
    """Configuration for the trace replay test"""
    api_endpoint: str
    trace_directory: str
    output_dir: str
    max_context: int
    max_ttft: float
    ttft_metric: str  # max, avg, p95
    min_output_tokens_per_req: Optional[float]
    start_users: int
    max_users: int
    max_delay: float
    time_scale: float
    timing_strategy: str  # "original", "think-only", "api-scaled"
    api_time_scale: float  # multiplier for api_time with api-scaled strategy
    assessment_period: int
    test_duration: Optional[int]
    recycle: bool
    chunk_size: int
    verbose: bool
    tokenizer_id: str
    min_requests: int = 1
    max_new_tokens_per_period: int = 500000  # Cache pressure limit per period
    max_working_set_tokens: int = 0  # 0 = unlimited, else cap total working set
    seed: Optional[int] = None  # Random seed for reproducibility
    # Generation parameters (None = use model defaults or auto-detect)
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    # Rate limiting
    enable_request_rate_limiting: bool = False
    # Admission control
    max_concurrent_requests: Optional[int] = None  # None = unlimited
    # Warm prefix for cross-conversation cache sharing
    warm_prefix_pct: float = 0.5  # Default 50% of tool+system tokens warm
    # Trace advancement: start users partway through their traces
    advance_min: float = 0.0  # Minimum start position as fraction (0.0-1.0)
    advance_max: float = 0.0  # Maximum start position as fraction (0.0-1.0)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RequestMetrics:
    """Metrics for a single request"""
    user_id: str
    request_idx: int
    trace_id: str
    timestamp: float
    request_type: str  # streaming or non_streaming
    input_tokens: int
    output_tokens_expected: int
    output_tokens_actual: int
    cache_hit_blocks: int
    cache_miss_blocks: int
    ttft: float
    ttlt: float
    itl: float
    delay_expected: float
    delay_actual: float
    success: bool
    error_message: Optional[str] = None
    # Timestamps for period attribution
    request_start_time: float = 0.0  # When request was sent
    prefill_complete_time: float = 0.0  # When first token received (TTFT)
    request_complete_time: float = 0.0  # When last token received
    # Token chunk timing for proportional output attribution
    token_timestamps: List[float] = field(default_factory=list)
    tokens_per_chunk: List[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class UserLifecycleEvent:
    """Event tracking user lifecycle"""
    timestamp: float
    user_id: str
    event_type: str  # started, completed, truncated, idle, active
    trace_id: str
    details: str = ""


@dataclass
class AssessmentPeriodMetrics:
    """Metrics for an assessment period"""
    period_number: int
    start_time: float
    end_time: float
    active_users: int  # Users with in-flight requests at assessment time
    idle_users: int    # Users idle at assessment time
    users_with_requests: int  # Users who sent at least 1 request this period
    requests_completed: int  # Total requests completed this period
    requests_launched: int  # Requests started this period
    requests_completed_new: int  # Requests started AND completed this period
    requests_completed_prior: int  # Requests started prior, completed this period
    requests_in_progress: int  # Requests still in flight at assessment time
    requests_in_progress_new: int  # In-flight requests started this period
    requests_in_progress_prior: int  # In-flight requests started prior periods
    requests_per_second: float
    input_tokens_per_second: float
    output_tokens_per_second: float
    ttft_avg: float
    ttft_p50: float
    ttft_p95: float
    ttft_p99: float
    avg_cache_hit_rate: float
    working_set_blocks: int
    users_added: int
    users_completed: int
    total_request_time: float  # Sum of all request durations (ttlt)
    idle_time_pct: float  # Percentage of period users were idle
    new_tokens_ingested: int = 0  # Cache miss tokens this period
    ttft_headroom_pct: float = 0.0  # How much below threshold (0-100%)
    rate_limited_users: int = 0  # Users in rate_limited state at assessment time
    rate_limit_events: int = 0  # Total rate-limit events this period
    # Admission control metrics
    admission_blocked_events: int = 0  # Times dispatch was blocked this period
    dispatch_delay_avg: float = 0.0  # Avg seconds behind schedule
    dispatch_delay_max: float = 0.0  # Max seconds behind schedule
    in_flight_prefilling: int = 0  # Requests awaiting/in prefill at assessment time
    in_flight_decoding: int = 0  # Requests in decode phase at assessment time


# =============================================================================
# Trace Normalization (new format support)
# =============================================================================

def normalize_request(req: dict, base_time: float) -> dict:
    """Normalize a single request from new trace format to internal format."""
    result = {
        'timestamp': base_time + req.get('t', 0.0),
        'type': 'streaming',  # Always streaming (ignore s/n distinction)
        'input_tokens': req.get('in', 0),
        'output_tokens': req.get('out', 0),
        'hash_ids': req.get('hash_ids', []),
        'input_types': req.get('input_types', []),
        'output_types': req.get('output_types', []),
        'stop_reason': req.get('stop', ''),
        'model': req.get('model', ''),
    }
    # Timing breakdown (optional, from newer traces)
    if 'api_time' in req:
        result['api_time'] = req['api_time']
    if 'think_time' in req:
        result['think_time'] = req['think_time']
    return result


def flatten_requests(requests: list, base_time: float) -> list:
    """Flatten requests including subagents into a single timeline.

    Subagents have type='subagent' with their own nested requests array.
    Their timestamps are relative to the subagent start time.
    """
    result = []
    for req in requests:
        if req.get('type') == 'subagent':
            # Recursively flatten subagent requests with adjusted timestamps
            subagent_requests = flatten_requests(
                req.get('requests', []),
                base_time=base_time + req.get('t', 0.0)
            )
            result.extend(subagent_requests)
        else:
            result.append(normalize_request(req, base_time))

    # Sort by absolute timestamp (subagent requests may interleave with parent)
    result.sort(key=lambda r: r['timestamp'])
    return result


def normalize_trace(trace: dict) -> dict:
    """Convert new trace format to internal normalized format.

    New format has compact field names (t, in, out) and subagent support.
    Sub-agent entries (type='subagent') are preserved as markers in the
    request list rather than flattened. The orchestrator spawns separate
    UserSessions for sub-agents when encountered during replay.
    """
    raw_requests = trace.get('requests', [])
    requests = []
    for req in raw_requests:
        if req.get('type') == 'subagent':
            # Preserve sub-agent entry as a marker (not flattened)
            requests.append(req)
        else:
            requests.append(normalize_request(req, base_time=0.0))

    # Compute stats from parent requests only (exclude sub-agent markers)
    parent_requests = [r for r in requests if r.get('type') != 'subagent']
    total_input_tokens = sum(r['input_tokens'] for r in parent_requests)

    # Compute cache hit rate from hash_ids (parent requests only)
    cache_hits = 0
    total_blocks = 0
    for i, req in enumerate(parent_requests):
        hash_ids = req.get('hash_ids', [])
        if i > 0 and hash_ids:
            prev_hash_ids = set(parent_requests[i-1].get('hash_ids', []))
            for h in hash_ids:
                total_blocks += 1
                if h in prev_hash_ids:
                    cache_hits += 1
                else:
                    break
        elif hash_ids:
            total_blocks += len(hash_ids)

    cache_hit_rate = cache_hits / total_blocks if total_blocks > 0 else 0.0

    normalized = {
        'metadata': {
            'conversation_id': trace.get('id', 'unknown'),
            'models': trace.get('models', []),
            'block_size': trace.get('block_size', 64),
            'hash_id_scope': trace.get('hash_id_scope', 'per_context'),
            'tool_tokens': trace.get('tool_tokens', 0),
            'system_tokens': trace.get('system_tokens', 0),
            'request_count': len(parent_requests),
            'total_input_tokens': total_input_tokens,
            'cache_hit_rate': cache_hit_rate,
        },
        'requests': requests,
    }
    return normalized


# =============================================================================
# Trace Manager
# =============================================================================

class TraceManager:
    """Manages loading and sampling of conversation traces"""

    def __init__(self, trace_dir: Path, max_context: int, min_requests: int = 1, seed: Optional[int] = None):
        self.trace_dir = trace_dir
        self.max_context = max_context
        self.min_requests = min_requests
        self.seed = seed
        self.traces: List[dict] = []
        self.used_trace_ids: Set[str] = set()
        self.stats: Optional[TraceStats] = None

        # Deterministic selection: fixed order + counter (unaffected by recycle timing)
        self.trace_order: List[dict] = []  # Fixed shuffled order for selection
        self.available_ids: Set[str] = set()  # Currently available trace IDs
        self.next_idx: int = 0  # Counter for round-robin selection

        # Use dedicated RNG for deterministic behavior (isolated from global state)
        self.rng = random.Random(seed) if seed is not None else random.Random()

    def load_traces(self) -> int:
        """Load all traces from directory and filter by max_context"""
        logger.info(f"Loading traces from {self.trace_dir}...")

        all_traces = []
        trace_files = list(self.trace_dir.glob("*.json"))

        if not trace_files:
            raise ValueError(f"No trace files found in {self.trace_dir}")

        for filepath in trace_files:
            try:
                with open(filepath) as f:
                    trace = json.load(f)
                    trace = normalize_trace(trace)  # Convert new format to internal format
                    trace['_filepath'] = str(filepath)
                    all_traces.append(trace)
            except Exception as e:
                logger.warning(f"Failed to load {filepath}: {e}")

        # Filter by max context and min requests
        self.traces = []
        for trace in all_traces:
            if trace['requests']:
                first_input = trace['requests'][0]['input_tokens']
                num_requests = len(trace['requests'])
                # Allow if first request fits and has enough requests
                if first_input <= self.max_context and num_requests >= self.min_requests:
                    self.traces.append(trace)

        # Sort traces by conversation_id for reproducibility, then shuffle with dedicated RNG
        self.traces.sort(key=lambda t: t['metadata']['conversation_id'])
        self.trace_order = list(self.traces)
        self.rng.shuffle(self.trace_order)
        # All traces start as available
        self.available_ids = {t['metadata']['conversation_id'] for t in self.traces}

        # Compute statistics
        self._compute_stats(len(all_traces))

        logger.info(f"Loaded {len(self.traces)} traces (filtered from {len(all_traces)})")
        return len(self.traces)

    def _compute_stats(self, total_before_filter: int):
        """Compute aggregate statistics"""
        if not self.traces:
            self.stats = TraceStats(
                total_traces=total_before_filter,
                filtered_traces=0,
                total_requests=0,
                avg_requests_per_trace=0,
                avg_cache_hit_rate=0,
                min_input_tokens=0,
                max_input_tokens=0,
                total_input_tokens=0,
                traces_with_tool_use=0,
                max_shared_prefix_tokens=0
            )
            return

        total_requests = sum(t['metadata']['request_count'] for t in self.traces)
        cache_rates = [t['metadata'].get('cache_hit_rate', 0) for t in self.traces]
        input_tokens = [t['metadata']['total_input_tokens'] for t in self.traces]

        # Check for tool_use
        tool_use_count = 0
        for trace in self.traces:
            for req in trace['requests']:
                if 'tool_use' in req.get('stop_reason', ''):
                    tool_use_count += 1
                    break

        # Calculate max shared prefix (tool_tokens + system_tokens) across all traces
        max_shared_prefix = 0
        for trace in self.traces:
            tool_tokens = trace['metadata'].get('tool_tokens', 0)
            system_tokens = trace['metadata'].get('system_tokens', 0)
            max_shared_prefix = max(max_shared_prefix, tool_tokens + system_tokens)

        self.stats = TraceStats(
            total_traces=total_before_filter,
            filtered_traces=len(self.traces),
            total_requests=total_requests,
            avg_requests_per_trace=total_requests / len(self.traces),
            avg_cache_hit_rate=np.mean(cache_rates) if cache_rates else 0,
            min_input_tokens=min(input_tokens) if input_tokens else 0,
            max_input_tokens=max(input_tokens) if input_tokens else 0,
            total_input_tokens=sum(input_tokens),
            traces_with_tool_use=tool_use_count,
            max_shared_prefix_tokens=max_shared_prefix
        )

    def get_random_trace(self) -> Optional[dict]:
        """Get next trace from deterministic order (round-robin through available)"""
        if not self.available_ids:
            return None

        # Walk through fixed order to find next available trace
        start_idx = self.next_idx
        num_traces = len(self.trace_order)

        for _ in range(num_traces):
            trace = self.trace_order[self.next_idx % num_traces]
            trace_id = trace['metadata']['conversation_id']
            self.next_idx += 1

            if trace_id in self.available_ids:
                self.available_ids.remove(trace_id)
                self.used_trace_ids.add(trace_id)
                return trace

        return None  # No available traces

    def return_trace(self, trace: dict):
        """Return trace to pool for recycling"""
        trace_id = trace['metadata']['conversation_id']
        self.available_ids.add(trace_id)
        # No shuffling needed - selection order is fixed by trace_order

    def get_stats(self) -> TraceStats:
        """Return computed statistics"""
        return self.stats


# =============================================================================
# Synthetic Message Generator
# =============================================================================

class SyntheticMessageGenerator:
    """Generates realistic synthetic content for different message types.

    Only generates USER messages - assistant messages come from actual model responses.
    Two distinct content pools:
    - User text: Natural language prompts, questions, requests
    - Tool results: File contents, bash output, paths, errors
    """

    def __init__(self, tokenizer_id: str, chunk_size: int = 64):
        self.tokenizer = None
        self.tokenizer_id = tokenizer_id
        self.chunk_size = chunk_size

        # Separate content pools for different message types
        self._user_text_pool_tokens: Optional[List[int]] = None
        self._tool_result_pool_tokens: Optional[List[int]] = None
        self._pool_size = 500_000  # 500K tokens per pool

        # Canonical warm prefix (shared across all users for cache sharing)
        self._canonical_prefix_content: Optional[str] = None
        self._canonical_prefix_tokens: int = 0

        # Initialize vocabulary
        self._init_vocabulary()

    def _init_vocabulary(self):
        """Initialize vocabulary for different message types"""

        # =================================================================
        # USER TEXT VOCABULARY - Natural language coding prompts/questions
        # =================================================================

        # Prompt starters and templates
        self.user_prompt_starters = [
            "Can you help me", "I need to", "Please", "How can I", "I'm trying to",
            "Could you", "I want to", "What's the best way to", "I'm getting an error",
            "Can you explain", "Help me understand", "I'm working on", "I need help with",
            "Can I add", "Where do I find", "How do I", "Is there a way to",
            "Can you fix", "Please review", "I'm having trouble with", "What should I do",
        ]

        # Action verbs for coding tasks
        self.user_action_verbs = [
            "implement", "debug", "fix", "refactor", "optimize", "add", "remove", "update",
            "create", "build", "deploy", "test", "review", "explain", "understand", "find",
            "search", "copy", "move", "rename", "delete", "modify", "change", "improve",
            "configure", "setup", "install", "run", "execute", "check", "verify", "validate",
        ]

        # Technical nouns for coding context
        self.user_tech_nouns = [
            "function", "class", "method", "variable", "parameter", "argument", "module",
            "package", "dependency", "import", "export", "component", "service", "handler",
            "controller", "model", "view", "template", "configuration", "setting", "option",
            "feature", "bug", "error", "exception", "issue", "problem", "test", "spec",
            "file", "directory", "path", "endpoint", "route", "API", "database", "query",
            "schema", "migration", "index", "cache", "session", "token", "authentication",
            "authorization", "permission", "user", "request", "response", "middleware",
        ]

        # Connecting words and phrases
        self.user_connectors = [
            "the", "a", "an", "this", "that", "these", "those", "my", "our", "your",
            "in", "on", "at", "to", "from", "with", "for", "of", "by", "about",
            "and", "or", "but", "so", "because", "when", "where", "how", "what", "why",
            "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
            "do", "does", "did", "will", "would", "could", "should", "can", "may", "might",
        ]

        # Question endings
        self.user_question_endings = [
            "?", "work?", "do this?", "help?", "possible?", "correct?", "right?",
            "better?", "faster?", "easier?", "needed?", "required?", "working?",
        ]

        # System reminder templates (injected by Claude Code)
        self.system_reminder_templates = [
            "<system-reminder>\nThis is a reminder that your todo list is currently empty.</system-reminder>",
            "<system-reminder>\nThe TodoWrite tool hasn't been used recently.</system-reminder>",
            "<system-reminder>\nConsider using the Task tool for complex operations.</system-reminder>",
        ]

        # =================================================================
        # TOOL RESULT VOCABULARY - File contents, bash output, paths, errors
        # =================================================================

        # File paths and directories
        self.tool_directories = [
            "/home/user/project", "/mnt/weka", "/app", "/workspace", "/repo",
            "/var/log", "/etc", "/usr/local", "/opt", "~/.config", "~/.claude",
            "/src", "/lib", "/pkg", "/cmd", "/internal", "/api", "/tests",
        ]

        self.tool_file_extensions = [
            ".py", ".js", ".ts", ".tsx", ".go", ".rs", ".java", ".cpp", ".c", ".h",
            ".md", ".json", ".yaml", ".yml", ".toml", ".xml", ".html", ".css",
            ".sh", ".bash", ".zsh", ".sql", ".graphql", ".proto", ".env",
        ]

        # Code keywords (for file content generation)
        self.tool_code_keywords = [
            "import", "from", "def", "class", "function", "const", "let", "var",
            "return", "if", "else", "elif", "for", "while", "try", "except", "catch",
            "finally", "with", "async", "await", "yield", "raise", "throw", "new",
            "public", "private", "protected", "static", "final", "override", "virtual",
            "interface", "struct", "enum", "type", "extends", "implements", "super",
            "self", "this", "None", "null", "undefined", "true", "false", "True", "False",
        ]

        # Code symbols and operators
        self.tool_code_symbols = [
            "(", ")", "{", "}", "[", "]", "<", ">", ":", ";", ",", ".", "=", "==",
            "!=", "<=", ">=", "+=", "-=", "*=", "/=", "=>", "->", "::", "...", "@",
            "#", "//", "/*", "*/", '"""', "'''", "`", "$", "&", "|", "^", "~", "!",
        ]

        # Bash command prefixes and common output
        self.tool_bash_prefixes = [
            "$ ", ">>> ", "# ", "> ", "% ", "user@host:~$ ",
        ]

        self.tool_bash_commands = [
            "ls", "cd", "cat", "grep", "find", "mkdir", "rm", "cp", "mv", "chmod",
            "git status", "git diff", "git log", "git branch", "npm install", "npm run",
            "pip install", "python", "node", "cargo", "go build", "make", "docker",
        ]

        self.tool_bash_output_words = [
            "total", "drwxr-xr-x", "-rw-r--r--", "user", "group", "bytes", "modified",
            "created", "directory", "file", "symlink", "permission", "denied", "found",
            "installed", "updated", "removed", "copied", "moved", "completed", "success",
            "error", "warning", "failed", "not found", "no such file", "already exists",
        ]

        # Error message components
        self.tool_error_types = [
            "Error", "Exception", "TypeError", "ValueError", "KeyError", "IndexError",
            "AttributeError", "ImportError", "ModuleNotFoundError", "FileNotFoundError",
            "ConnectionError", "TimeoutError", "PermissionError", "SyntaxError",
            "RuntimeError", "AssertionError", "NotImplementedError", "OSError",
        ]

        self.tool_error_messages = [
            "File has not been read yet", "No files found", "Permission denied",
            "Connection refused", "Timeout exceeded", "Invalid argument", "Not found",
            "Already exists", "Cannot be empty", "Must be a valid", "Expected",
            "Unexpected token", "Undefined variable", "Type mismatch", "Out of range",
        ]

        # JSON structure words
        self.tool_json_keys = [
            "id", "name", "type", "value", "data", "result", "status", "message",
            "error", "code", "path", "file", "line", "column", "start", "end",
            "input", "output", "config", "options", "settings", "metadata", "content",
        ]

    def load_tokenizer(self):
        """Lazy load tokenizer"""
        if self.tokenizer is None:
            logger.info(f"Loading tokenizer: {self.tokenizer_id}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_id,
                trust_remote_code=True
            )
            if hasattr(self.tokenizer, 'model_max_length'):
                self.tokenizer.model_max_length = 1_000_000

    def _ensure_user_text_pool(self):
        """Pre-generate content pool for user text messages."""
        if self._user_text_pool_tokens is not None:
            return

        self.load_tokenizer()
        logger.info(f"Pre-generating user text pool ({self._pool_size:,} tokens)...")

        np.random.seed(42)
        chunks = []
        estimated_tokens = 0
        target_tokens = self._pool_size + 100_000

        while estimated_tokens < target_tokens:
            # Generate natural language prompts
            content_type = np.random.choice(['prompt', 'question', 'mixed'], p=[0.4, 0.3, 0.3])

            if content_type == 'prompt':
                # "Can you help me implement a function that handles the authentication..."
                lines = []
                for _ in range(10):
                    starter = np.random.choice(self.user_prompt_starters)
                    verb = np.random.choice(self.user_action_verbs)
                    nouns = np.random.choice(self.user_tech_nouns, size=3)
                    connectors = np.random.choice(self.user_connectors, size=4)
                    line = f"{starter} {verb} {connectors[0]} {nouns[0]} {connectors[1]} {nouns[1]} {connectors[2]} {connectors[3]} {nouns[2]}"
                    lines.append(line)
                chunk = ". ".join(lines)

            elif content_type == 'question':
                # Generate question-style content
                lines = []
                for _ in range(10):
                    starter = np.random.choice(["How do I", "What is the", "Where can I find", "Why does", "Can you explain"])
                    nouns = np.random.choice(self.user_tech_nouns, size=2)
                    connectors = np.random.choice(self.user_connectors, size=2)
                    ending = np.random.choice(self.user_question_endings)
                    line = f"{starter} {nouns[0]} {connectors[0]} {connectors[1]} {nouns[1]} {ending}"
                    lines.append(line)
                chunk = " ".join(lines)

            else:  # mixed
                # Mix of technical description and natural language
                words = list(np.random.choice(self.user_tech_nouns, size=20)) + \
                        list(np.random.choice(self.user_connectors, size=30)) + \
                        list(np.random.choice(self.user_action_verbs, size=10))
                np.random.shuffle(words)
                chunk = " ".join(words)

            chunks.append(chunk)
            estimated_tokens += len(chunk.split()) * 1.3

        full_text = "\n\n".join(chunks)
        self._user_text_pool_tokens = self.tokenizer.encode(full_text, add_special_tokens=False)
        logger.info(f"User text pool ready: {len(self._user_text_pool_tokens):,} tokens")

    def _ensure_tool_result_pool(self):
        """Pre-generate content pool for tool result messages."""
        if self._tool_result_pool_tokens is not None:
            return

        self.load_tokenizer()
        logger.info(f"Pre-generating tool result pool ({self._pool_size:,} tokens)...")

        np.random.seed(43)  # Different seed for different content
        chunks = []
        estimated_tokens = 0
        target_tokens = self._pool_size + 100_000

        while estimated_tokens < target_tokens:
            content_type = np.random.choice(['file', 'bash', 'json', 'error', 'path'],
                                           p=[0.35, 0.25, 0.15, 0.1, 0.15])

            if content_type == 'file':
                # File contents with line numbers (like Read tool output)
                lines = []
                for i in range(1, np.random.randint(50, 150)):
                    indent = "    " * np.random.randint(0, 4)
                    keywords = np.random.choice(self.tool_code_keywords, size=np.random.randint(2, 6))
                    symbols = np.random.choice(self.tool_code_symbols, size=np.random.randint(1, 4))
                    content = indent + " ".join(list(keywords) + list(symbols))
                    lines.append(f"     {i}→{content}")
                chunk = "\n".join(lines)

            elif content_type == 'bash':
                # Bash command output
                lines = []
                prefix = np.random.choice(self.tool_bash_prefixes)
                cmd = np.random.choice(self.tool_bash_commands)
                lines.append(f"{prefix}{cmd}")
                for _ in range(np.random.randint(5, 30)):
                    words = np.random.choice(self.tool_bash_output_words, size=np.random.randint(3, 8))
                    lines.append(" ".join(words))
                chunk = "\n".join(lines)

            elif content_type == 'json':
                # JSON-like structured output
                lines = ["{"]
                for _ in range(np.random.randint(5, 20)):
                    key = np.random.choice(self.tool_json_keys)
                    value_type = np.random.choice(['string', 'number', 'bool'])
                    if value_type == 'string':
                        value = f'"{np.random.choice(self.tool_bash_output_words)}"'
                    elif value_type == 'number':
                        value = str(np.random.randint(0, 10000))
                    else:
                        value = np.random.choice(['true', 'false'])
                    lines.append(f'  "{key}": {value},')
                lines.append("}")
                chunk = "\n".join(lines)

            elif content_type == 'error':
                # Error messages
                error_type = np.random.choice(self.tool_error_types)
                error_msg = np.random.choice(self.tool_error_messages)
                traceback_lines = []
                for i in range(np.random.randint(3, 8)):
                    file_path = np.random.choice(self.tool_directories) + np.random.choice(self.tool_file_extensions)
                    line_no = np.random.randint(1, 500)
                    traceback_lines.append(f'  File "{file_path}", line {line_no}')
                chunk = f"Traceback (most recent call last):\n" + "\n".join(traceback_lines) + f"\n{error_type}: {error_msg}"

            else:  # path
                # File path listings
                lines = []
                for _ in range(np.random.randint(5, 30)):
                    path = np.random.choice(self.tool_directories)
                    name = "".join(np.random.choice(list("abcdefghijklmnopqrstuvwxyz_"), size=np.random.randint(5, 15)))
                    ext = np.random.choice(self.tool_file_extensions)
                    lines.append(f"{path}/{name}{ext}")
                chunk = "\n".join(lines)

            chunks.append(chunk)
            estimated_tokens += len(chunk.split()) * 1.3

        full_text = "\n\n".join(chunks)
        self._tool_result_pool_tokens = self.tokenizer.encode(full_text, add_special_tokens=False)
        logger.info(f"Tool result pool ready: {len(self._tool_result_pool_tokens):,} tokens")

    def _get_from_pool(self, pool_tokens: List[int], num_tokens: int, seed: int) -> str:
        """Get content from a token pool using seed for offset."""
        np.random.seed(seed)
        max_offset = max(0, len(pool_tokens) - num_tokens - 1)
        offset = np.random.randint(0, max_offset) if max_offset > 0 else 0
        tokens = pool_tokens[offset:offset + num_tokens]
        return self.tokenizer.decode(tokens)

    def generate_user_text(self, num_tokens: int, seed: int) -> str:
        """Generate natural language user prompt content."""
        self._ensure_user_text_pool()
        return self._get_from_pool(self._user_text_pool_tokens, num_tokens, seed)

    def generate_tool_result(self, num_tokens: int, seed: int) -> str:
        """Generate tool execution output (file contents, bash output, etc.)."""
        self._ensure_tool_result_pool()
        return self._get_from_pool(self._tool_result_pool_tokens, num_tokens, seed)

    def build_user_message(self, num_tokens: int, msg_type: str, seed: int) -> dict:
        """Build a single user message of the specified type.

        Args:
            num_tokens: Target token count for the message
            msg_type: Either 'text' or 'tool_result'
            seed: Random seed for deterministic generation

        Returns:
            OpenAI-format message dict with role='user'
        """
        # Select question from bank (use seed to rotate deterministically)
        question = QUESTION_BANK[seed % len(QUESTION_BANK)]

        # Calculate tokens needed for question (including separator)
        self.load_tokenizer()
        question_with_sep = "\n\n" + question
        question_tokens = len(self.tokenizer.encode(question_with_sep, add_special_tokens=False))

        # Subtract question tokens from target to stay within token budget
        content_tokens = max(0, num_tokens - question_tokens)

        if msg_type == 'tool_result' or msg_type == 'tool_results':
            content = self.generate_tool_result(content_tokens, seed) if content_tokens > 0 else ""
        else:
            content = self.generate_user_text(content_tokens, seed) if content_tokens > 0 else ""

        # Append question to encourage long, detailed responses
        content = content + question_with_sep

        return {"role": "user", "content": content}

    def generate_canonical_prefix(self, num_tokens: int) -> str:
        """Generate the canonical shared prefix content (deterministic, no user salt).

        This prefix is shared by ALL users to enable cross-conversation cache hits.
        Uses a fixed seed to ensure the same content is generated every time.

        Args:
            num_tokens: Number of tokens for the canonical prefix

        Returns:
            String content that will be used as the shared prefix for all users
        """
        if self._canonical_prefix_content is not None:
            return self._canonical_prefix_content

        if num_tokens <= 0:
            return ""

        # Use fixed seed for reproducibility (no user_id salt)
        seed = hash("canonical_warm_prefix_v1") % (2**32)
        self._canonical_prefix_content = self.generate_user_text(num_tokens, seed)
        self._canonical_prefix_tokens = num_tokens
        logger.info(f"{Colors.OKCYAN}Generated canonical warm prefix: {num_tokens:,} tokens{Colors.ENDC}")
        return self._canonical_prefix_content

    def get_canonical_prefix_tokens(self) -> int:
        """Return the token count of the canonical prefix."""
        return self._canonical_prefix_tokens


# =============================================================================
# Trace Advancement Helpers
# =============================================================================

def calculate_start_index(requests: list, rng: random.Random,
                          min_pct: float, max_pct: float,
                          max_context: int = 0) -> int:
    """Calculate starting request index based on advancement range.

    Args:
        requests: List of trace requests
        rng: Random number generator for deterministic selection
        min_pct: Minimum position as fraction (0.0-1.0)
        max_pct: Maximum position as fraction (0.0-1.0)
        max_context: If >0, clamp to last index where input_tokens <= max_context

    Returns:
        Index into requests list where user should start
    """
    if max_pct <= 0 or len(requests) <= 1:
        return 0

    min_idx = int(len(requests) * min_pct)
    max_idx = min(int(len(requests) * max_pct), len(requests) - 1)

    # Clamp max_idx so we don't start beyond max_context
    if max_context > 0:
        while max_idx > min_idx and requests[max_idx].get('input_tokens', 0) > max_context:
            max_idx -= 1
        # If even min_idx exceeds context, return 0 (start from beginning)
        if requests[min_idx].get('input_tokens', 0) > max_context:
            return 0

    if min_idx >= max_idx:
        return min_idx

    return rng.randint(min_idx, max_idx)


def adjust_for_request_pairs(requests: list, start_idx: int) -> int:
    """Adjust index to avoid starting at second part of a request pair.

    Request pairs (streaming + non-streaming) have identical hash_ids.
    If we land on the second request, back up to the first.
    """
    if start_idx <= 0:
        return start_idx

    current = set(requests[start_idx].get('hash_ids', []))
    prev = set(requests[start_idx - 1].get('hash_ids', []))

    if current == prev and len(current) > 0:
        return start_idx - 1
    return start_idx


def skip_subagent_markers(requests: list, start_idx: int) -> int:
    """Skip past any subagent markers at start position."""
    while start_idx < len(requests) and requests[start_idx].get('type') == 'subagent':
        start_idx += 1
    return start_idx


def get_first_real_request(requests: list) -> Optional[dict]:
    """Get the first non-subagent request from a list of requests.

    Subagent markers have type='subagent' and 0 input_tokens.
    This helper skips them to find the first actual request.
    """
    for req in requests:
        if req.get('type') != 'subagent':
            return req
    return None


# =============================================================================
# User Session
# =============================================================================

class UserSession:
    """Represents a user stepping through a conversation trace.

    Key design:
    - Each request in the trace has a target `input_tokens` count
    - We track accumulated tokens (user content + actual assistant responses)
    - For each request, we calculate delta = target - accumulated
    - Generate new user content for the delta, append to conversation
    - Use actual model responses (not synthetic) for assistant turns

    The trace's hash_ids tell us about cache behavior (prefix reuse vs replacement),
    but we don't need to reconstruct individual messages - just hit the token targets.
    """

    def __init__(self, user_id: str, trace: dict, generator: SyntheticMessageGenerator, max_context: int):
        self.user_id = user_id
        self.trace = trace
        self.generator = generator
        self.max_context = max_context

        self.trace_id = trace['metadata']['conversation_id']
        self.requests = trace['requests']
        self.current_idx = 0
        self.state: Literal["active", "idle", "completed", "truncated", "rate_limited"] = "idle"

        self.start_time: Optional[float] = None
        self.last_request_time: Optional[float] = None
        self.metrics: List[RequestMetrics] = []

        # Track previous hash_ids for cache hit calculation
        self.prev_hash_ids: Set[int] = set()

        # Conversation history: list of {"role": "user"|"assistant", "content": str}
        self.conversation: List[dict] = []

        # Track hash_ids from previous request for cache behavior
        self.prev_request_hash_ids: Set[int] = set()

        # Track input_tokens for accurate token generation
        self.prev_input_tokens: int = 0
        self.stored_response_tokens: int = 0  # Tokens from model response we stored

        # Track token shortfall when model generates less than expected
        # This gets added to the next user message to maintain token counts
        self.token_shortfall: int = 0

        # Rate-limiting state tracking
        self.rate_limit_until: Optional[float] = None  # Timestamp when rate-limit expires
        self.rate_limit_count: int = 0  # Times rate-limited for current request attempt
        self.total_rate_limit_count: int = 0  # Total times rate-limited across all requests

        # Sub-agent tracking
        self.pending_subagents: List[str] = []      # user_ids of active sub-agent sessions
        self.parent_user_id: Optional[str] = None   # set if this is a sub-agent
        self.is_subagent: bool = False
        self._subagent_counter: int = 0

    def get_total_requests(self) -> int:
        return len(self.requests)

    def get_completed_requests(self) -> int:
        return self.current_idx

    def get_next_request(self) -> Optional[dict]:
        """Get next request to process, or None if done/truncated.

        Returns sub-agent entries as-is (type='subagent') — the orchestrator
        checks the type field and spawns a separate UserSession instead of
        sending an API request.
        """
        if self.current_idx >= len(self.requests):
            self.state = "completed"
            return None

        request = self.requests[self.current_idx]

        # Sub-agent markers don't have input_tokens — return as-is for orchestrator
        if request.get('type') == 'subagent':
            return request

        # Check context limit
        if request['input_tokens'] > self.max_context:
            self.state = "truncated"
            return None

        return request

    def get_delay_until_next(self) -> float:
        """Get delay in seconds until next request"""
        if self.current_idx == 0:
            return 0.0

        if self.current_idx >= len(self.requests):
            return 0.0

        curr = self.requests[self.current_idx]
        prev = self.requests[self.current_idx - 1]

        # Sub-agent markers use 't' field, normalized requests use 'timestamp'
        curr_ts = curr.get('timestamp', curr.get('t', 0.0))
        prev_ts = prev.get('timestamp', prev.get('t', 0.0))

        return max(0.0, curr_ts - prev_ts)

    def store_assistant_response(self, response_text: str, response_tokens: int, request: dict):
        """Store the actual assistant response.

        For request pairs (streaming + non-streaming with same hash_ids),
        only store after the non-streaming request to avoid double-counting.
        Also tracks stored_response_tokens for accurate token generation in next request.
        """
        current_hash_ids = set(request.get('hash_ids', []))

        # Check if next request is a pair (same hash_ids)
        next_idx = self.current_idx + 1
        if next_idx < len(self.requests):
            next_req = self.requests[next_idx]
            next_hash_ids = set(next_req.get('hash_ids', []))
            if current_hash_ids == next_hash_ids and next_req.get('type') == 'non_streaming':
                # This is the streaming part of a pair - don't store yet
                return

        # Store the actual response and track its token count
        self.conversation.append({"role": "assistant", "content": response_text})
        self.stored_response_tokens = response_tokens

    def advance(self):
        """Move to next request"""
        if self.current_idx < len(self.requests):
            # Update prev_hash_ids for cache tracking
            request = self.requests[self.current_idx]
            current_hash_ids = set(request.get('hash_ids', []))
            self.prev_hash_ids = current_hash_ids
            self.current_idx += 1
            self.last_request_time = time.time()

    def reconstruct_state_at_index(self, start_idx: int, generator_seed: int):
        """Reconstruct session state to start at given index.

        Sets up state as if requests 0..start_idx-1 had been processed.
        This enables trace advancement without replaying all previous requests.
        """
        if start_idx <= 0:
            return

        prev_request = self.requests[start_idx - 1]

        # Set hash_id state from previous request for cache tracking
        self.prev_hash_ids = set(prev_request.get('hash_ids', []))
        self.prev_request_hash_ids = set(prev_request.get('hash_ids', []))
        self.prev_input_tokens = prev_request.get('input_tokens', 0)

        # Reset response tracking (no accumulated tokens from skipped requests)
        self.stored_response_tokens = 0
        self.token_shortfall = 0

        # Set position
        self.current_idx = start_idx

        # Build synthetic conversation matching the starting request's input_tokens
        current_request = self.requests[start_idx]
        target_tokens = current_request.get('input_tokens', 0)
        content = self.generator.generate_user_text(target_tokens, generator_seed)
        self.conversation = [{"role": "user", "content": content}]

    def calculate_cache_hits(self, request: dict) -> Tuple[int, int]:
        """Calculate cache hits and misses for this request based on hash_ids."""
        current_hash_ids = set(request.get('hash_ids', []))

        if not self.prev_hash_ids:
            # First request - all misses
            return 0, len(current_hash_ids)

        hits = len(current_hash_ids & self.prev_hash_ids)
        misses = len(current_hash_ids) - hits

        return hits, misses

    def _get_user_message_type(self, request: dict) -> str:
        """Determine the type of user message to generate based on trace.

        Uses input_types field from normalized trace format.
        """
        input_types = request.get('input_types', [])
        if 'tool_result' in input_types:
            return 'tool_result'
        return 'text'

    def build_messages(self, request: dict, canonical_prefix: str = None,
                        canonical_prefix_tokens: int = 0) -> Tuple[List[dict], int]:
        """Build messages for this request.

        Uses hash_ids and input_tokens together:
        - Same hash_ids as previous → re-send same content (request pair)
        - Many blocks removed (>10% pull-back) → reset to kept boundary, regenerate
        - Few blocks removed (normal boundary) → append new content

        If canonical_prefix is provided and this is the first request, uses the
        shared prefix content for cross-conversation cache sharing.

        The hash_ids tell us about cache behavior and chunk boundaries.
        The input_tokens tells us the exact token count to achieve.

        Args:
            request: Request dict with input_tokens, output_tokens, hash_ids
            canonical_prefix: Optional shared prefix content for warm prefix
            canonical_prefix_tokens: Token count of the canonical prefix

        Returns:
            Tuple of (messages list, max_tokens for this request)
        """
        max_tokens = max(1, request.get('output_tokens', 100))
        current_hash_ids = set(request.get('hash_ids', []))
        current_input_tokens = request.get('input_tokens', 0)

        # For first request with warm prefix enabled - use canonical shared prefix
        if self.current_idx == 0 and canonical_prefix and canonical_prefix_tokens > 0:
            # Calculate how much of the request should use canonical prefix
            prefix_tokens = min(canonical_prefix_tokens, current_input_tokens)

            # Remaining part: user-specific content
            remaining_tokens = max(0, current_input_tokens - prefix_tokens)

            if remaining_tokens > 0:
                # Generate user-specific content for the remainder
                seed = hash(f"{self.user_id}_{self.current_idx}_remainder_{remaining_tokens}") % (2**32)
                msg_type = self._get_user_message_type(request)
                user_content = self.generator.generate_user_text(remaining_tokens, seed)

                # Combine: canonical prefix + user content
                combined_content = canonical_prefix + "\n\n" + user_content
            else:
                # First request fits entirely within canonical prefix
                combined_content = canonical_prefix

            self.conversation.append({"role": "user", "content": combined_content})

            # Update for next request
            self.prev_request_hash_ids = current_hash_ids
            self.prev_input_tokens = current_input_tokens

            return list(self.conversation), max_tokens

        # Check if this is a request pair (identical hash_ids to previous)
        if current_hash_ids == self.prev_request_hash_ids and len(current_hash_ids) > 0:
            # Same content - re-send same conversation (cache hit test)
            return list(self.conversation), max_tokens

        # Analyze hash_id changes
        new_hash_ids = current_hash_ids - self.prev_request_hash_ids
        removed_hash_ids = self.prev_request_hash_ids - current_hash_ids
        kept_hash_ids = current_hash_ids & self.prev_request_hash_ids

        # Check for pull-back: significant blocks removed (>10% of previous)
        is_pullback = (len(self.prev_request_hash_ids) > 0 and
                       len(removed_hash_ids) > len(self.prev_request_hash_ids) * 0.1)

        if is_pullback:
            # Pull-back case: reset to kept boundary and regenerate
            # Get block_size from trace metadata (default 64 for backwards compat)
            block_size = self.trace['metadata'].get('block_size', 64)
            kept_tokens = len(kept_hash_ids) * block_size
            # Generate: input_tokens - kept_tokens (plus any shortfall)
            tokens_to_generate = max(0, current_input_tokens - kept_tokens + self.token_shortfall)
            self.token_shortfall = 0
            self.stored_response_tokens = 0

            # Reset conversation and regenerate
            self.conversation = []
            if tokens_to_generate > 0:
                seed = hash(f"{self.user_id}_{self.current_idx}_pullback_{tokens_to_generate}") % (2**32)
                msg_type = self._get_user_message_type(request)
                new_user_msg = self.generator.build_user_message(tokens_to_generate, msg_type, seed)
                self.conversation.append(new_user_msg)

        elif len(new_hash_ids) > 0 or len(removed_hash_ids) > 0:
            # Normal growth: few blocks removed (boundary replacement), new blocks added
            # Use token delta for more accurate token counts
            token_delta = current_input_tokens - self.prev_input_tokens
            tokens_to_generate = max(0, token_delta - self.stored_response_tokens + self.token_shortfall)
            self.token_shortfall = 0
            self.stored_response_tokens = 0  # Reset after using

            if tokens_to_generate > 0:
                seed = hash(f"{self.user_id}_{self.current_idx}_{tokens_to_generate}") % (2**32)
                msg_type = self._get_user_message_type(request)
                new_user_msg = self.generator.build_user_message(tokens_to_generate, msg_type, seed)
                self.conversation.append(new_user_msg)

        # Update for next request
        self.prev_request_hash_ids = current_hash_ids
        self.prev_input_tokens = current_input_tokens

        # Safety: ensure conversation is never empty
        if not self.conversation:
            logger.warning(f"{self.user_id} req {self.current_idx}: Empty conversation after build, generating minimal message")
            seed = hash(f"{self.user_id}_{self.current_idx}_fallback") % (2**32)
            self.conversation.append(self.generator.build_user_message(max(100, current_input_tokens), 'text', seed))

        return list(self.conversation), max_tokens

    def record_shortfall(self, expected_tokens: int, actual_tokens: int):
        """Record token shortfall if model generated less than 80% of expected."""
        if expected_tokens > 0 and actual_tokens < expected_tokens * 0.8:
            self.token_shortfall = expected_tokens - actual_tokens

    def get_summary(self) -> dict:
        """Get summary statistics for this session"""
        total_cache_hits = sum(m.cache_hit_blocks for m in self.metrics)
        total_cache_misses = sum(m.cache_miss_blocks for m in self.metrics)
        total_blocks = total_cache_hits + total_cache_misses

        return {
            "user_id": self.user_id,
            "trace_id": self.trace_id,
            "state": self.state,
            "requests_completed": self.current_idx,
            "requests_total": len(self.requests),
            "avg_cache_hit_rate": total_cache_hits / total_blocks if total_blocks > 0 else 0,
            "avg_ttft": np.mean([m.ttft for m in self.metrics]) if self.metrics else 0,
        }


# =============================================================================
# API Client
# =============================================================================

class APIClient:
    """Manages OpenAI API client"""

    # Model-specific default generation parameters
    MODEL_DEFAULTS = {
        "qwen3-coder": {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "repetition_penalty": 1.05,
        },
    }

    def __init__(self, api_endpoint: str, model: str = "",
                 temperature: Optional[float] = None,
                 top_p: Optional[float] = None,
                 top_k: Optional[int] = None,
                 repetition_penalty: Optional[float] = None):
        self.api_endpoint = api_endpoint
        self.model = model

        # Store user-specified overrides (None means use model defaults or auto-detect)
        self._user_temperature = temperature
        self._user_top_p = top_p
        self._user_top_k = top_k
        self._user_repetition_penalty = repetition_penalty

        # Actual parameters to use (set after model detection)
        self.temperature: Optional[float] = temperature
        self.top_p: Optional[float] = top_p
        self.top_k: Optional[int] = top_k
        self.repetition_penalty: Optional[float] = repetition_penalty

        # Ensure base_url ends with /v1
        base_url = api_endpoint.rstrip('/')
        if not base_url.endswith('/v1'):
            base_url = base_url + '/v1'

        self.client = openai.AsyncOpenAI(
            api_key="EMPTY",
            base_url=base_url
        )

        logger.info(f"API Client initialized: {api_endpoint}")

    def _apply_model_defaults(self):
        """Apply model-specific default parameters if not overridden by user."""
        model_lower = self.model.lower()

        # Check for matching model patterns
        matched_defaults = None
        matched_pattern = None
        for pattern, defaults in self.MODEL_DEFAULTS.items():
            if pattern in model_lower:
                matched_defaults = defaults
                matched_pattern = pattern
                break

        if matched_defaults:
            applied_settings = []
            if self._user_temperature is None and "temperature" in matched_defaults:
                self.temperature = matched_defaults["temperature"]
                applied_settings.append(f"temperature={self.temperature}")
            if self._user_top_p is None and "top_p" in matched_defaults:
                self.top_p = matched_defaults["top_p"]
                applied_settings.append(f"top_p={self.top_p}")
            if self._user_top_k is None and "top_k" in matched_defaults:
                self.top_k = matched_defaults["top_k"]
                applied_settings.append(f"top_k={self.top_k}")
            if self._user_repetition_penalty is None and "repetition_penalty" in matched_defaults:
                self.repetition_penalty = matched_defaults["repetition_penalty"]
                applied_settings.append(f"repetition_penalty={self.repetition_penalty}")

            if applied_settings:
                logger.info(f"{Colors.OKCYAN}Detected {matched_pattern} model - applying settings: {', '.join(applied_settings)}{Colors.ENDC}")

    async def detect_model(self) -> str:
        """Auto-detect model from API and apply model-specific settings."""
        try:
            models_url = self.api_endpoint.rstrip('/') + '/v1/models'
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(models_url, timeout=10) as response:
                    data = await response.json()
                    if 'data' in data and len(data['data']) > 0:
                        self.model = data['data'][0]['id']
                        logger.info(f"Auto-detected model: {self.model}")
                        # Apply model-specific defaults
                        self._apply_model_defaults()
                        return self.model
        except Exception as e:
            logger.warning(f"Could not auto-detect model: {e}")

        return self.model

    def _build_request_params(self, messages: List[dict], max_tokens: int, stream: bool) -> dict:
        """Build request parameters including model-specific settings."""
        params = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        # Add generation parameters (vLLM supports all of these directly)
        extra_body = {}
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.top_k is not None:
            extra_body["top_k"] = self.top_k
        if self.repetition_penalty is not None:
            extra_body["repetition_penalty"] = self.repetition_penalty

        if extra_body:
            params["extra_body"] = extra_body

        return params

    async def send_request(self, messages: List[dict], max_tokens: int, stream: bool = True,
                           on_first_token: Optional[callable] = None) -> dict:
        """
        Send request and return metrics.

        Args:
            messages: Chat messages to send
            max_tokens: Maximum tokens to generate
            stream: Whether to use streaming (default True)
            on_first_token: Optional callback invoked when first token arrives (prefill complete)

        Returns dict with: response_text, ttft, ttlt, actual_output_tokens, error_type,
                          start_time, first_token_time, complete_time (absolute timestamps),
                          token_timestamps, tokens_per_chunk (for proportional attribution)
        """
        start_time = time.time()
        first_token_time = None
        response_text = ""
        token_count = 0
        token_timestamps: List[float] = []
        tokens_per_chunk: List[int] = []

        try:
            params = self._build_request_params(messages, max_tokens, stream)

            if stream:
                response = await self.client.chat.completions.create(**params)

                async for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        chunk_time = time.time()
                        if first_token_time is None:
                            first_token_time = chunk_time
                            # Signal that prefill is complete (first token received)
                            if on_first_token:
                                on_first_token()
                        response_text += chunk.choices[0].delta.content
                        token_count += 1  # Approximate: 1 token per chunk
                        # Track timestamp and token count for this chunk
                        token_timestamps.append(chunk_time)
                        tokens_per_chunk.append(1)

                complete_time = time.time()

            else:
                response = await self.client.chat.completions.create(**params)

                first_token_time = time.time()
                complete_time = first_token_time

                if response.choices:
                    response_text = response.choices[0].message.content or ""
                    token_count = response.usage.completion_tokens if response.usage else len(response_text.split())
                    # For non-streaming, all tokens are attributed to completion time
                    token_timestamps.append(complete_time)
                    tokens_per_chunk.append(token_count)

            ttft = (first_token_time - start_time) if first_token_time else 0
            ttlt = complete_time - start_time

            return {
                'response_text': response_text,
                'ttft': ttft,
                'ttlt': ttlt,
                'output_tokens': token_count,
                'error_type': None,
                'start_time': start_time,
                'first_token_time': first_token_time or start_time,
                'complete_time': complete_time,
                'token_timestamps': token_timestamps,
                'tokens_per_chunk': tokens_per_chunk
            }

        except Exception as e:
            error_str = str(e).lower()
            if "connection" in error_str or "connect" in error_str or "refused" in error_str:
                error_type = "connection"
                logger.error(f"Connection error: {e}")
            else:
                error_type = "other"
                logger.error(f"Request failed: {e}")
            return {
                'response_text': "",
                'ttft': 0,
                'ttlt': 0,
                'output_tokens': 0,
                'error_type': error_type,
                'start_time': start_time,
                'first_token_time': start_time,
                'complete_time': time.time(),
                'token_timestamps': [],
                'tokens_per_chunk': []
            }


# =============================================================================
# Test Orchestrator
# =============================================================================

class TestOrchestrator:
    """Orchestrates the trace replay test"""

    def __init__(self, config: TestConfig, trace_manager: TraceManager,
                 generator: SyntheticMessageGenerator, api_client: APIClient):
        self.config = config
        self.trace_manager = trace_manager
        self.generator = generator
        self.api_client = api_client

        self.users: Dict[str, UserSession] = {}
        self.user_counter = 0
        self.lifecycle_events: List[UserLifecycleEvent] = []
        self.all_metrics: List[RequestMetrics] = []
        self.assessment_periods: List[AssessmentPeriodMetrics] = []

        self.test_start_time: Optional[float] = None
        self.current_period_start: Optional[float] = None
        self.period_metrics: List[RequestMetrics] = []
        self.period_new_tokens: int = 0  # Cache miss tokens this period
        self.period_users_added: int = 0  # Users added this period
        self.period_rate_limit_events: int = 0  # Rate-limit events this period
        self.peak_working_set_tokens: int = 0  # Track peak working set
        self.peak_users: int = 0  # Track peak concurrent users

        # Block-level working set tracking with timestamps for time-based aging
        # Keys are (trace_id, hash_id) tuples to make hash_ids unique per conversation
        self.block_last_access: Dict[Tuple[str, int], float] = {}
        # Time-ordered list of (timestamp, block_key) for efficient pruning
        self.block_access_order: deque = deque()

        # Admission control
        self.in_flight_requests: int = 0  # Current count of requests in flight
        self.in_flight_decoding: int = 0  # Requests that have received first token (in decode phase)
        self.period_admission_blocked: int = 0  # Times dispatch was blocked this period
        self.period_dispatch_delays: List[float] = []  # Dispatch delays this period

        # Cooldown tracking for ramp controller
        self.consecutive_exceeded_periods: int = 0   # How many consecutive periods exceeded TTFT
        self.consecutive_good_periods: int = 0       # How many consecutive periods under TTFT
        self.cooldown_required: int = 0              # Good periods required before next ramp
        self.post_cooldown_ramps: int = 0            # Ramps since last cooldown (for throttling)

        # Error tracking
        self.consecutive_connection_errors: int = 0
        self.total_connection_errors: int = 0
        self.max_consecutive_errors: int = 10  # Fail test after this many consecutive errors

        # Warm prefix for cross-conversation cache sharing
        self.canonical_prefix_content: str = ""
        self.canonical_prefix_tokens: int = 0

        self.running = True

    def create_user(self, enforce_budgets: bool = True) -> Optional[UserSession]:
        """Create a new user from available traces.

        Args:
            enforce_budgets: If True, check per-period and working set budgets.
                            Set to False for initial users (warning only).
        """
        trace = self.trace_manager.get_random_trace()
        if trace is None:
            return None

        if enforce_budgets:
            # Try up to max_attempts traces to find one that fits budget
            max_attempts = 5
            for attempt in range(max_attempts):
                # Check global working set budget
                can_add_ws, ws_tokens = self.can_add_user_for_working_set(trace)
                if not can_add_ws:
                    current_ws = self.get_current_working_set_tokens()
                    available = self.config.max_working_set_tokens - current_ws
                    logger.info(f"  ⚠️ Working set limit: trace needs ~{ws_tokens:,} tokens, "
                               f"only {available:,} available")
                    self.trace_manager.return_trace(trace)
                    trace = self.trace_manager.get_random_trace()
                    if trace is None:
                        return None
                    continue

                # Check per-period budget
                can_add_period, period_tokens = self.can_add_user_for_period_budget(trace)
                if not can_add_period:
                    available = self.config.max_new_tokens_per_period - self.period_new_tokens
                    logger.info(f"  ⚠️ Period budget: trace needs ~{period_tokens:,} tokens, "
                               f"only {available:,} available")
                    self.trace_manager.return_trace(trace)
                    trace = self.trace_manager.get_random_trace()
                    if trace is None:
                        return None
                    continue

                # Found a suitable trace - update budget and break
                self.period_new_tokens += period_tokens
                break
            else:
                # Exhausted attempts
                logger.warning(f"Could not find trace fitting budget after {max_attempts} attempts")
                if trace:
                    self.trace_manager.return_trace(trace)
                return None

        self.user_counter += 1
        user_id = f"User-{self.user_counter:03d}"

        user = UserSession(user_id, trace, self.generator, self.config.max_context)
        user.start_time = time.time()

        # Apply trace advancement if configured
        advancement_pct = 0.0
        if self.config.advance_max > 0:
            start_idx = calculate_start_index(
                trace['requests'], self.trace_manager.rng,
                self.config.advance_min, self.config.advance_max,
                self.config.max_context)
            start_idx = adjust_for_request_pairs(trace['requests'], start_idx)
            start_idx = skip_subagent_markers(trace['requests'], start_idx)

            if start_idx > 0 and start_idx < len(trace['requests']):
                seed = hash(f"{user_id}_advanced_{start_idx}") % (2**32)
                user.reconstruct_state_at_index(start_idx, seed)
                advancement_pct = (start_idx / len(trace['requests'])) * 100

        self.users[user_id] = user

        if advancement_pct > 0:
            remaining_reqs = len(trace['requests']) - user.current_idx
            current_req = user.requests[user.current_idx] if user.current_idx < len(user.requests) else None
            current_tokens = current_req['input_tokens'] if current_req else 0
            self.log_lifecycle_event(user_id, "started", trace['metadata']['conversation_id'],
                                     f"{remaining_reqs} requests remaining (advanced {advancement_pct:.0f}%), "
                                     f"{current_tokens:,} tokens at start")
            logger.info(f"{Colors.USER}  👤 {user_id} started{Colors.ENDC} (trace: {user.trace_id[:10]}, "
                       f"📍 advanced to req {user.current_idx} ({advancement_pct:.0f}%), "
                       f"{remaining_reqs} remaining, {current_tokens:,} tokens)")
        else:
            initial_tokens = trace['requests'][0]['input_tokens']
            self.log_lifecycle_event(user_id, "started", trace['metadata']['conversation_id'],
                                     f"{len(trace['requests'])} requests, {initial_tokens:,} initial tokens")
            logger.info(f"{Colors.USER}  👤 {user_id} started{Colors.ENDC} (trace: {user.trace_id[:10]}, {len(trace['requests'])} requests, {initial_tokens:,} initial tokens)")

        return user

    async def create_users_batch(self, count: int, delay_ms: int = 50,
                                  enforce_budgets: bool = True) -> List[UserSession]:
        """Create multiple users with a small delay between each to avoid overwhelming the server.

        Args:
            count: Number of users to create
            delay_ms: Delay between user creation in milliseconds
            enforce_budgets: If True, check per-period and working set budgets
        """
        users = []
        for i in range(count):
            user = self.create_user(enforce_budgets=enforce_budgets)
            if user:
                users.append(user)
                # Add delay between users (not after the last one)
                if i < count - 1:
                    await asyncio.sleep(delay_ms / 1000.0)
            else:
                break  # No more traces available or budget exhausted
        return users

    def remove_user(self, user_id: str, reason: str):
        """Remove a user and optionally recycle their trace"""
        if user_id not in self.users:
            return

        user = self.users[user_id]
        summary = user.get_summary()

        self.log_lifecycle_event(user_id, reason, user.trace_id,
                                 f"{summary['requests_completed']}/{summary['requests_total']} requests, {summary['avg_cache_hit_rate']:.1%} cache hit")

        if reason == "completed":
            logger.info(f"{Colors.SUCCESS}  ✓ {user_id} completed{Colors.ENDC} ({summary['requests_completed']}/{summary['requests_total']} requests, {summary['avg_cache_hit_rate']:.1%} cache hit)")
        else:
            logger.warning(f"  ⚠️ {user_id} truncated at request {summary['requests_completed']}/{summary['requests_total']} (context exceeded {self.config.max_context})")

        # Recycle trace if enabled
        if self.config.recycle:
            self.trace_manager.return_trace(user.trace)

        del self.users[user_id]

    def _compute_delay(self, user: UserSession) -> float:
        """Compute the delay before the user's next request fires.

        Supports timing strategies:
        - original: use trace timestamp differences (default, backward compatible)
        - think-only: use only client think_time (simulates instant server)
        - api-scaled: use prev api_time * scale + think_time (simulates faster/slower server)
        """
        strategy = self.config.timing_strategy
        idx = user.current_idx

        if strategy != "original" and idx > 0 and idx < len(user.requests):
            curr = user.requests[idx]
            prev = user.requests[idx - 1]

            think_time = curr.get('think_time')
            prev_api_time = prev.get('api_time')

            if think_time is not None:
                if strategy == "think-only":
                    delay = think_time
                elif strategy == "api-scaled" and prev_api_time is not None:
                    delay = prev_api_time * self.config.api_time_scale + think_time
                else:
                    # Fallback for api-scaled without api_time data
                    delay = user.get_delay_until_next()
                return min(delay, self.config.max_delay) * self.config.time_scale

        # Default: original behavior
        delay = user.get_delay_until_next()
        return min(delay, self.config.max_delay) * self.config.time_scale

    def spawn_subagent(self, parent: UserSession, subagent_entry: dict):
        """Spawn a sub-agent as an independent UserSession.

        Called when the parent's get_next_request() returns a type='subagent' entry.
        The sub-agent runs concurrently with the parent as a separate user.
        """
        if not subagent_entry.get('requests'):
            return  # Skip empty/failed sub-agents

        parent._subagent_counter += 1
        sa_id = f"{parent.user_id}-SA{parent._subagent_counter}"

        # Build a mini-trace for the sub-agent
        sa_trace = normalize_trace({
            'id': f"{parent.trace_id}:{subagent_entry.get('agent_id', 'unknown')}",
            'models': subagent_entry.get('models', []),
            'block_size': parent.trace['metadata'].get('block_size', 64),
            'hash_id_scope': parent.trace['metadata'].get('hash_id_scope', 'per_context'),
            'tool_tokens': subagent_entry.get('tool_tokens', 0),
            'system_tokens': subagent_entry.get('system_tokens', 0),
            'requests': subagent_entry['requests'],
        })

        sa_user = UserSession(sa_id, sa_trace, parent.generator, parent.max_context)
        sa_user.is_subagent = True
        sa_user.parent_user_id = parent.user_id
        sa_user.start_time = time.time()

        self.users[sa_id] = sa_user
        parent.pending_subagents.append(sa_id)

        sa_reqs = len([r for r in sa_trace['requests'] if r.get('type') != 'subagent'])
        logger.info(f"{Colors.HEADER}  🔀 {sa_id} spawned from {parent.user_id} "
                     f"({sa_reqs} requests, {subagent_entry.get('subagent_type', 'unknown')}){Colors.ENDC}")

    def log_lifecycle_event(self, user_id: str, event_type: str, trace_id: str, details: str = ""):
        """Log a user lifecycle event"""
        event = UserLifecycleEvent(
            timestamp=time.time(),
            user_id=user_id,
            event_type=event_type,
            trace_id=trace_id,
            details=details
        )
        self.lifecycle_events.append(event)

    def get_user_counts(self) -> Tuple[int, int]:
        """Return (active_count, idle_count)"""
        active = sum(1 for u in self.users.values() if u.state == "active")
        idle = sum(1 for u in self.users.values() if u.state == "idle")
        return active, idle

    def get_ttft_value(self) -> Optional[float]:
        """Get the current TTFT value based on configured metric. Returns None if no data."""
        if not self.period_metrics:
            return None  # No data available

        ttfts = [m.ttft for m in self.period_metrics if m.success]
        if not ttfts:
            return None  # No successful requests

        if self.config.ttft_metric == 'max':
            return max(ttfts)
        elif self.config.ttft_metric == 'avg':
            return np.mean(ttfts)
        else:  # p95
            return np.percentile(ttfts, 95)

    def calculate_users_to_add(self) -> int:
        """
        Calculate how many users to add based on TTFT headroom with cooldown gating.

        Uses consecutive good/bad period tracking to prevent ramping after brief
        recovery from sustained overload. Budget enforcement (per-period and working
        set) happens in create_user() on a per-trace basis.

        Returns number of users to add (0 if gated by any condition).
        """
        ttft_value = self.get_ttft_value()

        # No data = don't add users (system may be overloaded or starting up)
        if ttft_value is None:
            return 0

        # Over TTFT threshold = don't add
        if ttft_value >= self.config.max_ttft:
            return 0

        # Gate: in-flight > 75% of max concurrent
        if (self.config.max_concurrent_requests and
            self.in_flight_requests > self.config.max_concurrent_requests * 0.75):
            logger.info(f"{Colors.WARNING}  \u23f8\ufe0f Skipping user addition: in-flight at {self.in_flight_requests}/{self.config.max_concurrent_requests} (>75%){Colors.ENDC}")
            return 0

        # Gate: cooldown not yet satisfied
        if self.cooldown_required > 0 and self.consecutive_good_periods < self.cooldown_required:
            logger.info(f"{Colors.WARNING}  \u23f8\ufe0f Cooldown: {self.consecutive_good_periods}/{self.cooldown_required} good periods required before ramp{Colors.ENDC}")
            return 0

        # Cooldown satisfied — reset it
        if self.cooldown_required > 0:
            logger.info(f"  \u2713 Cooldown satisfied ({self.consecutive_good_periods} consecutive good periods)")
            self.cooldown_required = 0

        # Gate: minimum headroom of 20%
        headroom_pct = (self.config.max_ttft - ttft_value) / self.config.max_ttft * 100
        if headroom_pct < 20:
            logger.info(f"{Colors.WARNING}  \u23f8\ufe0f Headroom too low ({headroom_pct:.0f}% < 20% minimum){Colors.ENDC}")
            return 0

        # Throttle after cooldown: first 2 ramps = +1 only
        self.post_cooldown_ramps += 1
        if self.post_cooldown_ramps <= 2:
            return 1

        # Normal ramp: scale by headroom (less aggressive than before)
        # 1 base + 1 per 15% headroom (e.g., 75% headroom = 1 + 5 = 6 users)
        return max(1, 1 + int(headroom_pct / 15))

    def check_thresholds(self) -> bool:
        """Check if performance thresholds are met. Returns True if we can add users."""
        return self.calculate_users_to_add() > 0

    def should_rate_limit_dispatch(self) -> Tuple[bool, float]:
        """Check if new request dispatch should be rate-limited.

        When TTFT exceeds the configured threshold, we delay dispatching new
        requests to reduce queue depth in vLLM's scheduler. This allows existing
        prefills to complete faster, naturally reducing TTFT.

        Returns:
            (should_limit, backoff_seconds): Whether to rate-limit and how long to wait
        """
        if not self.config.enable_request_rate_limiting:
            return False, 0.0

        ttft_value = self.get_ttft_value()
        if ttft_value is None:
            return False, 0.0

        if ttft_value <= self.config.max_ttft:
            return False, 0.0

        # Calculate backoff proportional to overage (1s base, capped at 10s)
        overage_ratio = ttft_value / self.config.max_ttft
        backoff = min(1.0 * overage_ratio, 10.0)
        return True, backoff

    def prune_old_blocks(self, max_age: float = 900):
        """Remove blocks older than max_age seconds. O(k) where k = expired blocks.

        Args:
            max_age: Maximum age in seconds (default 900 = 15 minutes)
        """
        cutoff = time.time() - max_age
        while self.block_access_order and self.block_access_order[0][0] < cutoff:
            ts, key = self.block_access_order.popleft()
            # Only delete if this is still the latest access for this block
            if key in self.block_last_access and self.block_last_access[key] == ts:
                del self.block_last_access[key]

    def get_current_working_set_tokens(self) -> int:
        """Calculate working set size in tokens (blocks accessed within 15m window).

        Returns the number of unique blocks in block_last_access * chunk_size.
        This is O(1) since we just return the dict size.
        """
        return len(self.block_last_access) * self.config.chunk_size

    def compute_windowed_working_set(self) -> Dict[str, int]:
        """Compute working set for 1m, 5m, 15m time windows.

        Returns a dict mapping window label to token count.
        Single pass through block_last_access dict, counting by age buckets.
        O(n) where n = active blocks (bounded by pruning to 15m max).
        """
        now = time.time()
        cutoff_1m = now - 60
        cutoff_5m = now - 300
        # cutoff_15m not needed - all blocks in dict are within 15m after pruning

        count_1m = count_5m = count_15m = 0
        for ts in self.block_last_access.values():
            count_15m += 1  # All blocks in dict are within 15m after pruning
            if ts >= cutoff_5m:
                count_5m += 1
                if ts >= cutoff_1m:
                    count_1m += 1

        return {
            '1m': count_1m * self.config.chunk_size,
            '5m': count_5m * self.config.chunk_size,
            '15m': count_15m * self.config.chunk_size
        }

    def can_add_user_for_working_set(self, trace: dict) -> Tuple[bool, int]:
        """
        Check if adding a user would exceed global working set budget.

        Returns:
            Tuple of (can_add, estimated_new_tokens)
        """
        if self.config.max_working_set_tokens == 0:
            return True, 0  # Unlimited

        # Estimate new tokens from first real request's hash_ids (skip subagent markers)
        first_req = get_first_real_request(trace['requests'])
        if first_req is None:
            return True, 0  # No real requests
        estimated_new_blocks = len(first_req.get('hash_ids', []))
        estimated_new_tokens = estimated_new_blocks * self.config.chunk_size

        current_working_set = self.get_current_working_set_tokens()

        can_add = (current_working_set + estimated_new_tokens) <= self.config.max_working_set_tokens
        return can_add, estimated_new_tokens

    def can_add_user_for_period_budget(self, trace: dict) -> Tuple[bool, int]:
        """
        Check if adding a user would exceed per-period new token budget.

        Returns:
            Tuple of (can_add, estimated_tokens)
        """
        if self.config.max_new_tokens_per_period == 0:
            return True, 0

        # First request is all cache misses - use input_tokens as estimate
        first_req = trace['requests'][0]
        estimated_tokens = first_req.get('input_tokens', 0)

        remaining_budget = self.config.max_new_tokens_per_period - self.period_new_tokens

        can_add = estimated_tokens <= remaining_budget
        return can_add, estimated_tokens

    def compute_assessment_metrics(self, period_number: int, period_end_time: Optional[float] = None,
                                     pending_task_start_times: Optional[List[float]] = None) -> AssessmentPeriodMetrics:
        """
        Compute metrics for the current assessment period.

        Uses timestamps to attribute metrics to periods:
        - Input tokens: counted when prefill completes (at TTFT) in this period
        - Output tokens: proportionally attributed based on when tokens were GENERATED
        - TTFT stats: from requests where prefill completed in this period
        """
        start_time = self.current_period_start or time.time()
        end_time = period_end_time or time.time()
        duration = end_time - start_time

        active_users, idle_users = self.get_user_counts()
        total_users = active_users + idle_users

        # Filter metrics by timestamp - requests where prefill completed in this period
        # These are requests that got their TTFT during this period
        period_prefill_metrics = [
            m for m in self.all_metrics
            if start_time < m.prefill_complete_time <= end_time and m.success
        ]

        # Request lifecycle tracking
        # Launched: requests that started in this period
        requests_launched = sum(
            1 for m in self.all_metrics
            if start_time < m.request_start_time <= end_time
        )

        # Completed: requests that finished (got last token) in this period
        period_completed_metrics = [
            m for m in self.all_metrics
            if start_time < m.request_complete_time <= end_time and m.success
        ]

        # Break down completed into new (started this period) vs prior (started earlier)
        requests_completed_new = sum(
            1 for m in period_completed_metrics
            if m.request_start_time > start_time
        )
        requests_completed_prior = len(period_completed_metrics) - requests_completed_new

        # Break down in-progress into new (started this period) vs prior
        pending_start_times = pending_task_start_times or []
        requests_in_progress = len(pending_start_times)
        requests_in_progress_new = sum(1 for t in pending_start_times if t > start_time)
        requests_in_progress_prior = requests_in_progress - requests_in_progress_new

        # Count unique users who had a request complete prefill this period
        users_with_requests = len(set(m.user_id for m in period_prefill_metrics))

        # TTFT stats from requests that got first token in this period
        ttfts = [m.ttft for m in period_prefill_metrics]

        # Input tokens: attributed when prefill completes (at TTFT)
        input_tokens = sum(m.input_tokens for m in period_prefill_metrics)

        # Output tokens: proportionally attributed based on when tokens were GENERATED
        # This handles requests that span multiple periods correctly
        output_tokens = 0
        for req in self.all_metrics:
            if req.success and req.token_timestamps and req.tokens_per_chunk:
                for chunk_time, chunk_tokens in zip(req.token_timestamps, req.tokens_per_chunk):
                    if start_time < chunk_time <= end_time:
                        output_tokens += chunk_tokens

        # Cache stats from prefill completions
        cache_hits = sum(m.cache_hit_blocks for m in period_prefill_metrics)
        cache_total = cache_hits + sum(m.cache_miss_blocks for m in period_prefill_metrics)

        # Count working set blocks (15m window - unique blocks accessed recently)
        working_set_blocks = len(self.block_last_access)

        # Update peak working set
        working_set_tokens = working_set_blocks * self.config.chunk_size
        if working_set_tokens > self.peak_working_set_tokens:
            self.peak_working_set_tokens = working_set_tokens

        # Update peak users
        if total_users > self.peak_users:
            self.peak_users = total_users

        # Calculate total request processing time from requests that started (prefill) this period
        total_request_time = sum(m.ttlt for m in period_prefill_metrics)
        # Aggregate user-time = duration * avg users (approximate)
        aggregate_user_time = duration * total_users if total_users > 0 else 1
        idle_time_pct = max(0, 1 - (total_request_time / aggregate_user_time)) * 100 if aggregate_user_time > 0 else 0

        # Calculate new tokens (cache misses) - each block is chunk_size tokens
        # This is the actual cache miss count for reporting purposes
        cache_miss_blocks = sum(m.cache_miss_blocks for m in period_prefill_metrics)
        new_tokens = cache_miss_blocks * self.config.chunk_size

        # Note: period_new_tokens is now tracked incrementally in create_user() for
        # budget enforcement. We don't overwrite it here - the estimate-based tracking
        # in create_user() is used for admission control, while new_tokens_ingested
        # in the metrics shows the actual cache misses for reporting.

        # Store period metrics for print_assessment (filtered by prefill time)
        self.period_metrics = period_prefill_metrics

        # Calculate TTFT headroom (None means no data = no headroom)
        ttft_value = self.get_ttft_value()
        if ttft_value is None:
            ttft_headroom_pct = 0  # No data = no headroom claimed
        elif self.config.max_ttft > 0:
            ttft_headroom_pct = max(0, (self.config.max_ttft - ttft_value) / self.config.max_ttft * 100)
        else:
            ttft_headroom_pct = 0

        # Count rate-limited users
        rate_limited_users = sum(1 for u in self.users.values() if u.state == "rate_limited")

        return AssessmentPeriodMetrics(
            period_number=period_number,
            start_time=start_time,
            end_time=end_time,
            active_users=active_users,
            idle_users=idle_users,
            users_with_requests=users_with_requests,
            requests_completed=len(period_completed_metrics),
            requests_launched=requests_launched,
            requests_completed_new=requests_completed_new,
            requests_completed_prior=requests_completed_prior,
            requests_in_progress=requests_in_progress,
            requests_in_progress_new=requests_in_progress_new,
            requests_in_progress_prior=requests_in_progress_prior,
            requests_per_second=len(period_completed_metrics) / duration if duration > 0 else 0,
            input_tokens_per_second=input_tokens / duration if duration > 0 else 0,
            output_tokens_per_second=output_tokens / duration if duration > 0 else 0,
            ttft_avg=np.mean(ttfts) if ttfts else 0,
            ttft_p50=np.percentile(ttfts, 50) if ttfts else 0,
            ttft_p95=np.percentile(ttfts, 95) if ttfts else 0,
            ttft_p99=np.percentile(ttfts, 99) if ttfts else 0,
            avg_cache_hit_rate=cache_hits / cache_total if cache_total > 0 else 0,
            working_set_blocks=working_set_blocks,
            users_added=self.period_users_added,
            users_completed=0,  # Could track this if needed
            total_request_time=total_request_time,
            idle_time_pct=idle_time_pct,
            new_tokens_ingested=new_tokens,
            ttft_headroom_pct=ttft_headroom_pct,
            rate_limited_users=rate_limited_users,
            rate_limit_events=self.period_rate_limit_events,
            admission_blocked_events=self.period_admission_blocked,
            dispatch_delay_avg=np.mean(self.period_dispatch_delays) if self.period_dispatch_delays else 0.0,
            dispatch_delay_max=max(self.period_dispatch_delays) if self.period_dispatch_delays else 0.0,
            in_flight_prefilling=self.in_flight_requests - self.in_flight_decoding,
            in_flight_decoding=self.in_flight_decoding
        )

    def print_assessment(self, metrics: AssessmentPeriodMetrics):
        """Print assessment period summary"""
        # Select the right metric based on config
        ttft_value = self.get_ttft_value()

        if ttft_value is None:
            # No TTFT data available
            measured_ttft = None
            metric_name = "TTFT"
            threshold_status = "⏳ No data"
        elif self.config.ttft_metric == 'max':
            # For max, we need to compute it from individual metrics
            ttfts = [m.ttft for m in self.period_metrics if m.success]
            measured_ttft = max(ttfts) if ttfts else 0
            metric_name = "Max TTFT"
            threshold_status = "✓" if measured_ttft < self.config.max_ttft else "⚠️ EXCEEDED"
        elif self.config.ttft_metric == 'avg':
            measured_ttft = metrics.ttft_avg
            metric_name = "Avg TTFT"
            threshold_status = "✓" if measured_ttft < self.config.max_ttft else "⚠️ EXCEEDED"
        else:  # p95
            measured_ttft = metrics.ttft_p95
            metric_name = "P95 TTFT"
            threshold_status = "✓" if measured_ttft < self.config.max_ttft else "⚠️ EXCEEDED"
        working_set_tokens = metrics.working_set_blocks * self.config.chunk_size

        logger.info(f"{Colors.PHASE}{'='*120}{Colors.ENDC}")
        logger.info(f"{Colors.PHASE}Assessment Period {metrics.period_number}{Colors.ENDC}")
        logger.info(f"{Colors.PHASE}{'='*120}{Colors.ENDC}")
        logger.info(f"  Users: {metrics.active_users + metrics.idle_users} total ({metrics.users_with_requests} active this period)")
        logger.info(f"  Requests: {metrics.requests_launched} launched | {metrics.requests_completed} completed ({metrics.requests_completed_new} new, {metrics.requests_completed_prior} prior) | {metrics.requests_in_progress} in-progress ({metrics.requests_in_progress_new} new, {metrics.requests_in_progress_prior} prior) ({metrics.requests_per_second:.2f} req/s)")
        if measured_ttft is None:
            logger.info(f"  {metric_name}: {threshold_status} (threshold: {self.config.max_ttft}s)")
        else:
            logger.info(f"  {metric_name}: {measured_ttft:.2f}s {threshold_status} (threshold: {self.config.max_ttft}s, headroom: {metrics.ttft_headroom_pct:.0f}%)")
        logger.info(f"  Throughput: {metrics.input_tokens_per_second:,.0f} input tok/s | {metrics.output_tokens_per_second:,.0f} output tok/s")
        logger.info(f"  Cache Hit Rate: {metrics.avg_cache_hit_rate:.1%} | New input tokens: {metrics.new_tokens_ingested:,} (budget: {self.config.max_new_tokens_per_period:,})")

        # Show working set with budget status if limit is configured
        if self.config.max_working_set_tokens > 0:
            pct_used = working_set_tokens * 100 / self.config.max_working_set_tokens
            logger.info(f"  Working Set: {working_set_tokens:,} / {self.config.max_working_set_tokens:,} tokens ({pct_used:.0f}% used)")
        else:
            logger.info(f"  Working Set: {working_set_tokens:,} tokens ({metrics.working_set_blocks} blocks)")

        # Show time-windowed working set (1m, 5m, 15m)
        windowed = self.compute_windowed_working_set()
        logger.info(f"  Working Set by Age: 1m: {windowed['1m']:,} | 5m: {windowed['5m']:,} | 15m: {windowed['15m']:,} tokens")


        # Show admission control metrics if enabled
        if self.config.max_concurrent_requests:
            total_in_flight = metrics.in_flight_prefilling + metrics.in_flight_decoding
            logger.info(f"  Admission Control: {total_in_flight}/{self.config.max_concurrent_requests} in-flight "
                       f"({metrics.in_flight_prefilling} prefilling, {metrics.in_flight_decoding} decoding) | "
                       f"{metrics.admission_blocked_events} blocked | "
                       f"dispatch delay: {metrics.dispatch_delay_avg:.2f}s avg, {metrics.dispatch_delay_max:.2f}s max")

            # Warn if the limit appears to be constraining throughput
            if metrics.admission_blocked_events > 100 or metrics.dispatch_delay_avg > 10.0:
                logger.warning(f"{Colors.WARNING}  ⚠️ Max concurrent requests limit ({self.config.max_concurrent_requests}) "
                              f"may be constraining throughput. Consider increasing --max-concurrent-requests{Colors.ENDC}")

    async def run_user_request(self, user: UserSession) -> Optional[RequestMetrics]:
        """Execute a single request for a user"""
        request = user.get_next_request()
        if request is None:
            return None

        user.state = "active"
        logger.debug(f"  📤 {user.user_id} req {user.current_idx + 1}: firing ({request['input_tokens']:,} input tokens)")

        # Track hash_ids for working set when request fires (not when it completes)
        # Prefix with trace_id to make hash_ids unique per conversation
        # (raw hash_ids are sequential integers that overlap across traces)
        current_time = time.time()
        for h in request.get('hash_ids', []):
            key = (user.trace_id, h)
            self.block_last_access[key] = current_time
            self.block_access_order.append((current_time, key))

        # Build messages (only user messages are generated, assistant messages come from history)
        # Pass canonical prefix for cross-conversation cache sharing on first request
        messages, max_tokens = user.build_messages(
            request,
            canonical_prefix=self.canonical_prefix_content,
            canonical_prefix_tokens=self.canonical_prefix_tokens
        )

        # Calculate expected cache hits
        cache_hits, cache_misses = user.calculate_cache_hits(request)

        # Send request
        stream = True  # Always stream for accurate TTFT measurement
        expected_output = request.get('output_tokens', 100)

        # Track if first token was received (for proper counter management)
        first_token_received = False

        def on_first_token():
            nonlocal first_token_received
            first_token_received = True
            self.in_flight_decoding += 1

        result = await self.api_client.send_request(
            messages,
            max_tokens,
            stream=stream,
            on_first_token=on_first_token
        )

        # Decrement decode counter if first token was received
        # (request has moved from decode phase to complete)
        if first_token_received:
            self.in_flight_decoding -= 1

        # Unpack result
        response_text = result['response_text']
        ttft = result['ttft']
        ttlt = result['ttlt']
        actual_output = result['output_tokens']
        error_type = result['error_type']
        request_start_time = result['start_time']
        prefill_complete_time = result['first_token_time']
        request_complete_time = result['complete_time']
        token_timestamps = result['token_timestamps']
        tokens_per_chunk = result['tokens_per_chunk']

        # Track connection errors
        if error_type == "connection":
            self.consecutive_connection_errors += 1
            self.total_connection_errors += 1
            if self.consecutive_connection_errors >= self.max_consecutive_errors:
                logger.error(
                    f"{Colors.FAIL}FATAL: {self.consecutive_connection_errors} consecutive connection errors. "
                    f"Server appears to be down. Stopping test.{Colors.ENDC}"
                )
                self.running = False
                return None
        elif error_type is None:
            # Successful request resets consecutive error counter
            self.consecutive_connection_errors = 0
            logger.debug(f"  📥 {user.user_id} req {user.current_idx + 1}: complete (TTFT: {ttft:.2f}s, {actual_output} output tokens, {ttlt:.2f}s total)")

        # Store actual assistant response for use in subsequent requests' history
        # For request pairs, only store after the non-streaming request
        user.store_assistant_response(response_text, actual_output, request)

        # Track shortfall if output tokens significantly lower than expected (< 80%)
        # The shortfall will be added to the next user message to maintain token counts
        user.record_shortfall(expected_output, actual_output)
        if expected_output > 0 and actual_output < expected_output * 0.8:
            logger.warning(f"  ⚠️ {user.user_id} req {user.current_idx}: Output tokens {actual_output} < 80% of expected {expected_output}")

        # Calculate ITL
        itl = (ttlt - ttft) / (actual_output - 1) if actual_output > 1 and ttlt > ttft else 0

        # Create metrics
        metrics = RequestMetrics(
            user_id=user.user_id,
            request_idx=user.current_idx,
            trace_id=user.trace_id,
            timestamp=time.time(),
            request_type=request['type'],
            input_tokens=request['input_tokens'],
            output_tokens_expected=expected_output,
            output_tokens_actual=actual_output,
            cache_hit_blocks=cache_hits,
            cache_miss_blocks=cache_misses,
            ttft=ttft,
            ttlt=ttlt,
            itl=itl,
            delay_expected=user.get_delay_until_next(),
            delay_actual=0,  # Will be set by orchestrator
            success=error_type is None,
            request_start_time=request_start_time,
            prefill_complete_time=prefill_complete_time,
            request_complete_time=request_complete_time,
            token_timestamps=token_timestamps,
            tokens_per_chunk=tokens_per_chunk
        )

        user.metrics.append(metrics)
        user.advance()

        if user.current_idx >= len(user.requests):
            user.state = "completed"
        else:
            user.state = "idle"

        return metrics

    async def run(self):
        """Main test loop"""
        self.test_start_time = time.time()
        self.current_period_start = time.time()
        period_number = 0

        logger.info(f"\n{Colors.HEADER}Starting test with {self.config.start_users} user(s)...{Colors.ENDC}\n")

        # Create initial users (with delay between each to avoid overwhelming server)
        # Don't enforce budgets for initial users - just warn if exceeded
        await self.create_users_batch(self.config.start_users, enforce_budgets=False)

        # Warn if initial users exceed budgets
        initial_new_tokens = sum(
            user.trace['requests'][0]['input_tokens']
            for user in self.users.values()
        )
        if initial_new_tokens > self.config.max_new_tokens_per_period:
            logger.warning(
                f"{Colors.WARNING}Initial users will ingest {initial_new_tokens:,} new input tokens, "
                f"exceeding per-period budget of {self.config.max_new_tokens_per_period:,} tokens. "
                f"No additional users will be added until budget allows.{Colors.ENDC}"
            )

        # Also check working set budget
        if self.config.max_working_set_tokens > 0:
            initial_working_set = self.get_current_working_set_tokens()
            if initial_working_set > self.config.max_working_set_tokens:
                logger.warning(
                    f"{Colors.WARNING}Initial users have working set of {initial_working_set:,} tokens, "
                    f"exceeding limit of {self.config.max_working_set_tokens:,} tokens. "
                    f"No additional users will be added until working set decreases.{Colors.ENDC}"
                )

        # Track in-flight tasks: maps task -> (user_id, start_time)
        pending_tasks: Dict[asyncio.Task, Tuple[str, float]] = {}

        try:
            while self.running:
                # Check test duration
                if self.config.test_duration:
                    elapsed = time.time() - self.test_start_time
                    if elapsed >= self.config.test_duration:
                        logger.info(f"\n{Colors.HEADER}Test duration reached ({self.config.test_duration}s){Colors.ENDC}")
                        break

                # Check if we have any users or pending tasks
                if not self.users and not pending_tasks:
                    if not self.config.recycle:
                        logger.info("All traces completed and recycling disabled")
                        break
                    else:
                        self.create_user()

                now = time.time()
                users_to_remove = []

                # Phase 0: Process sub-agent markers before collecting ready users
                # When a user's next request is type='subagent', spawn it and advance
                for user_id, user in list(self.users.items()):
                    if user.state != "idle":
                        continue
                    # Process consecutive sub-agent markers (e.g., 4 agents spawned at once)
                    while user.current_idx < len(user.requests):
                        req = user.requests[user.current_idx]
                        if req.get('type') != 'subagent':
                            break
                        self.spawn_subagent(user, req)
                        user.current_idx += 1
                    # Check if we exhausted all requests (only had sub-agents left)
                    if user.current_idx >= len(user.requests):
                        user.state = "completed"

                # Phase 1: Collect ready users with their ready_at times
                ready_users = []
                for user_id, user in list(self.users.items()):
                    if user.state == "completed":
                        users_to_remove.append((user_id, "completed"))
                    elif user.state == "truncated":
                        users_to_remove.append((user_id, "truncated"))
                    elif user.state == "rate_limited":
                        # Check if backoff has elapsed
                        if now >= user.rate_limit_until:
                            user.state = "idle"  # Transition back for retry
                            logger.debug(f"  ↻ {user.user_id} exiting rate-limit (attempt #{user.rate_limit_count})")
                    elif user.state == "idle":
                        # Parent waiting for sub-agents to complete — don't dispatch
                        if user.pending_subagents:
                            continue
                        # Calculate when this user became ready
                        capped_delay = self._compute_delay(user)
                        last_time = user.last_request_time or user.start_time
                        ready_at = last_time + capped_delay

                        if now >= ready_at:
                            ready_users.append((user_id, user, ready_at))
                    # "active" users already have in-flight requests, skip them

                # Phase 2: Sort by ready_at (fair ordering - longest waiting first)
                ready_users.sort(key=lambda x: x[2])

                # Phase 3: Dispatch in fair order, respecting admission limit
                for user_id, user, ready_at in ready_users:
                    # Admission control check
                    if (self.config.max_concurrent_requests and
                        self.in_flight_requests >= self.config.max_concurrent_requests):
                        self.period_admission_blocked += 1
                        continue  # At capacity, skip this user

                    # Check if rate-limiting is needed before dispatching
                    should_limit, backoff = self.should_rate_limit_dispatch()
                    if should_limit:
                        # Apply exponential backoff (1.5x per retry)
                        actual_backoff = backoff * (1.5 ** user.rate_limit_count)
                        user.state = "rate_limited"
                        user.rate_limit_until = now + actual_backoff
                        user.rate_limit_count += 1
                        user.total_rate_limit_count += 1
                        self.period_rate_limit_events += 1
                        logger.info(f"{Colors.WARNING}  ⏱️ {user.user_id} rate-limited "
                                   f"(backoff: {actual_backoff:.1f}s, attempt #{user.rate_limit_count}){Colors.ENDC}")
                        continue

                    # Reset rate-limit count on successful dispatch
                    user.rate_limit_count = 0

                    # Track dispatch delay (how far behind schedule)
                    dispatch_delay = now - ready_at
                    self.period_dispatch_delays.append(dispatch_delay)

                    # Increment in-flight counter and dispatch
                    self.in_flight_requests += 1
                    task = asyncio.create_task(self.run_user_request(user))
                    pending_tasks[task] = (user_id, now)

                # Remove completed/truncated users
                for user_id, reason in users_to_remove:
                    user = self.users.get(user_id)
                    if user and user.is_subagent:
                        # Clean up sub-agent reference from parent
                        parent_id = user.parent_user_id
                        if parent_id in self.users:
                            parent = self.users[parent_id]
                            if user_id in parent.pending_subagents:
                                parent.pending_subagents.remove(user_id)
                        logger.info(f"{Colors.SUCCESS}  ✅ {user_id} subagent completed{Colors.ENDC}")
                    self.remove_user(user_id, reason)
                    # Add replacement if recycling
                    if self.config.recycle and len(self.users) < self.config.max_users:
                        self.create_user()

                # Check for completed tasks (non-blocking with short timeout)
                if pending_tasks:
                    done, _ = await asyncio.wait(
                        pending_tasks.keys(),
                        timeout=0.1,  # Short timeout to keep loop responsive
                        return_when=asyncio.FIRST_COMPLETED
                    )

                    # Process completed tasks
                    for task in done:
                        user_id, _start_time = pending_tasks.pop(task)
                        # Decrement in-flight counter (request completed)
                        self.in_flight_requests -= 1
                        try:
                            result = await task
                            if isinstance(result, RequestMetrics):
                                self.all_metrics.append(result)
                                # Note: period_metrics will be calculated based on timestamps
                        except Exception as e:
                            logger.error(f"Task failed for {user_id}: {e}")
                else:
                    # No pending tasks, small sleep to prevent tight loop
                    await asyncio.sleep(0.01)

                # Check for assessment period (runs on fixed schedule regardless of in-flight requests)
                if time.time() - self.current_period_start >= self.config.assessment_period:
                    period_number += 1
                    period_end_time = self.current_period_start + self.config.assessment_period

                    # Prune old blocks from working set tracking (keeps 15m window)
                    self.prune_old_blocks()

                    # Calculate metrics based on timestamps (which requests' TTFT fell in this period)
                    pending_start_times = [start_time for _, start_time in pending_tasks.values()]
                    assessment = self.compute_assessment_metrics(period_number, period_end_time, pending_start_times)
                    self.assessment_periods.append(assessment)
                    self.print_assessment(assessment)

                    # Update cooldown tracking based on TTFT
                    ttft_val = self.get_ttft_value()
                    if ttft_val is not None and ttft_val < self.config.max_ttft:
                        self.consecutive_good_periods += 1
                        self.consecutive_exceeded_periods = 0
                    else:
                        self.consecutive_exceeded_periods += 1
                        self.consecutive_good_periods = 0
                        # Set cooldown based on severity of distress
                        if self.consecutive_exceeded_periods >= 10:
                            self.cooldown_required = 5
                        elif self.consecutive_exceeded_periods >= 3:
                            self.cooldown_required = 3
                        else:
                            self.cooldown_required = 2
                        self.post_cooldown_ramps = 0  # Reset throttle counter

                    # Calculate how many users to add based on headroom and budget
                    users_to_add = self.calculate_users_to_add()
                    max_to_add = min(users_to_add, self.config.max_users - len(self.users))
                    if max_to_add > 0:
                        new_users_list = await self.create_users_batch(max_to_add)
                        users_added = len(new_users_list)
                    else:
                        users_added = 0

                    if users_added > 0:
                        old_users = assessment.active_users + assessment.idle_users - users_added
                        new_users = old_users + users_added
                        headroom = assessment.ttft_headroom_pct
                        cooldown_info = f", cooldown: {self.cooldown_required}" if self.cooldown_required > 0 else ""
                        logger.info(f"{Colors.SUCCESS}  \u2192 Users {old_users} \u2192 {new_users} (+{users_added}) (headroom: {headroom:.0f}%{cooldown_info}, budget remaining: {self.config.max_new_tokens_per_period - self.period_new_tokens:,} tokens){Colors.ENDC}")

                    # Reset period counters - use period_end_time to maintain contiguous periods
                    self.current_period_start = period_end_time
                    self.period_users_added = users_added  # Store for next period's metrics
                    self.period_new_tokens = 0  # Reset token budget for new period
                    self.period_rate_limit_events = 0  # Reset rate-limit counter
                    self.period_admission_blocked = 0  # Reset admission control counter
                    self.period_dispatch_delays = []  # Reset dispatch delay tracking

        except KeyboardInterrupt:
            logger.info(f"\n{Colors.WARNING}Test interrupted by user{Colors.ENDC}")

        # Wait for remaining in-flight requests to complete
        if pending_tasks:
            logger.info(f"Waiting for {len(pending_tasks)} in-flight requests to complete...")
            done, pending = await asyncio.wait(pending_tasks.keys(), timeout=60)
            for task in done:
                try:
                    result = await task
                    if isinstance(result, RequestMetrics):
                        self.all_metrics.append(result)
                except Exception as e:
                    logger.error(f"Task failed during cleanup: {e}")
            if pending:
                logger.warning(f"{len(pending)} requests still outstanding after 60s timeout — cancelling")
                for task in pending:
                    task.cancel()

        self.running = False
        self.print_summary()

    def print_summary(self):
        """Print final test summary"""
        elapsed = time.time() - self.test_start_time if self.test_start_time else 0

        ttfts = [m.ttft for m in self.all_metrics if m.success]
        total_input = sum(m.input_tokens for m in self.all_metrics)
        total_output = sum(m.output_tokens_actual for m in self.all_metrics)

        cache_hits = sum(m.cache_hit_blocks for m in self.all_metrics)
        cache_total = cache_hits + sum(m.cache_miss_blocks for m in self.all_metrics)

        logger.info(f"{Colors.PHASE}{'='*120}{Colors.ENDC}")
        logger.info(f"{Colors.PHASE}{Colors.BOLD}Test Complete{Colors.ENDC}")
        logger.info(f"{Colors.PHASE}{'='*120}{Colors.ENDC}")
        logger.info(f"Duration: {elapsed:.1f}s")
        logger.info(f"Total Requests: {len(self.all_metrics)}")
        logger.info(f"Total Users Created: {self.user_counter}")
        logger.info(f"Peak Concurrent Users: {self.peak_users}")
        logger.info(f"")
        logger.info(f"{Colors.METRIC}Performance Summary:{Colors.ENDC}")
        if ttfts:
            logger.info(f"  TTFT avg/p50/p95/max: {np.mean(ttfts):.2f}s / {np.percentile(ttfts, 50):.2f}s / {np.percentile(ttfts, 95):.2f}s / {max(ttfts):.2f}s")
        logger.info(f"  Throughput: {total_input/elapsed:,.0f} input tok/s | {total_output/elapsed:,.0f} output tok/s")
        logger.info(f"  Avg Cache Hit Rate: {cache_hits/cache_total:.1%}" if cache_total > 0 else "  Cache hits: N/A")
        logger.info(f"  Peak Working Set: {self.peak_working_set_tokens:,} tokens")
        if self.canonical_prefix_tokens > 0:
            logger.info(f"  Warm Prefix: {self.canonical_prefix_tokens:,} tokens ({self.config.warm_prefix_pct:.0%} of max tool+system)")
        if self.total_connection_errors > 0:
            logger.info(f"  {Colors.WARNING}Connection Errors: {self.total_connection_errors}{Colors.ENDC}")
        logger.info(f"")
        logger.info(f"{Colors.SUCCESS}Results saved to: {self.config.output_dir}{Colors.ENDC}")
        logger.info(f"{Colors.PHASE}{'='*120}{Colors.ENDC}")


# =============================================================================
# Output and Graphs
# =============================================================================

def generate_graphs(orchestrator: TestOrchestrator, config: TestConfig):
    """Generate interactive Plotly graphs for trace replay results"""
    output_path = Path(config.output_dir)

    if not orchestrator.assessment_periods:
        logger.warning("No assessment periods to graph")
        return

    periods = orchestrator.assessment_periods

    # Graph 1: Performance over time (TTFT and throughput)
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=('TTFT Over Time', 'Throughput Over Time'),
        vertical_spacing=0.12,
        specs=[[{"secondary_y": False}], [{"secondary_y": True}]]
    )

    x_vals = [p.period_number for p in periods]

    # TTFT traces
    fig.add_trace(
        go.Scatter(x=x_vals, y=[p.ttft_p50 for p in periods],
                   name='TTFT p50', mode='lines+markers', line=dict(color='#2ecc71')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=x_vals, y=[p.ttft_p95 for p in periods],
                   name='TTFT p95', mode='lines+markers', line=dict(color='#e74c3c')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=x_vals, y=[p.ttft_p99 for p in periods],
                   name='TTFT p99', mode='lines+markers', line=dict(color='#9b59b6', dash='dash')),
        row=1, col=1
    )

    # Add threshold line
    fig.add_hline(y=config.max_ttft, line_dash="dash", line_color="red",
                  annotation_text=f"Threshold ({config.max_ttft}s)", row=1, col=1)

    # Throughput traces
    fig.add_trace(
        go.Scatter(x=x_vals, y=[p.input_tokens_per_second for p in periods],
                   name='Input tok/s', mode='lines+markers', line=dict(color='#3498db')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=x_vals, y=[p.output_tokens_per_second for p in periods],
                   name='Output tok/s', mode='lines+markers', line=dict(color='#e67e22')),
        row=2, col=1, secondary_y=True
    )

    # Add user count on secondary y-axis (same axis as output tokens)
    fig.add_trace(
        go.Scatter(x=x_vals, y=[p.active_users + p.idle_users for p in periods],
                   name='Total Users', mode='lines', line=dict(color='#95a5a6', dash='dot')),
        row=2, col=1, secondary_y=True
    )

    fig.update_layout(
        title=f'Trace Replay Performance Over Time<br><sub>Model: {orchestrator.api_client.model}</sub>',
        height=700,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis2=dict(title='Assessment Period')
    )

    # Update y-axis titles
    fig.update_yaxes(title_text='TTFT (seconds)', row=1, col=1)
    fig.update_yaxes(title_text='Input tok/s', row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text='Output tok/s', row=2, col=1, secondary_y=True)

    fig.write_html(output_path / "trace_replay_performance_over_time.html")
    logger.info("Generated: trace_replay_performance_over_time.html")

    # Graph 2: Cache hit rate and users over time
    fig2 = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=('Cache Hit Rate', 'Users and Requests'),
        vertical_spacing=0.12
    )

    # Cache hit rate
    fig2.add_trace(
        go.Scatter(x=x_vals, y=[p.avg_cache_hit_rate * 100 for p in periods],
                   name='Cache Hit Rate %', mode='lines+markers',
                   fill='tozeroy', line=dict(color='#27ae60')),
        row=1, col=1
    )

    # Users and requests
    fig2.add_trace(
        go.Bar(x=x_vals, y=[p.active_users + p.idle_users for p in periods],
               name='Total Users', marker_color='#3498db', opacity=0.7),
        row=2, col=1
    )
    fig2.add_trace(
        go.Scatter(x=x_vals, y=[p.requests_completed for p in periods],
                   name='Requests', mode='lines+markers', line=dict(color='#e74c3c'),
                   yaxis='y4'),
        row=2, col=1
    )

    fig2.update_layout(
        title='Cache and User Activity Over Time',
        height=600,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(title='Cache Hit Rate (%)', range=[0, 100]),
        yaxis2=dict(title='Users'),
        yaxis4=dict(title='Requests', overlaying='y2', side='right'),
        xaxis2=dict(title='Assessment Period')
    )

    fig2.write_html(output_path / "trace_replay_cache_hit_over_time.html")
    logger.info("Generated: trace_replay_cache_hit_over_time.html")

    # Graph 3: User timeline (if we have lifecycle events)
    if orchestrator.lifecycle_events:
        events = orchestrator.lifecycle_events
        test_start = orchestrator.test_start_time or 0

        # Group events by user
        user_events = defaultdict(list)
        for e in events:
            user_events[e.user_id].append(e)

        fig3 = go.Figure()

        colors = {
            'started': '#27ae60',
            'completed': '#3498db',
            'truncated': '#e74c3c',
            'active': '#f39c12',
            'idle': '#95a5a6'
        }

        for user_id, user_evts in user_events.items():
            for evt in user_evts:
                t = evt.timestamp - test_start
                fig3.add_trace(go.Scatter(
                    x=[t],
                    y=[user_id],
                    mode='markers+text',
                    marker=dict(size=12, color=colors.get(evt.event_type, '#333')),
                    text=[evt.event_type[0].upper()],
                    textposition='middle center',
                    name=evt.event_type,
                    hovertemplate=f"{user_id}<br>{evt.event_type}<br>t={t:.1f}s<br>{evt.details}<extra></extra>",
                    showlegend=False
                ))

        # Add request markers
        if orchestrator.all_metrics:
            # Group requests by user
            user_requests = defaultdict(list)
            for m in orchestrator.all_metrics:
                user_requests[m.user_id].append(m)

            for user_id, requests in user_requests.items():
                for req in requests:
                    t = req.timestamp - test_start
                    color = '#2ecc71' if req.success else '#e74c3c'
                    fig3.add_trace(go.Scatter(
                        x=[t],
                        y=[user_id],
                        mode='markers',
                        marker=dict(size=6, color=color, symbol='diamond'),
                        hovertemplate=f"{user_id}<br>Request {req.request_idx}<br>t={t:.1f}s<br>TTFT: {req.ttft:.2f}s<br>In: {req.input_tokens:,} tok<br>Out: {req.output_tokens_actual} tok<extra></extra>",
                        showlegend=False
                    ))

        # Add legend entries
        for event_type, color in colors.items():
            if event_type in ('started', 'completed', 'truncated'):
                fig3.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode='markers',
                    marker=dict(size=12, color=color),
                    name=event_type.capitalize()
                ))

        # Add legend entry for requests
        fig3.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=6, color='#2ecc71', symbol='diamond'),
            name='Request'
        ))

        fig3.update_layout(
            title='User Timeline',
            xaxis_title='Time (seconds)',
            yaxis_title='User',
            height=max(400, len(user_events) * 30 + 100),
            showlegend=True
        )

        fig3.write_html(output_path / "trace_replay_user_timeline.html")
        logger.info("Generated: trace_replay_user_timeline.html")


def save_results(orchestrator: TestOrchestrator, config: TestConfig):
    """Save all results to files"""
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save detailed metrics
    if orchestrator.all_metrics:
        df = pd.DataFrame([m.to_dict() for m in orchestrator.all_metrics])
        df.to_csv(output_path / "detailed_results.csv", index=False)
        logger.info(f"Saved detailed results: {len(df)} requests")

    # Save assessment periods
    if orchestrator.assessment_periods:
        df = pd.DataFrame([asdict(p) for p in orchestrator.assessment_periods])
        df.to_csv(output_path / "summary_trace_replay.csv", index=False)
        logger.info(f"Saved assessment periods: {len(df)} periods")

    # Save user lifecycle
    if orchestrator.lifecycle_events:
        df = pd.DataFrame([asdict(e) for e in orchestrator.lifecycle_events])
        df.to_csv(output_path / "user_lifecycle.csv", index=False)
        logger.info(f"Saved user lifecycle: {len(df)} events")

    # Save test metadata
    metadata = {
        "model": orchestrator.api_client.model,
        "api_endpoint": config.api_endpoint,
        "mode": "trace_replay",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trace_stats": asdict(orchestrator.trace_manager.stats) if orchestrator.trace_manager.stats else {},
        "config": config.to_dict()
    }
    with open(output_path / "test_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    # Save progress for compatibility
    progress = {
        "parameters": config.to_dict(),
        "completed": True
    }
    with open(output_path / "progress.json", 'w') as f:
        json.dump(progress, f, indent=2)


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Trace Replay Performance Testing Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Required arguments
    parser.add_argument("--api-endpoint", type=str, required=True,
                        help="API server endpoint (e.g., http://localhost:8000)")
    parser.add_argument("--trace-directory", type=str, required=True,
                        help="Directory containing trace JSON files")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for results")

    # Performance thresholds
    parser.add_argument("--max-ttft", type=float, default=2.0,
                        help="Maximum acceptable TTFT in seconds (default: 2.0)")
    parser.add_argument("--ttft-metric", type=str, default="p95",
                        choices=["max", "avg", "p95"],
                        help="TTFT metric to use for threshold: max (maximum), avg (average), p95 (95th percentile). Default: p95")
    parser.add_argument("--min-output-tokens-per-req", type=float, default=None,
                        help="Minimum output tokens/s per request (optional)")

    # User management
    parser.add_argument("--start-users", type=int, default=1,
                        help="Initial number of users (default: 1)")
    parser.add_argument("--max-users", type=int, default=50,
                        help="Maximum concurrent users (default: 50)")
    parser.add_argument("--recycle", action="store_true",
                        help="Replace completed users with new traces")
    parser.add_argument("--max-new-tokens-per-period", type=int, default=500000,
                        help="Max new (cache miss) tokens allowed per assessment period for user scaling (default: 500000)")
    parser.add_argument("--max-working-set-tokens", type=int, default=0,
                        help="Maximum working set size in tokens (0 = unlimited). "
                             "Limits total unique tokens across all active users.")

    # Trace filtering
    parser.add_argument("--min-requests", type=int, default=1,
                        help="Minimum requests per trace to include (default: 1)")

    # Timing
    parser.add_argument("--max-delay", type=float, default=60.0,
                        help="Maximum delay between requests in seconds (default: 60)")
    parser.add_argument("--time-scale", type=float, default=1.0,
                        help="Time scaling factor (1.0 = real-time, 0.5 = 2x faster)")
    parser.add_argument("--timing-strategy", type=str, default="original",
                        choices=["original", "think-only", "api-scaled"],
                        help="Timing strategy: original (use t differences), "
                             "think-only (client think time only, ignore API time), "
                             "api-scaled (api_time * api-time-scale + think_time)")
    parser.add_argument("--api-time-scale", type=float, default=1.0,
                        help="Multiplier for API processing time with api-scaled strategy (default: 1.0)")
    parser.add_argument("--test-duration", type=int, default=None,
                        help="Maximum test duration in seconds (default: unlimited)")
    parser.add_argument("--assessment-period", type=int, default=30,
                        help="Assessment period in seconds (default: 30)")

    # Context and tokenizer
    parser.add_argument("--max-context", type=int, default=128000,
                        help="Maximum input tokens per request (default: 128000)")
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-Coder-32B-Instruct",
                        help="Tokenizer to use for synthetic data generation")
    parser.add_argument("--chunk-size", type=int, default=64,
                        help="Cache block size in tokens (default: 64)")

    # Output control
    parser.add_argument("--no-color", action="store_true",
                        help="Disable colored output (useful for light terminal backgrounds)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--skip-graphs", action="store_true",
                        help="Skip graph generation")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible trace selection")

    # Generation parameter overrides (None = use model-specific defaults if available)
    parser.add_argument("--temperature", type=float, default=None,
                        help="Override temperature for generation (e.g., 0.7)")
    parser.add_argument("--top-p", type=float, default=None,
                        help="Override top_p for generation (e.g., 0.8)")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Override top_k for generation (e.g., 20)")
    parser.add_argument("--repetition-penalty", type=float, default=None,
                        help="Override repetition_penalty for generation (e.g., 1.05)")

    # Rate limiting
    parser.add_argument("--enable-request-rate-limiting", action="store_true",
                        help="Enable request-level rate-limiting when TTFT exceeds threshold "
                             "(delays new request dispatch with exponential backoff)")

    # Admission control
    parser.add_argument("--max-concurrent-requests", type=int, default=50,
                        help="Max concurrent in-flight requests (admission control). "
                             "When reached, new dispatches are blocked until requests complete. "
                             "Default: 50. Set to 0 to disable.")

    # Warm prefix for cross-conversation cache sharing
    parser.add_argument("--warm-prefix-pct", type=float, default=0.5,
                        help="Percentage (0.0-1.0) of tool+system tokens to pre-warm for "
                             "cross-conversation cache sharing. All users share this prefix "
                             "content, enabling cache hits after user 1. Default: 0.5 (50%%). "
                             "Set to 0 to disable.")

    # Trace advancement
    parser.add_argument("--advance-min", type=float, default=0.0,
                        help="Minimum start position as fraction (0.0-1.0). Default: 0.0 (beginning)")
    parser.add_argument("--advance-max", type=float, default=0.0,
                        help="Maximum start position as fraction (0.0-1.0). Default: 0.0 (beginning)")

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

async def main():
    args = parse_arguments()

    # Disable colors if requested
    if args.no_color:
        Colors.disable()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)

    # Create config
    config = TestConfig(
        api_endpoint=args.api_endpoint,
        trace_directory=args.trace_directory,
        output_dir=args.output_dir,
        max_context=args.max_context,
        max_ttft=args.max_ttft,
        ttft_metric=args.ttft_metric,
        min_output_tokens_per_req=args.min_output_tokens_per_req,
        start_users=args.start_users,
        max_users=args.max_users,
        max_delay=args.max_delay,
        time_scale=args.time_scale,
        timing_strategy=args.timing_strategy,
        api_time_scale=args.api_time_scale,
        assessment_period=args.assessment_period,
        test_duration=args.test_duration,
        recycle=args.recycle,
        chunk_size=args.chunk_size,
        verbose=args.verbose,
        tokenizer_id=args.tokenizer,
        min_requests=args.min_requests,
        max_new_tokens_per_period=args.max_new_tokens_per_period,
        max_working_set_tokens=args.max_working_set_tokens,
        seed=args.seed,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        enable_request_rate_limiting=args.enable_request_rate_limiting,
        max_concurrent_requests=args.max_concurrent_requests,
        warm_prefix_pct=args.warm_prefix_pct,
        advance_min=args.advance_min,
        advance_max=args.advance_max
    )

    # Print header
    logger.info(f"{Colors.PHASE}{'='*120}{Colors.ENDC}")
    logger.info(f"{Colors.PHASE}{Colors.BOLD}Trace Replay Performance Tester v{__version__}{Colors.ENDC}")
    logger.info(f"{Colors.PHASE}{'='*120}{Colors.ENDC}")

    # Load traces
    trace_manager = TraceManager(Path(config.trace_directory), config.max_context, config.min_requests, config.seed)
    num_traces = trace_manager.load_traces()

    if num_traces == 0:
        logger.error("No valid traces found!")
        return

    stats = trace_manager.get_stats()
    logger.info(f"Trace Statistics:")
    logger.info(f"  Total files: {stats.total_traces}")
    logger.info(f"  After filtering (max_context <= {config.max_context}): {stats.filtered_traces}")
    logger.info(f"  Total requests: {stats.total_requests:,}")
    logger.info(f"  Avg requests/trace: {stats.avg_requests_per_trace:.1f}")
    logger.info(f"  Avg cache hit rate: {stats.avg_cache_hit_rate:.1%}")
    logger.info(f"  Traces with tool_use: {stats.traces_with_tool_use}")
    logger.info(f"  Max shared prefix (tool+system): {stats.max_shared_prefix_tokens:,} tokens")

    logger.info(f"{'-' * 120}")
    logger.info(f"Configuration:")
    logger.info(f"  Max TTFT: {config.max_ttft}s ({config.ttft_metric})")
    logger.info(f"  Max Context: {config.max_context:,} tokens")
    logger.info(f"  Max Delay: {config.max_delay}s")
    if config.timing_strategy != "original":
        logger.info(f"  Timing Strategy: {config.timing_strategy}" +
                     (f" (api_time_scale={config.api_time_scale})" if config.timing_strategy == "api-scaled" else ""))
    logger.info(f"  Start Users: {config.start_users}")
    logger.info(f"  Max Users: {config.max_users}")
    logger.info(f"  Recycle: {config.recycle}")
    logger.info(f"  New Token Budget: {config.max_new_tokens_per_period:,} tokens/period")
    if config.max_working_set_tokens > 0:
        logger.info(f"  Working Set Limit: {config.max_working_set_tokens:,} tokens")
    else:
        logger.info(f"  Working Set Limit: unlimited")
    if config.test_duration:
        logger.info(f"  Test Duration: {config.test_duration}s")
    if config.max_concurrent_requests:
        logger.info(f"  Max Concurrent Requests: {config.max_concurrent_requests}")
    if config.warm_prefix_pct > 0:
        warm_tokens = int(config.warm_prefix_pct * stats.max_shared_prefix_tokens) if stats.max_shared_prefix_tokens > 0 else 0
        logger.info(f"  Warm Prefix: {config.warm_prefix_pct:.0%} of tool+system ({warm_tokens:,} tokens)")
    else:
        logger.info(f"  Warm Prefix: disabled")
    if config.advance_max > 0:
        logger.info(f"  Trace Advancement: {config.advance_min:.0%} - {config.advance_max:.0%}")
    logger.info(f"{Colors.HEADER}{'=' * 120}{Colors.ENDC}")

    # Initialize components
    generator = SyntheticMessageGenerator(config.tokenizer_id, config.chunk_size)
    api_client = APIClient(
        config.api_endpoint,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        repetition_penalty=config.repetition_penalty
    )
    await api_client.detect_model()

    # Create orchestrator
    orchestrator = TestOrchestrator(config, trace_manager, generator, api_client)

    # Generate canonical warm prefix for cross-conversation cache sharing
    if config.warm_prefix_pct > 0:
        max_shared = stats.max_shared_prefix_tokens
        if max_shared > 0:
            warm_tokens = int(config.warm_prefix_pct * max_shared)
            orchestrator.canonical_prefix_content = generator.generate_canonical_prefix(warm_tokens)
            orchestrator.canonical_prefix_tokens = warm_tokens
            logger.info(f"{Colors.OKCYAN}Warm prefix enabled: {warm_tokens:,} tokens "
                       f"({config.warm_prefix_pct:.0%} of {max_shared:,} max tool+system){Colors.ENDC}")
        else:
            logger.info(f"{Colors.WARNING}Warm prefix disabled: no tool_tokens/system_tokens in traces{Colors.ENDC}")

    await orchestrator.run()

    # Save results
    save_results(orchestrator, config)

    # Generate graphs and index.html
    if not args.skip_graphs:
        try:
            generate_graphs(orchestrator, config)
        except Exception as e:
            logger.warning(f"Failed to generate graphs: {e}")

        try:
            import subprocess
            subprocess.run([
                sys.executable, "generate_index.py",
                config.output_dir
            ], check=True)
        except Exception as e:
            logger.warning(f"Failed to generate index.html: {e}")


if __name__ == "__main__":
    asyncio.run(main())
