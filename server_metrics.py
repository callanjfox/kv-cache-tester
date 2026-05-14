"""
Metrics collector for inference servers during benchmarks.
Polls /metrics endpoint and generates visualizations.
Supports vLLM and sglang backends (auto-detected from metrics prefix).
"""

import asyncio
import csv
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp
import matplotlib.pyplot as plt

logger = logging.getLogger("trace_replay")


@dataclass
class MetricsSnapshot:
    timestamp: float
    kv_cache_usage: float = 0.0
    cpu_kv_cache_usage: float = 0.0
    num_requests_running: int = 0
    num_requests_waiting: int = 0
    prefix_cache_hits: int = 0
    prefix_cache_queries: int = 0
    cpu_prefix_cache_hits: int = 0
    cpu_prefix_cache_queries: int = 0
    prompt_tokens: int = 0
    generation_tokens: int = 0
    num_preemptions: int = 0
    request_success: int = 0
    # KV offload transfer metrics (cumulative)
    kv_offload_bytes_gpu_to_cpu: float = 0.0
    kv_offload_bytes_cpu_to_gpu: float = 0.0
    kv_offload_time_gpu_to_cpu: float = 0.0
    kv_offload_time_cpu_to_gpu: float = 0.0
    # Prompt tokens by source (cumulative)
    prompt_tokens_local_compute: int = 0
    prompt_tokens_local_cache_hit: int = 0
    prompt_tokens_external_kv_transfer: int = 0
    # Prefill KV computed tokens (cumulative sum from histogram)
    prefill_kv_computed_tokens_sum: int = 0
    prefill_kv_computed_tokens_count: int = 0
    # SGLang PD KV transfer metrics (cumulative, from histograms _sum/_count)
    kv_transfer_latency_ms_sum: float = 0.0
    kv_transfer_latency_ms_count: int = 0
    kv_transfer_total_mb_sum: float = 0.0
    kv_transfer_total_mb_count: int = 0
    kv_transfer_speed_gb_s_sum: float = 0.0
    kv_transfer_speed_gb_s_count: int = 0
    # SGLang HiCache host-tier metrics
    hicache_host_used_tokens: int = 0
    hicache_host_total_tokens: int = 0
    # SGLang per-tier cache hit tokens (cumulative counters)
    cache_hit_tokens_l1: int = 0  # GPU/device cache
    cache_hit_tokens_l2: int = 0  # host/DRAM cache
    cache_hit_tokens_l3: int = 0  # storage (UMBP/SSD)
    cache_miss_tokens: int = 0


# =============================================================================
# Metrics Parsers — one per backend
# =============================================================================

def _get_value(text: str, pattern: str, default: float = 0.0) -> float:
    """Extract a gauge/counter value from Prometheus text using a regex."""
    match = re.search(pattern, text)
    return float(match.group(1)) if match else default


def _get_values(text: str, pattern: str) -> list[float]:
    """Extract all values matching a Prometheus text regex."""
    return [float(match.group(1)) for match in re.finditer(pattern, text)]


def _sum_values(text: str, pattern: str) -> float:
    """Sum all values matching a Prometheus text regex."""
    return sum(_get_values(text, pattern))


class VLLMMetricsParser:
    """Parse vLLM Prometheus metrics (prefix: vllm:)."""

    def parse(self, text: str) -> MetricsSnapshot:
        snapshot = MetricsSnapshot(timestamp=time.time())
        g = lambda p, d=0.0: _get_value(text, p, d)

        # KV cache usage (0-1 scale)
        snapshot.kv_cache_usage = g(r'vllm:gpu_cache_usage_perc\{[^}]*\}\s+([\d.e+-]+)')
        if snapshot.kv_cache_usage == 0.0:
            snapshot.kv_cache_usage = g(r'vllm:kv_cache_usage_perc\{[^}]*\}\s+([\d.e+-]+)')

        snapshot.cpu_kv_cache_usage = g(r'vllm:cpu_cache_usage_perc\{[^}]*\}\s+([\d.e+-]+)')

        snapshot.num_requests_running = int(g(r'vllm:num_requests_running\{[^}]*\}\s+([\d.e+-]+)'))
        snapshot.num_requests_waiting = int(g(r'vllm:num_requests_waiting\{[^}]*\}\s+([\d.e+-]+)'))

        snapshot.prefix_cache_hits = int(g(r'vllm:prefix_cache_hits_total\{[^}]*\}\s+([\d.e+-]+)'))
        snapshot.prefix_cache_queries = int(g(r'vllm:prefix_cache_queries_total\{[^}]*\}\s+([\d.e+-]+)'))

        snapshot.cpu_prefix_cache_hits = int(g(r'vllm:external_prefix_cache_hits_total\{[^}]*\}\s+([\d.e+-]+)'))
        snapshot.cpu_prefix_cache_queries = int(g(r'vllm:external_prefix_cache_queries_total\{[^}]*\}\s+([\d.e+-]+)'))

        snapshot.prompt_tokens = int(g(r'vllm:prompt_tokens_total\{[^}]*\}\s+([\d.e+-]+)'))
        snapshot.generation_tokens = int(g(r'vllm:generation_tokens_total\{[^}]*\}\s+([\d.e+-]+)'))

        snapshot.num_preemptions = int(g(r'vllm:num_preemptions_total\{[^}]*\}\s+([\d.e+-]+)'))

        for match in re.finditer(
            r'vllm:request_success_total\{[^}]*finished_reason="[^"]*"[^}]*\}\s+([\d.e+-]+)', text
        ):
            snapshot.request_success += int(float(match.group(1)))

        snapshot.kv_offload_bytes_gpu_to_cpu = g(r'vllm:kv_offload_total_bytes_total\{[^}]*transfer_type="GPU_to_CPU"[^}]*\}\s+([\d.e+-]+)')
        snapshot.kv_offload_bytes_cpu_to_gpu = g(r'vllm:kv_offload_total_bytes_total\{[^}]*transfer_type="CPU_to_GPU"[^}]*\}\s+([\d.e+-]+)')
        snapshot.kv_offload_time_gpu_to_cpu = g(r'vllm:kv_offload_total_time_total\{[^}]*transfer_type="GPU_to_CPU"[^}]*\}\s+([\d.e+-]+)')
        snapshot.kv_offload_time_cpu_to_gpu = g(r'vllm:kv_offload_total_time_total\{[^}]*transfer_type="CPU_to_GPU"[^}]*\}\s+([\d.e+-]+)')

        snapshot.prompt_tokens_local_compute = int(g(r'vllm:prompt_tokens_by_source_total\{[^}]*source="local_compute"[^}]*\}\s+([\d.e+-]+)'))
        snapshot.prompt_tokens_local_cache_hit = int(g(r'vllm:prompt_tokens_by_source_total\{[^}]*source="local_cache_hit"[^}]*\}\s+([\d.e+-]+)'))
        snapshot.prompt_tokens_external_kv_transfer = int(g(r'vllm:prompt_tokens_by_source_total\{[^}]*source="external_kv_transfer"[^}]*\}\s+([\d.e+-]+)'))

        snapshot.prefill_kv_computed_tokens_sum = int(g(r'vllm:request_prefill_kv_computed_tokens_sum\{[^}]*\}\s+([\d.e+-]+)'))
        snapshot.prefill_kv_computed_tokens_count = int(g(r'vllm:request_prefill_kv_computed_tokens_count\{[^}]*\}\s+([\d.e+-]+)'))

        return snapshot


class SGLangMetricsParser:
    """Parse sglang Prometheus metrics (prefix: sglang:)."""

    def parse(self, text: str) -> MetricsSnapshot:
        snapshot = MetricsSnapshot(timestamp=time.time())
        g = lambda p, d=0.0: _get_value(text, p, d)

        has_prefill_engine = 'engine_type="prefill"' in text
        has_decode_engine = 'engine_type="decode"' in text
        prefill_only = has_prefill_engine and not has_decode_engine
        decode_only = has_decode_engine and not has_prefill_engine

        # KV cache usage — sglang reports token_usage as a ratio (0-1)
        full_token_usage = g(r'sglang:full_token_usage\{[^}]*\}\s+([\d.e+-]+)')
        token_usage = g(r'sglang:token_usage\{[^}]*\}\s+([\d.e+-]+)')
        used = g(r'sglang:num_used_tokens\{[^}]*\}\s+([\d.e+-]+)')
        total = g(r'sglang:max_total_num_tokens\{[^}]*\}\s+([\d.e+-]+)')
        computed_token_usage = used / total if total > 0 else 0.0
        gpu_occupancy = g(r'sglang:gpu_kv_cache_occupancy\{[^}]*\}\s+([\d.e+-]+)')
        for value in (full_token_usage, token_usage, computed_token_usage, gpu_occupancy):
            if value > 0:
                snapshot.kv_cache_usage = value
                break

        host_used = g(r'sglang:hicache_host_used_tokens\{[^}]*\}\s+([\d.e+-]+)')
        host_total = g(r'sglang:hicache_host_total_tokens\{[^}]*\}\s+([\d.e+-]+)')
        snapshot.hicache_host_used_tokens = int(host_used)
        snapshot.hicache_host_total_tokens = int(host_total)
        if host_total > 0:
            snapshot.cpu_kv_cache_usage = host_used / host_total

        snapshot.num_requests_running = int(g(r'sglang:num_running_reqs\{[^}]*\}\s+([\d.e+-]+)'))
        snapshot.num_requests_waiting = int(g(r'sglang:num_queue_reqs\{[^}]*\}\s+([\d.e+-]+)'))

        raw_prompt_tokens = int(g(r'sglang:prompt_tokens_total\{[^}]*\}\s+([\d.e+-]+)'))
        raw_generation_tokens = int(g(r'sglang:generation_tokens_total\{[^}]*\}\s+([\d.e+-]+)'))
        snapshot.prompt_tokens = 0 if decode_only else raw_prompt_tokens
        snapshot.generation_tokens = 0 if prefill_only else raw_generation_tokens

        # Preemptions — sglang calls them "retractions"
        snapshot.num_preemptions = int(g(r'sglang:num_retracted_reqs\{[^}]*\}\s+([\d.e+-]+)'))

        snapshot.request_success = 0 if prefill_only else int(g(r'sglang:num_requests_total\{[^}]*\}\s+([\d.e+-]+)'))

        # Token source breakdown from realtime_tokens_total (cumulative)
        if not decode_only:
            snapshot.prompt_tokens_local_compute = int(g(
                r'sglang:realtime_tokens_total\{[^}]*mode="prefill_compute"[^}]*\}\s+([\d.e+-]+)'))
            realtime_cache_hit = int(g(
                r'sglang:realtime_tokens_total\{[^}]*mode="prefill_cache"[^}]*\}\s+([\d.e+-]+)'))
            device_cache_hit = int(g(
                r'sglang:cached_tokens_total\{[^}]*cache_source="device"[^}]*\}\s+([\d.e+-]+)'))
            host_cache_hit = int(g(
                r'sglang:cached_tokens_total\{[^}]*cache_source="host"[^}]*\}\s+([\d.e+-]+)'))

            if device_cache_hit > 0 or host_cache_hit > 0:
                snapshot.prompt_tokens_local_cache_hit = device_cache_hit
                snapshot.prompt_tokens_external_kv_transfer = host_cache_hit
            else:
                snapshot.prompt_tokens_local_cache_hit = realtime_cache_hit

            snapshot.cache_hit_tokens_l1 = int(_sum_values(
                text,
                r'sglang:cache_hit_tokens_l1_total\{[^}]*\}\s+([\d.e+-]+)',
            ))
            snapshot.cache_hit_tokens_l2 = int(_sum_values(
                text,
                r'sglang:cache_hit_tokens_l2_total\{[^}]*\}\s+([\d.e+-]+)',
            ))
            snapshot.cache_hit_tokens_l3 = int(_sum_values(
                text,
                r'sglang:cache_hit_tokens_l3_total\{[^}]*\}\s+([\d.e+-]+)',
            ))
            snapshot.cache_miss_tokens = int(_sum_values(
                text,
                r'sglang:cache_miss_tokens_total\{[^}]*\}\s+([\d.e+-]+)',
            ))
            if not any((
                snapshot.cache_hit_tokens_l1,
                snapshot.cache_hit_tokens_l2,
                snapshot.cache_hit_tokens_l3,
                snapshot.cache_miss_tokens,
            )):
                snapshot.cache_hit_tokens_l1 = device_cache_hit
                snapshot.cache_hit_tokens_l2 = host_cache_hit
                snapshot.cache_miss_tokens = snapshot.prompt_tokens_local_compute

        # Derive cumulative hits/queries from the per-source token counters.
        # This is the correct cumulative cache hit ratio — unlike sglang's
        # instantaneous `cache_hit_rate` gauge, which is 0 during decode-only
        # periods and thus yielded spurious 0% hit rates when sampled at
        # benchmark shutdown.
        snapshot.prefix_cache_hits = (
            snapshot.prompt_tokens_local_cache_hit
            + snapshot.prompt_tokens_external_kv_transfer
        )
        snapshot.prefix_cache_queries = (
            snapshot.prefix_cache_hits
            + snapshot.prompt_tokens_local_compute
        )

        snapshot.kv_offload_bytes_gpu_to_cpu = _sum_values(
            text,
            r'sglang:evicted_bytes_total\{[^}]*\}\s+([\d.e+-]+)',
        )
        snapshot.kv_offload_bytes_cpu_to_gpu = _sum_values(
            text,
            r'sglang:load_back_bytes_total\{[^}]*\}\s+([\d.e+-]+)',
        )
        snapshot.kv_transfer_latency_ms_sum = _sum_values(
            text,
            r'sglang:kv_transfer_latency_ms_sum\{[^}]*\}\s+([\d.e+-]+)',
        )
        snapshot.kv_transfer_latency_ms_count = int(_sum_values(
            text,
            r'sglang:kv_transfer_latency_ms_count\{[^}]*\}\s+([\d.e+-]+)',
        ))
        snapshot.kv_transfer_total_mb_sum = _sum_values(
            text,
            r'sglang:kv_transfer_total_mb_sum\{[^}]*\}\s+([\d.e+-]+)',
        )
        snapshot.kv_transfer_total_mb_count = int(_sum_values(
            text,
            r'sglang:kv_transfer_total_mb_count\{[^}]*\}\s+([\d.e+-]+)',
        ))
        snapshot.kv_transfer_speed_gb_s_sum = _sum_values(
            text,
            r'sglang:kv_transfer_speed_gb_s_sum\{[^}]*\}\s+([\d.e+-]+)',
        )
        snapshot.kv_transfer_speed_gb_s_count = int(_sum_values(
            text,
            r'sglang:kv_transfer_speed_gb_s_count\{[^}]*\}\s+([\d.e+-]+)',
        ))

        return snapshot


def detect_backend(text: str) -> str:
    """Very sophisticated way to auto-detect backend from metrics text."""
    if 'vllm:' in text:
        return 'vllm'
    elif 'sglang:' in text:
        return 'sglang'
    return 'unknown'


def get_parser(backend: str):
    """Get the appropriate parser for the backend."""
    if backend == 'sglang':
        return SGLangMetricsParser()
    return VLLMMetricsParser()  # default


@dataclass
class MetricsCollector:
    base_url: str | None = None
    poll_interval: float = 1.0
    metrics_endpoints: str | list[str] | None = None
    snapshots: list[MetricsSnapshot] = field(default_factory=list)
    _running: bool = False
    _task: asyncio.Task | None = None
    _metrics_urls: list[str] = field(default_factory=list, init=False)
    _parsers: dict[str, VLLMMetricsParser | SGLangMetricsParser] = field(default_factory=dict, init=False)
    _backends: dict[str, str] = field(default_factory=dict, init=False)
    gpu_transfer_collector: object = None

    def __post_init__(self) -> None:
        self._metrics_urls = self._normalize_metrics_endpoints(
            self.metrics_endpoints if self.metrics_endpoints else self.base_url
        )

    @staticmethod
    def _normalize_metrics_endpoints(endpoints: str | list[str] | None) -> list[str]:
        if endpoints is None:
            return []

        if isinstance(endpoints, str):
            entries = endpoints.split(',')
        else:
            entries = endpoints

        metrics_urls = []
        for entry in entries:
            endpoint = str(entry).strip().rstrip('/')
            if not endpoint:
                continue
            if not endpoint.endswith('/metrics'):
                endpoint = f"{endpoint}/metrics"
            metrics_urls.append(endpoint)
        return metrics_urls

    def describe_endpoints(self) -> str:
        return ', '.join(self._metrics_urls) if self._metrics_urls else '<none>'

    def _parse_metrics(self, text: str, metrics_url: str) -> MetricsSnapshot:
        """Parse Prometheus metrics text, auto-detecting backend on first call."""
        parser = self._parsers.get(metrics_url)
        if parser is None:
            backend = detect_backend(text)
            parser = get_parser(backend)
            self._parsers[metrics_url] = parser
            self._backends[metrics_url] = backend
            if backend != 'unknown':
                logger.info(f"Auto-detected metrics backend for {metrics_url}: {backend}")
        return parser.parse(text)

    async def _fetch_snapshot(
        self,
        session: aiohttp.ClientSession,
        metrics_url: str,
    ) -> MetricsSnapshot | None:
        try:
            async with session.get(metrics_url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status != 200:
                    logger.warning(f"Metrics poll error from {metrics_url}: HTTP {resp.status}")
                    return None
                text = await resp.text()
                return self._parse_metrics(text, metrics_url)
        except Exception as e:
            logger.warning(f"Metrics poll error from {metrics_url}: {e}")
            return None

    @staticmethod
    def _combine_snapshots(snapshots: list[MetricsSnapshot]) -> MetricsSnapshot:
        combined = MetricsSnapshot(timestamp=time.time())
        combined.kv_cache_usage = max(s.kv_cache_usage for s in snapshots)
        combined.cpu_kv_cache_usage = max(s.cpu_kv_cache_usage for s in snapshots)
        combined.num_requests_running = sum(s.num_requests_running for s in snapshots)
        combined.num_requests_waiting = sum(s.num_requests_waiting for s in snapshots)
        combined.prefix_cache_hits = sum(s.prefix_cache_hits for s in snapshots)
        combined.prefix_cache_queries = sum(s.prefix_cache_queries for s in snapshots)
        combined.cpu_prefix_cache_hits = sum(s.cpu_prefix_cache_hits for s in snapshots)
        combined.cpu_prefix_cache_queries = sum(s.cpu_prefix_cache_queries for s in snapshots)
        combined.prompt_tokens = sum(s.prompt_tokens for s in snapshots)
        combined.generation_tokens = sum(s.generation_tokens for s in snapshots)
        combined.num_preemptions = sum(s.num_preemptions for s in snapshots)
        combined.request_success = sum(s.request_success for s in snapshots)
        combined.kv_offload_bytes_gpu_to_cpu = sum(s.kv_offload_bytes_gpu_to_cpu for s in snapshots)
        combined.kv_offload_bytes_cpu_to_gpu = sum(s.kv_offload_bytes_cpu_to_gpu for s in snapshots)
        combined.kv_offload_time_gpu_to_cpu = sum(s.kv_offload_time_gpu_to_cpu for s in snapshots)
        combined.kv_offload_time_cpu_to_gpu = sum(s.kv_offload_time_cpu_to_gpu for s in snapshots)
        combined.prompt_tokens_local_compute = sum(s.prompt_tokens_local_compute for s in snapshots)
        combined.prompt_tokens_local_cache_hit = sum(s.prompt_tokens_local_cache_hit for s in snapshots)
        combined.prompt_tokens_external_kv_transfer = sum(s.prompt_tokens_external_kv_transfer for s in snapshots)
        combined.prefill_kv_computed_tokens_sum = sum(s.prefill_kv_computed_tokens_sum for s in snapshots)
        combined.prefill_kv_computed_tokens_count = sum(s.prefill_kv_computed_tokens_count for s in snapshots)
        combined.kv_transfer_latency_ms_sum = sum(s.kv_transfer_latency_ms_sum for s in snapshots)
        combined.kv_transfer_latency_ms_count = sum(s.kv_transfer_latency_ms_count for s in snapshots)
        combined.kv_transfer_total_mb_sum = sum(s.kv_transfer_total_mb_sum for s in snapshots)
        combined.kv_transfer_total_mb_count = sum(s.kv_transfer_total_mb_count for s in snapshots)
        combined.kv_transfer_speed_gb_s_sum = sum(s.kv_transfer_speed_gb_s_sum for s in snapshots)
        combined.kv_transfer_speed_gb_s_count = sum(s.kv_transfer_speed_gb_s_count for s in snapshots)
        combined.hicache_host_used_tokens = max(s.hicache_host_used_tokens for s in snapshots)
        combined.hicache_host_total_tokens = max(s.hicache_host_total_tokens for s in snapshots)
        combined.cache_hit_tokens_l1 = sum(s.cache_hit_tokens_l1 for s in snapshots)
        combined.cache_hit_tokens_l2 = sum(s.cache_hit_tokens_l2 for s in snapshots)
        combined.cache_hit_tokens_l3 = sum(s.cache_hit_tokens_l3 for s in snapshots)
        combined.cache_miss_tokens = sum(s.cache_miss_tokens for s in snapshots)
        return combined

    async def _poll_loop(self) -> None:
        """Background polling loop."""
        if not self._metrics_urls:
            logger.warning("No metrics endpoints configured")
            return

        async with aiohttp.ClientSession() as session:
            while self._running:
                results = await asyncio.gather(
                    *(self._fetch_snapshot(session, url) for url in self._metrics_urls),
                    return_exceptions=True,
                )
                snapshots = [s for s in results if isinstance(s, MetricsSnapshot)]
                if snapshots:
                    self.snapshots.append(self._combine_snapshots(snapshots))

                await asyncio.sleep(self.poll_interval)

    def start(self) -> None:
        """Start background metrics collection."""
        if self._running:
            return
        self._running = True
        self.snapshots = []
        self._task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        """Stop metrics collection."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def _trim_idle_prefix(self) -> None:
        """Drop leading snapshots where the server was idle (no running requests
        and no prompt tokens processed). Keeps plot x-axis starting at the first
        real activity instead of showing a long zero-flat prefix."""
        first_active = next(
            (
                i for i, s in enumerate(self.snapshots)
                if s.num_requests_running > 0 or s.prompt_tokens > 0
            ),
            None,
        )
        if first_active is not None and first_active > 0:
            dropped = first_active
            self.snapshots = self.snapshots[first_active:]
            logger.info(f"Trimmed {dropped} idle leading snapshots before output")

    def generate_plots(
        self,
        output_prefix: str = "metrics",
        client_metrics: list | None = None,
    ) -> None:
        """Generate visualization plots from collected metrics.

        Args:
            output_prefix: Prefix for output file names
            client_metrics: Optional list of RequestStats from benchmark clients
        """
        self._trim_idle_prefix()

        if len(self.snapshots) < 2:
            logger.warning("Not enough data points for plots")
            return

        # Convert to relative time (seconds from start)
        start_time = self.snapshots[0].timestamp
        times = [(s.timestamp - start_time) for s in self.snapshots]

        # Create figure with subplots
        num_rows = 6 if client_metrics else 4
        fig, axes = plt.subplots(num_rows, 2, figsize=(14, 4 * num_rows))
        fig.suptitle("Server Metrics During Benchmark", fontsize=14)

        # 1. KV Cache Usage vs Time
        ax = axes[0, 0]
        kv_usage = [min(s.kv_cache_usage * 100, 100.0) for s in self.snapshots]
        ax.scatter(times, kv_usage, alpha=0.15, s=2, c='blue')
        kv_window = min(50, len(kv_usage) // 10) if len(kv_usage) > 10 else 1
        if kv_window > 1:
            rolling_kv = [
                sum(kv_usage[max(0, i - kv_window):i + 1]) / len(kv_usage[max(0, i - kv_window):i + 1])
                for i in range(len(kv_usage))
            ]
            ax.plot(times, rolling_kv, 'b-', label=f'GPU (avg n={kv_window})', linewidth=2)
        else:
            ax.plot(times, kv_usage, 'b-', label='GPU', linewidth=2)
        # Add external cache if available
        cpu_kv_usage = [s.cpu_kv_cache_usage * 100 for s in self.snapshots]
        if any(v > 0 for v in cpu_kv_usage):
            ax.plot(times, cpu_kv_usage, 'r--', label='External', linewidth=1.5)
        ax.legend(fontsize=8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("KV Cache Usage (%)")
        ax.set_title("KV Cache Utilization Over Time")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)

        # 2. Running & Waiting Requests vs Time (smoothed + total)
        ax = axes[0, 1]
        running = [s.num_requests_running for s in self.snapshots]
        waiting = [s.num_requests_waiting for s in self.snapshots]
        total_queue = [r + w for r, w in zip(running, waiting)]
        q_window = min(30, len(running) // 10) if len(running) > 10 else 1
        if q_window > 1:
            rolling_running = [
                sum(running[max(0, i - q_window):i + 1]) / len(running[max(0, i - q_window):i + 1])
                for i in range(len(running))
            ]
            rolling_waiting = [
                sum(waiting[max(0, i - q_window):i + 1]) / len(waiting[max(0, i - q_window):i + 1])
                for i in range(len(waiting))
            ]
            rolling_total = [
                sum(total_queue[max(0, i - q_window):i + 1]) / len(total_queue[max(0, i - q_window):i + 1])
                for i in range(len(total_queue))
            ]
            ax.plot(times, rolling_running, 'g-', label=f'Running (avg n={q_window})', linewidth=1.5)
            ax.plot(times, rolling_waiting, 'r-', label=f'Waiting (avg n={q_window})', linewidth=1.5)
            ax.plot(times, rolling_total, 'b-', label=f'Total (avg n={q_window})', linewidth=1.5)
        else:
            ax.plot(times, running, 'g-', label='Running', linewidth=1.5)
            ax.plot(times, waiting, 'r-', label='Waiting', linewidth=1.5)
            ax.plot(times, total_queue, 'b-', label='Total', linewidth=1.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Requests")
        ax.set_title("Request Queue Depth")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 3. Cache Hit Rate vs Time (computed from deltas between polling intervals)
        ax = axes[1, 0]
        gpu_hit_rates = []
        ext_hit_rates = []
        combined_hit_rates = []
        has_ext_cache = any(s.cpu_prefix_cache_queries > 0 for s in self.snapshots)
        for i in range(1, len(self.snapshots)):
            # GPU (HBM) cache hit rate for this interval
            gpu_delta_hits = self.snapshots[i].prefix_cache_hits - self.snapshots[i-1].prefix_cache_hits
            gpu_delta_queries = self.snapshots[i].prefix_cache_queries - self.snapshots[i-1].prefix_cache_queries
            if gpu_delta_queries > 0:
                gpu_hit_rates.append(100.0 * gpu_delta_hits / gpu_delta_queries)
            else:
                gpu_hit_rates.append(gpu_hit_rates[-1] if gpu_hit_rates else 0)

            # External cache hit rate for this interval
            if has_ext_cache:
                ext_delta_hits = self.snapshots[i].cpu_prefix_cache_hits - self.snapshots[i-1].cpu_prefix_cache_hits
                ext_delta_queries = self.snapshots[i].cpu_prefix_cache_queries - self.snapshots[i-1].cpu_prefix_cache_queries
                if ext_delta_queries > 0:
                    ext_hit_rates.append(100.0 * ext_delta_hits / ext_delta_queries)
                else:
                    ext_hit_rates.append(ext_hit_rates[-1] if ext_hit_rates else 0)

                # Combined hit rate: (gpu_hits + ext_hits) / (gpu_queries + ext_queries)
                total_hits = gpu_delta_hits + ext_delta_hits
                total_queries = gpu_delta_queries + ext_delta_queries
                if total_queries > 0:
                    combined_hit_rates.append(100.0 * total_hits / total_queries)
                else:
                    combined_hit_rates.append(combined_hit_rates[-1] if combined_hit_rates else 0)

        # Rolling window size
        window = min(50, len(gpu_hit_rates) // 10) if len(gpu_hit_rates) > 10 else 1

        # Scatter plot for GPU (HBM) cache hit rate
        ax.scatter(times[1:], gpu_hit_rates, alpha=0.3, s=5, c='purple', label='GPU (HBM)')
        if window > 1:
            rolling_gpu = [
                sum(gpu_hit_rates[max(0, i - window):i + 1]) / len(gpu_hit_rates[max(0, i - window):i + 1])
                for i in range(len(gpu_hit_rates))
            ]
            ax.plot(times[1:], rolling_gpu, 'purple', linewidth=1.5, label=f'GPU avg (n={window})')

        # External cache scatter + rolling (if available)
        if has_ext_cache and ext_hit_rates:
            ax.scatter(times[1:], ext_hit_rates, alpha=0.3, s=5, c='orange', label='External')
            if window > 1:
                rolling_ext = [
                    sum(ext_hit_rates[max(0, i - window):i + 1]) / len(ext_hit_rates[max(0, i - window):i + 1])
                    for i in range(len(ext_hit_rates))
                ]
                ax.plot(times[1:], rolling_ext, 'orange', linewidth=1.5, label=f'External avg (n={window})')

            # Combined/total hit rate (only if external exists)
            ax.scatter(times[1:], combined_hit_rates, alpha=0.2, s=3, c='green', label='Combined')
            if window > 1:
                rolling_combined = [
                    sum(combined_hit_rates[max(0, i - window):i + 1]) / len(combined_hit_rates[max(0, i - window):i + 1])
                    for i in range(len(combined_hit_rates))
                ]
                ax.plot(times[1:], rolling_combined, 'green', linewidth=2, label=f'Combined avg (n={window})')

        ax.legend(loc='best', fontsize=8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Hit Rate (%)")
        ax.set_title("Prefix Cache Hit Rate Per Interval (tokens hit / tokens queried)")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)

        # 4. Throughput vs Time (tokens/sec) with rolling average — decode + total
        ax = axes[1, 1]
        decode_throughputs = []
        total_throughputs = []
        for i in range(1, len(self.snapshots)):
            delta_gen = self.snapshots[i].generation_tokens - self.snapshots[i-1].generation_tokens
            delta_prompt = self.snapshots[i].prompt_tokens - self.snapshots[i-1].prompt_tokens
            delta_time = self.snapshots[i].timestamp - self.snapshots[i-1].timestamp
            if delta_time > 0:
                decode_throughputs.append(delta_gen / delta_time)
                total_throughputs.append((delta_gen + delta_prompt) / delta_time)
            else:
                decode_throughputs.append(0)
                total_throughputs.append(0)
        # Cumulative running average total throughput (total tokens / elapsed time)
        cumulative_total_avg = []
        t0 = self.snapshots[0].timestamp
        tokens0 = self.snapshots[0].generation_tokens + self.snapshots[0].prompt_tokens
        for i in range(1, len(self.snapshots)):
            elapsed = self.snapshots[i].timestamp - t0
            total_tokens = (self.snapshots[i].generation_tokens + self.snapshots[i].prompt_tokens) - tokens0
            cumulative_total_avg.append(total_tokens / elapsed if elapsed > 0 else 0)

        window = min(30, len(decode_throughputs) // 10) if len(decode_throughputs) > 10 else 1
        if window > 1:
            rolling_decode = [
                sum(decode_throughputs[max(0, i - window):i + 1]) / len(decode_throughputs[max(0, i - window):i + 1])
                for i in range(len(decode_throughputs))
            ]
            rolling_total = [
                sum(total_throughputs[max(0, i - window):i + 1]) / len(total_throughputs[max(0, i - window):i + 1])
                for i in range(len(total_throughputs))
            ]
            ax.plot(times[1:], rolling_total, 'steelblue', linewidth=1.5, label=f'Total (avg n={window})')
            ax.plot(times[1:], rolling_decode, 'orange', linewidth=1.5, label=f'Decode (avg n={window})')
            ax.legend(fontsize=8)
        else:
            ax.plot(times[1:], total_throughputs, 'steelblue', linewidth=1, alpha=0.8, label='Total')
            ax.plot(times[1:], decode_throughputs, 'orange', linewidth=1, alpha=0.8, label='Decode')
            ax.legend(fontsize=8)
        ax.plot(times[1:], cumulative_total_avg, 'red', linewidth=2, label='Total Running Avg')
        ax.legend(fontsize=8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Tokens/sec")
        ax.set_title("Throughput (Total & Decode)")
        ax.grid(True, alpha=0.3)

        # 5. KV Transfer Rate
        ax = axes[2, 0]
        has_pd_transfer = any(s.kv_transfer_total_mb_count > 0 for s in self.snapshots)
        if has_pd_transfer:
            interval_speeds = []
            interval_mb = []
            for i in range(1, len(self.snapshots)):
                dc = self.snapshots[i].kv_transfer_speed_gb_s_count - self.snapshots[i-1].kv_transfer_speed_gb_s_count
                ds = self.snapshots[i].kv_transfer_speed_gb_s_sum - self.snapshots[i-1].kv_transfer_speed_gb_s_sum
                dm = self.snapshots[i].kv_transfer_total_mb_sum - self.snapshots[i-1].kv_transfer_total_mb_sum
                interval_speeds.append(ds / dc if dc > 0 else 0)
                interval_mb.append(dm)
            if any(v > 0 for v in interval_speeds):
                ax.scatter(times[1:], interval_speeds, alpha=0.15, s=3, c='blue')
                xfer_window = max(3, min(30, len(interval_speeds) // 5)) if len(interval_speeds) >= 3 else len(interval_speeds)
                if xfer_window > 1:
                    rolling = [
                        sum(interval_speeds[max(0, i - xfer_window):i + 1]) / len(interval_speeds[max(0, i - xfer_window):i + 1])
                        for i in range(len(interval_speeds))
                    ]
                    ax.plot(times[1:], rolling, 'b-', linewidth=1.5, label=f'Rolling avg (n={xfer_window})')
                ax.legend(fontsize=8)
                ax2 = ax.twinx()
                width = max(0.5, (times[-1] - times[0]) / max(1, len(times)) * 0.8)
                ax2.bar(times[1:], interval_mb, width=width, alpha=0.15, color='orange', label='MB transferred')
                ax2.set_ylabel("MB/interval", color='orange')
                ax2.legend(fontsize=7, loc='upper left')
            ax.set_ylabel("Avg Speed (GB/s)")
            ax.set_title("KV Transfer Speed (PD Disagg)")
        else:
            gpu_to_cpu_rates = []
            cpu_to_gpu_rates = []
            for i in range(1, len(self.snapshots)):
                dt = self.snapshots[i].timestamp - self.snapshots[i-1].timestamp
                if dt > 0:
                    delta_g2c = self.snapshots[i].kv_offload_bytes_gpu_to_cpu - self.snapshots[i-1].kv_offload_bytes_gpu_to_cpu
                    delta_c2g = self.snapshots[i].kv_offload_bytes_cpu_to_gpu - self.snapshots[i-1].kv_offload_bytes_cpu_to_gpu
                    gpu_to_cpu_rates.append(delta_g2c / dt / 1e6)  # MB/s
                    cpu_to_gpu_rates.append(delta_c2g / dt / 1e6)  # MB/s
                else:
                    gpu_to_cpu_rates.append(0)
                    cpu_to_gpu_rates.append(0)
            if any(r > 0 for r in gpu_to_cpu_rates) or any(r > 0 for r in cpu_to_gpu_rates):
                ax.scatter(times[1:], gpu_to_cpu_rates, alpha=0.15, s=3, c='blue')
                ax.scatter(times[1:], cpu_to_gpu_rates, alpha=0.15, s=3, c='red')
                xfer_window = min(30, len(gpu_to_cpu_rates) // 10) if len(gpu_to_cpu_rates) > 10 else 1
                if xfer_window > 1:
                    rolling_g2c = [
                        sum(gpu_to_cpu_rates[max(0, i - xfer_window):i + 1]) / len(gpu_to_cpu_rates[max(0, i - xfer_window):i + 1])
                        for i in range(len(gpu_to_cpu_rates))
                    ]
                    rolling_c2g = [
                        sum(cpu_to_gpu_rates[max(0, i - xfer_window):i + 1]) / len(cpu_to_gpu_rates[max(0, i - xfer_window):i + 1])
                        for i in range(len(cpu_to_gpu_rates))
                    ]
                    ax.plot(times[1:], rolling_g2c, 'b-', linewidth=1.5, label=f'GPU→CPU (avg n={xfer_window})')
                    ax.plot(times[1:], rolling_c2g, 'r-', linewidth=1.5, label=f'CPU→GPU (avg n={xfer_window})')
                else:
                    ax.plot(times[1:], gpu_to_cpu_rates, 'b-', linewidth=1, alpha=0.8, label='GPU→CPU')
                    ax.plot(times[1:], cpu_to_gpu_rates, 'r-', linewidth=1, alpha=0.8, label='CPU→GPU')
                ax.legend(fontsize=8)
            ax.set_ylabel("Transfer Rate (MB/s)")
            ax.set_title("KV Offload Transfer Rate")
        ax.set_xlabel("Time (s)")
        ax.grid(True, alpha=0.3)

        # 6. Prompt Token Sources Over Time (cumulative percentage)
        ax = axes[2, 1]
        initial = self.snapshots[0]
        cum_compute_pct = []
        cum_cache_pct = []
        cum_ext_pct = []
        for s in self.snapshots:
            c = s.prompt_tokens_local_compute - initial.prompt_tokens_local_compute
            h = s.prompt_tokens_local_cache_hit - initial.prompt_tokens_local_cache_hit
            e = s.prompt_tokens_external_kv_transfer - initial.prompt_tokens_external_kv_transfer
            total = c + h + e
            if total > 0:
                cum_compute_pct.append(100.0 * c / total)
                cum_cache_pct.append(100.0 * h / total)
                cum_ext_pct.append(100.0 * e / total)
            else:
                cum_compute_pct.append(0)
                cum_cache_pct.append(0)
                cum_ext_pct.append(0)
        if any(v > 0 for v in cum_compute_pct):
            ax.stackplot(times, cum_compute_pct, cum_cache_pct, cum_ext_pct,
                        labels=['Prefill', 'HBM Cache Hit', 'Offload Cache Hit'],
                        colors=['coral', 'steelblue', 'mediumseagreen'], alpha=0.8)
            ax.legend(fontsize=8, loc='lower left')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("% of Prefill Tokens")
        ax.set_title("Cumulative Prefill Token Source Breakdown")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)

        # 7. HiCache / Cumulative KV Offload Transfers
        initial = self.snapshots[0]
        has_hicache = any(s.hicache_host_total_tokens > 0 for s in self.snapshots)
        has_tier_cache = any(
            s.cache_hit_tokens_l1 > 0
            or s.cache_hit_tokens_l2 > 0
            or s.cache_hit_tokens_l3 > 0
            or s.cache_miss_tokens > 0
            for s in self.snapshots
        )

        ax = axes[3, 0]
        if has_hicache:
            host_usage = [
                100.0 * s.hicache_host_used_tokens / s.hicache_host_total_tokens
                if s.hicache_host_total_tokens > 0 else 0.0
                for s in self.snapshots
            ]
            host_used = [s.hicache_host_used_tokens / 1e6 for s in self.snapshots]
            ax.plot(times, host_usage, 'g-', linewidth=1.5, label='Host tier usage')
            ax.fill_between(times, host_usage, alpha=0.15, color='green')
            ax2 = ax.twinx()
            ax2.plot(times, host_used, 'b--', linewidth=1, alpha=0.7, label='Used tokens')
            ax2.set_ylabel("Used Tokens (M)", color='blue')
            ax2.tick_params(axis='y', labelcolor='blue')
            ax.set_ylabel("Host KV Usage (%)", color='green')
            ax.tick_params(axis='y', labelcolor='green')
            ax.set_title("HiCache Host-Tier Usage")
        else:
            cum_g2c = [(s.kv_offload_bytes_gpu_to_cpu - initial.kv_offload_bytes_gpu_to_cpu) / 1e9
                        for s in self.snapshots]
            if any(v > 0 for v in cum_g2c):
                ax.plot(times, cum_g2c, 'b-', linewidth=1.5)
                ax.fill_between(times, cum_g2c, alpha=0.2, color='blue')
            ax.set_ylabel("Cumulative Transfer (GB)")
            ax.set_title("KV Offload: GPU → CPU (Cumulative)")
        ax.set_xlabel("Time (s)")
        ax.grid(True, alpha=0.3)

        ax = axes[3, 1]
        if has_tier_cache:
            l1 = [max(0, s.cache_hit_tokens_l1 - initial.cache_hit_tokens_l1) for s in self.snapshots]
            l2 = [max(0, s.cache_hit_tokens_l2 - initial.cache_hit_tokens_l2) for s in self.snapshots]
            l3 = [max(0, s.cache_hit_tokens_l3 - initial.cache_hit_tokens_l3) for s in self.snapshots]
            miss = [max(0, s.cache_miss_tokens - initial.cache_miss_tokens) for s in self.snapshots]
            ax.stackplot(
                times, l1, l2, l3, miss,
                labels=['L1 GPU', 'L2 Host', 'L3 Storage', 'Miss'],
                colors=['steelblue', 'mediumseagreen', 'gold', 'coral'],
                alpha=0.75,
            )
            ax.legend(fontsize=8, loc='upper left')
            ax.set_ylabel("Cumulative Tokens")
            ax.set_title("Cache Hit/Miss Breakdown by Tier")
        else:
            cum_c2g = [(s.kv_offload_bytes_cpu_to_gpu - initial.kv_offload_bytes_cpu_to_gpu) / 1e9
                        for s in self.snapshots]
            if any(v > 0 for v in cum_c2g):
                ax.plot(times, cum_c2g, 'r-', linewidth=1.5)
                ax.fill_between(times, cum_c2g, alpha=0.2, color='red')
            ax.set_ylabel("Cumulative Transfer (GB)")
            ax.set_title("KV Offload: CPU → GPU (Cumulative)")
        ax.set_xlabel("Time (s)")
        ax.grid(True, alpha=0.3)

        # 8 & 9. Client metrics plots (TTFT and Latency vs Time)
        if client_metrics and len(client_metrics) > 0:
            # Sort by start time
            sorted_metrics = sorted(client_metrics, key=lambda x: x.start_time_ms)
            # Convert to relative time (seconds from first request)
            first_start = sorted_metrics[0].start_time_ms
            request_times = [(m.start_time_ms - first_start) / 1000.0 for m in sorted_metrics]
            ttfts = [m.ttft_ms for m in sorted_metrics]
            latencies = [m.latency_ms for m in sorted_metrics]

            # 8. TTFT vs Time
            ax = axes[4, 0]
            ax.scatter(request_times, ttfts, alpha=0.3, s=5, c='blue')
            # Add rolling average
            window = min(50, len(ttfts) // 10) if len(ttfts) > 10 else 1
            if window > 1:
                rolling_ttft = [
                    sum(ttfts[max(0, i - window):i + 1]) / len(ttfts[max(0, i - window):i + 1])
                    for i in range(len(ttfts))
                ]
                ax.plot(request_times, rolling_ttft, 'r-', linewidth=1.5, label=f'Rolling avg (n={window})')
                ax.legend()
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("TTFT (ms)")
            ax.set_title("Time to First Token vs Time")
            ax.grid(True, alpha=0.3)

            # 9. Latency vs Time
            ax = axes[4, 1]
            ax.scatter(request_times, latencies, alpha=0.3, s=5, c='green')
            # Add rolling average
            if window > 1:
                rolling_latency = [
                    sum(latencies[max(0, i - window):i + 1]) / len(latencies[max(0, i - window):i + 1])
                    for i in range(len(latencies))
                ]
                ax.plot(request_times, rolling_latency, 'r-', linewidth=1.5, label=f'Rolling avg (n={window})')
                ax.legend()
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Latency (ms)")
            ax.set_title("Request Latency vs Time")
            ax.grid(True, alpha=0.3)

            # 10. Interactivity (1/TPOT = tokens/sec) vs Time
            ax = axes[5, 0]
            # Filter out zero TPOT values to avoid division by zero
            tpots = [m.tpot_ms for m in sorted_metrics]
            interactivity = [1000.0 / t if t > 0 else 0 for t in tpots]  # Convert to tokens/sec
            ax.scatter(request_times, interactivity, alpha=0.3, s=5, c='purple')
            # Add rolling average
            if window > 1:
                rolling_inter = [
                    sum(interactivity[max(0, i - window):i + 1]) / len(interactivity[max(0, i - window):i + 1])
                    for i in range(len(interactivity))
                ]
                ax.plot(request_times, rolling_inter, 'r-', linewidth=1.5, label=f'Rolling avg (n={window})')
                ax.legend()
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Interactivity (tokens/sec)")
            ax.set_title("Decode Speed (1/TPOT) vs Time")
            ax.grid(True, alpha=0.3)

            # 11. Preemptions over time
            ax = axes[5, 1]
            preemption_rates = []
            for i in range(1, len(self.snapshots)):
                dt = self.snapshots[i].timestamp - self.snapshots[i-1].timestamp
                delta = self.snapshots[i].num_preemptions - self.snapshots[i-1].num_preemptions
                preemption_rates.append(delta / dt if dt > 0 else 0)
            if any(r > 0 for r in preemption_rates):
                ax.scatter(times[1:], preemption_rates, alpha=0.15, s=3, c='red')
                preempt_window = min(30, len(preemption_rates) // 10) if len(preemption_rates) > 10 else 1
                if preempt_window > 1:
                    rolling_preempt = [
                        sum(preemption_rates[max(0, i - preempt_window):i + 1]) / len(preemption_rates[max(0, i - preempt_window):i + 1])
                        for i in range(len(preemption_rates))
                    ]
                    ax.plot(times[1:], rolling_preempt, 'r-', linewidth=1.5, label=f'Rolling avg (n={preempt_window})')
                # Cumulative on secondary axis
                ax2 = ax.twinx()
                cumulative = [self.snapshots[i].num_preemptions - self.snapshots[0].num_preemptions
                              for i in range(1, len(self.snapshots))]
                ax2.plot(times[1:], cumulative, 'b--', linewidth=1, alpha=0.5, label='Cumulative')
                ax2.set_ylabel("Cumulative Preemptions", color='blue')
                ax2.tick_params(axis='y', labelcolor='blue')
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Preemptions/sec", color='red')
            ax.tick_params(axis='y', labelcolor='red')
            ax.set_title("Preemptions Over Time")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_prefix}_plots.png", dpi=150)
        logger.info(f"Saved plots to {output_prefix}_plots.png")
        plt.close()

        # Also generate a summary
        self._print_summary()

    def _print_summary(self) -> None:
        """Print summary statistics."""
        if len(self.snapshots) < 2:
            return

        duration = self.snapshots[-1].timestamp - self.snapshots[0].timestamp
        total_gen_tokens = self.snapshots[-1].generation_tokens - self.snapshots[0].generation_tokens
        total_prompt_tokens = self.snapshots[-1].prompt_tokens - self.snapshots[0].prompt_tokens

        final = self.snapshots[-1]
        initial = self.snapshots[0]

        lines = [
            "=" * 60,
            "SERVER METRICS SUMMARY",
            "=" * 60,
            f"Duration: {duration:.1f}s",
            f"Total prompt tokens: {total_prompt_tokens:,}",
            f"Total generation tokens: {total_gen_tokens:,}",
            f"Avg generation throughput: {total_gen_tokens/duration:.1f} tok/s",
            f"Peak KV cache usage: {max(s.kv_cache_usage for s in self.snapshots)*100:.1f}%",
            f"Peak running requests: {max(s.num_requests_running for s in self.snapshots)}",
            f"Peak waiting requests: {max(s.num_requests_waiting for s in self.snapshots)}",
            f"Total preemptions: {final.num_preemptions - initial.num_preemptions}",
        ]

        if final.prefix_cache_queries > initial.prefix_cache_queries:
            delta_hits = final.prefix_cache_hits - initial.prefix_cache_hits
            delta_queries = final.prefix_cache_queries - initial.prefix_cache_queries
            hit_rate = 100.0 * delta_hits / delta_queries
            lines.append(f"Overall GPU cache hit rate: {hit_rate:.1f}%")
            lines.append(f"  - Cache hits: {delta_hits:,} tokens")
            lines.append(f"  - Cache queries: {delta_queries:,} tokens")

        if final.cpu_prefix_cache_queries > initial.cpu_prefix_cache_queries:
            cpu_delta_hits = final.cpu_prefix_cache_hits - initial.cpu_prefix_cache_hits
            cpu_delta_queries = final.cpu_prefix_cache_queries - initial.cpu_prefix_cache_queries
            cpu_hit_rate = 100.0 * cpu_delta_hits / cpu_delta_queries
            lines.append(f"Overall external cache hit rate: {cpu_hit_rate:.1f}%")
            lines.append(f"  - Cache hits: {cpu_delta_hits:,} tokens")
            lines.append(f"  - Cache queries: {cpu_delta_queries:,} tokens")

        total_compute = final.prompt_tokens_local_compute - initial.prompt_tokens_local_compute
        total_cache_hit = final.prompt_tokens_local_cache_hit - initial.prompt_tokens_local_cache_hit
        total_ext = final.prompt_tokens_external_kv_transfer - initial.prompt_tokens_external_kv_transfer
        total_by_source = total_compute + total_cache_hit + total_ext
        if total_by_source > 0:
            lines.append(f"Prompt token sources:")
            lines.append(f"  - Prefill:            {total_compute:>12,} ({100*total_compute/total_by_source:.1f}%)")
            lines.append(f"  - HBM cache hit:      {total_cache_hit:>12,} ({100*total_cache_hit/total_by_source:.1f}%)")
            lines.append(f"  - Offload cache hit:  {total_ext:>12,} ({100*total_ext/total_by_source:.1f}%)")

        g2c_bytes = final.kv_offload_bytes_gpu_to_cpu - initial.kv_offload_bytes_gpu_to_cpu
        c2g_bytes = final.kv_offload_bytes_cpu_to_gpu - initial.kv_offload_bytes_cpu_to_gpu
        g2c_time = final.kv_offload_time_gpu_to_cpu - initial.kv_offload_time_gpu_to_cpu
        c2g_time = final.kv_offload_time_cpu_to_gpu - initial.kv_offload_time_cpu_to_gpu
        if g2c_bytes > 0 or c2g_bytes > 0:
            lines.append(f"KV offload transfers:")
            lines.append(f"  GPU→CPU: {g2c_bytes/1e9:.2f} GB in {g2c_time:.2f}s ({g2c_bytes/g2c_time/1e9:.1f} GB/s)" if g2c_time > 0 else f"  GPU→CPU: {g2c_bytes/1e9:.2f} GB")
            lines.append(f"  CPU→GPU: {c2g_bytes/1e9:.2f} GB in {c2g_time:.2f}s ({c2g_bytes/c2g_time/1e9:.1f} GB/s)" if c2g_time > 0 else f"  CPU→GPU: {c2g_bytes/1e9:.2f} GB")

        kv_transfer_count = final.kv_transfer_total_mb_count - initial.kv_transfer_total_mb_count
        if kv_transfer_count > 0:
            kv_transfer_mb = final.kv_transfer_total_mb_sum - initial.kv_transfer_total_mb_sum
            speed_count = final.kv_transfer_speed_gb_s_count - initial.kv_transfer_speed_gb_s_count
            speed_sum = final.kv_transfer_speed_gb_s_sum - initial.kv_transfer_speed_gb_s_sum
            latency_count = final.kv_transfer_latency_ms_count - initial.kv_transfer_latency_ms_count
            latency_sum = final.kv_transfer_latency_ms_sum - initial.kv_transfer_latency_ms_sum
            lines.append(f"SGLang PD KV transfers:")
            lines.append(f"  Total: {kv_transfer_mb:.1f} MB across {kv_transfer_count:,} transfers")
            if speed_count > 0:
                lines.append(f"  Avg speed: {speed_sum/speed_count:.2f} GB/s")
            if latency_count > 0:
                lines.append(f"  Avg latency: {latency_sum/latency_count:.2f} ms")

        if final.hicache_host_total_tokens > 0:
            host_usage = 100.0 * final.hicache_host_used_tokens / final.hicache_host_total_tokens
            lines.append(f"HiCache host tier: {final.hicache_host_used_tokens:,}/{final.hicache_host_total_tokens:,} tokens ({host_usage:.1f}%)")

        delta_l1 = final.cache_hit_tokens_l1 - initial.cache_hit_tokens_l1
        delta_l2 = final.cache_hit_tokens_l2 - initial.cache_hit_tokens_l2
        delta_l3 = final.cache_hit_tokens_l3 - initial.cache_hit_tokens_l3
        delta_miss = final.cache_miss_tokens - initial.cache_miss_tokens
        tier_total = delta_l1 + delta_l2 + delta_l3 + delta_miss
        if tier_total > 0:
            lines.append(f"Cache tier token breakdown:")
            lines.append(f"  - L1 GPU:     {delta_l1:>12,} ({100*delta_l1/tier_total:.1f}%)")
            lines.append(f"  - L2 Host:    {delta_l2:>12,} ({100*delta_l2/tier_total:.1f}%)")
            lines.append(f"  - L3 Storage: {delta_l3:>12,} ({100*delta_l3/tier_total:.1f}%)")
            lines.append(f"  - Miss:       {delta_miss:>12,} ({100*delta_miss/tier_total:.1f}%)")

        delta_kv_sum = final.prefill_kv_computed_tokens_sum - initial.prefill_kv_computed_tokens_sum
        delta_kv_count = final.prefill_kv_computed_tokens_count - initial.prefill_kv_computed_tokens_count
        if delta_kv_count > 0:
            lines.append(f"Prefill KV computed tokens (excluding cached):")
            lines.append(f"  Total: {delta_kv_sum:,} tokens across {delta_kv_count:,} requests")
            lines.append(f"  Avg per request: {delta_kv_sum/delta_kv_count:.0f} tokens")

        lines.append("=" * 60)
        logger.info("\n" + "\n".join(lines))

    def export_csv(
        self,
        output_prefix: str = "metrics",
        client_metrics: list | None = None,
    ) -> None:
        """Export all time series data to CSV files.

        Args:
            output_prefix: Prefix for output file names
            client_metrics: Optional list of RequestStats from benchmark clients

        Generates:
            - {output_prefix}_server_metrics.csv: vLLM server metrics over time
            - {output_prefix}_gpu_transfer.csv: GPU PCIe transfer stats
            - {output_prefix}_client_metrics.csv: Per-request client metrics (if provided)
        """
        self._trim_idle_prefix()

        output_dir = Path(output_prefix).parent
        if output_dir and not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Export server metrics (from /metrics endpoint)
        if self.snapshots:
            server_csv = f"{output_prefix}_server_metrics.csv"
            start_time = self.snapshots[0].timestamp

            with open(server_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                # Header
                writer.writerow([
                    'timestamp_sec',
                    'relative_time_sec',
                    'kv_cache_usage_pct',
                    'cpu_kv_cache_usage_pct',
                    'num_requests_running',
                    'num_requests_waiting',
                    'prefix_cache_hits',
                    'prefix_cache_queries',
                    'cpu_prefix_cache_hits',
                    'cpu_prefix_cache_queries',
                    'prompt_tokens_total',
                    'generation_tokens_total',
                    'num_preemptions_total',
                    'request_success_total',
                    # KV offload metrics
                    'kv_offload_bytes_gpu_to_cpu',
                    'kv_offload_bytes_cpu_to_gpu',
                    'kv_offload_time_gpu_to_cpu',
                    'kv_offload_time_cpu_to_gpu',
                    # Prompt tokens by source
                    'prompt_tokens_local_compute',
                    'prompt_tokens_local_cache_hit',
                    'prompt_tokens_external_kv_transfer',
                    # Prefill KV computed
                    'prefill_kv_computed_tokens_sum',
                    'prefill_kv_computed_tokens_count',
                    # SGLang PD KV transfer histograms
                    'kv_transfer_latency_ms_sum',
                    'kv_transfer_latency_ms_count',
                    'kv_transfer_total_mb_sum',
                    'kv_transfer_total_mb_count',
                    'kv_transfer_speed_gb_s_sum',
                    'kv_transfer_speed_gb_s_count',
                    # SGLang HiCache / cache tiers
                    'hicache_host_used_tokens',
                    'hicache_host_total_tokens',
                    'cache_hit_tokens_l1',
                    'cache_hit_tokens_l2',
                    'cache_hit_tokens_l3',
                    'cache_miss_tokens',
                    # Computed per-interval metrics
                    'interval_cache_hit_rate_pct',
                    'interval_throughput_tok_per_sec',
                ])

                for i, s in enumerate(self.snapshots):
                    relative_time = s.timestamp - start_time

                    # Compute per-interval metrics
                    cache_hit_rate = 0.0
                    throughput = 0.0
                    if i > 0:
                        prev = self.snapshots[i - 1]
                        delta_hits = s.prefix_cache_hits - prev.prefix_cache_hits
                        delta_queries = s.prefix_cache_queries - prev.prefix_cache_queries
                        if delta_queries > 0:
                            cache_hit_rate = 100.0 * delta_hits / delta_queries

                        delta_gen = s.generation_tokens - prev.generation_tokens
                        delta_time = s.timestamp - prev.timestamp
                        if delta_time > 0:
                            throughput = delta_gen / delta_time

                    writer.writerow([
                        f"{s.timestamp:.3f}",
                        f"{relative_time:.3f}",
                        f"{s.kv_cache_usage * 100:.2f}",
                        f"{s.cpu_kv_cache_usage * 100:.2f}",
                        s.num_requests_running,
                        s.num_requests_waiting,
                        s.prefix_cache_hits,
                        s.prefix_cache_queries,
                        s.cpu_prefix_cache_hits,
                        s.cpu_prefix_cache_queries,
                        s.prompt_tokens,
                        s.generation_tokens,
                        s.num_preemptions,
                        s.request_success,
                        f"{s.kv_offload_bytes_gpu_to_cpu:.0f}",
                        f"{s.kv_offload_bytes_cpu_to_gpu:.0f}",
                        f"{s.kv_offload_time_gpu_to_cpu:.6f}",
                        f"{s.kv_offload_time_cpu_to_gpu:.6f}",
                        s.prompt_tokens_local_compute,
                        s.prompt_tokens_local_cache_hit,
                        s.prompt_tokens_external_kv_transfer,
                        s.prefill_kv_computed_tokens_sum,
                        s.prefill_kv_computed_tokens_count,
                        f"{s.kv_transfer_latency_ms_sum:.6f}",
                        s.kv_transfer_latency_ms_count,
                        f"{s.kv_transfer_total_mb_sum:.6f}",
                        s.kv_transfer_total_mb_count,
                        f"{s.kv_transfer_speed_gb_s_sum:.6f}",
                        s.kv_transfer_speed_gb_s_count,
                        s.hicache_host_used_tokens,
                        s.hicache_host_total_tokens,
                        s.cache_hit_tokens_l1,
                        s.cache_hit_tokens_l2,
                        s.cache_hit_tokens_l3,
                        s.cache_miss_tokens,
                        f"{cache_hit_rate:.2f}",
                        f"{throughput:.2f}",
                    ])

            logger.info(f"Exported server metrics to {server_csv}")

        # 2. Export GPU transfer stats (DEPRECATED - kept for backward compat)
        if self.gpu_transfer_collector and self.gpu_transfer_collector.snapshots:
            gpu_csv = f"{output_prefix}_gpu_transfer.csv"
            gpu_snaps = self.gpu_transfer_collector.snapshots
            gpu_start = gpu_snaps[0].timestamp

            with open(gpu_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp_sec',
                    'relative_time_sec',
                    'gpu_id',
                    'tx_pci_mb_per_sec',
                    'rx_pci_mb_per_sec',
                    'cumulative_tx_gb',
                    'cumulative_rx_gb',
                ])

                cumulative_tx = 0.0
                cumulative_rx = 0.0
                for i, s in enumerate(gpu_snaps):
                    relative_time = s.timestamp - gpu_start
                    if i > 0:
                        dt = s.timestamp - gpu_snaps[i - 1].timestamp
                        cumulative_tx += s.tx_pci * dt / 1024  # MB to GB
                        cumulative_rx += s.rx_pci * dt / 1024

                    writer.writerow([
                        f"{s.timestamp:.3f}",
                        f"{relative_time:.3f}",
                        s.gpu_id,
                        f"{s.tx_pci:.2f}",
                        f"{s.rx_pci:.2f}",
                        f"{cumulative_tx:.4f}",
                        f"{cumulative_rx:.4f}",
                    ])

            logger.info(f"Exported GPU transfer metrics to {gpu_csv}")

        # 3. Export client metrics (per-request stats)
        if client_metrics and len(client_metrics) > 0:
            client_csv = f"{output_prefix}_client_metrics.csv"
            sorted_metrics = sorted(client_metrics, key=lambda x: x.start_time_ms)
            first_start = sorted_metrics[0].start_time_ms

            with open(client_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'start_time_ms',
                    'relative_time_sec',
                    'ttft_ms',
                    'tpot_ms',
                    'latency_ms',
                    'input_num_turns',
                    'input_num_tokens',
                    'output_num_tokens',
                    'output_num_chunks',
                    'output_num_first_chunk_tokens',
                    'approx_cached_percent',
                    'conversation_id',
                    'client_id',
                    'interactivity_tok_per_sec',
                ])

                for m in sorted_metrics:
                    relative_time = (m.start_time_ms - first_start) / 1000.0
                    interactivity = 1000.0 / m.tpot_ms if m.tpot_ms > 0 else 0

                    writer.writerow([
                        f"{m.start_time_ms:.3f}",
                        f"{relative_time:.3f}",
                        f"{m.ttft_ms:.3f}",
                        f"{m.tpot_ms:.3f}",
                        f"{m.latency_ms:.3f}",
                        m.input_num_turns,
                        m.input_num_tokens,
                        m.output_num_tokens,
                        m.output_num_chunks,
                        m.output_num_first_chunk_tokens,
                        f"{m.approx_cached_percent:.2f}",
                        m.conversation_id,
                        m.client_id,
                        f"{interactivity:.2f}",
                    ])

            logger.info(f"Exported client metrics to {client_csv}")
