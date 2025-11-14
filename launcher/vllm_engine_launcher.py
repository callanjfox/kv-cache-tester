#!/usr/bin/env python3
"""
vLLM Engine Launcher and Testing Tool (v2.0)

Launches vLLM servers with different backends and runs performance tests.
Supports single prompt, cache rate, and working set testing modes.

Usage:
    # Run single prompt tests (default)
    python vllm_engine_launcher_v2.py --mode single-prompt --model Qwen/Qwen3-8B
    
    # Run cache rate tests
    python vllm_engine_launcher_v2.py --mode cache-rate --model Qwen/Qwen3-8B
    
    # Run working set tests
    python vllm_engine_launcher_v2.py --mode working-set --model Qwen/Qwen3-8B
    
    # Use custom config file
    python vllm_engine_launcher_v2.py --config my_config.yaml
    
    # Override with environment variables
    export KVTEST_OUTPUT_BASE_DIR="/my/custom/results"
    python vllm_engine_launcher_v2.py
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add parent directory to path for direct script execution
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import requests
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Please install: pip install requests pyyaml")
    sys.exit(1)

from launcher.config_manager import ConfigManager
from launcher.server_configs import ServerConfig, ServerConfigBuilder
from launcher.benchmark_runner import BenchmarkRunner


class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def init_logger(name: str, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    log_format = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    
    return logger


logger = init_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="vLLM Server Launcher and Testing Tool (v2.0)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Configuration file
    parser.add_argument("--config-file", type=str,
                       help="Path to YAML config file (e.g., config.yaml, config.local.yaml)")
    
    # Test mode
    parser.add_argument("--mode", type=str, 
                       choices=["single-prompt", "cache-rate", "working-set"],
                       default="single-prompt",
                       help="Test mode (default: single-prompt)")
    
    # Model configuration
    parser.add_argument("--model", type=str,
                       help="Model to use (default: from config)")
    parser.add_argument("--max-model-len", type=int,
                       help="Maximum context length (default: model's max)")
    parser.add_argument("--tokenizer", type=str,
                       help="Tokenizer to use (default: same as model)")
    
    # Server configuration
    parser.add_argument("--port", type=int,
                       help="vLLM server port (default: from config)")
    parser.add_argument("--gpu-memory-utilization", type=float,
                       help="GPU memory utilization (default: from config)")
    parser.add_argument("--tensor-parallel-size", type=int,
                       help="Tensor parallel size (default: from config)")
    
    # Backend selection
    parser.add_argument("--configs", type=str, nargs="+",
                       metavar="BACKEND",
                       choices=["dynamo-cpu", "dynamo-weka", "lmcache-dram", 
                               "lmcache-weka", "lmcache-nixl", "vanilla-nep", "vanilla-pc"],
                       help="Specific backends to test: dynamo-cpu, dynamo-weka, lmcache-dram, lmcache-weka, lmcache-nixl, vanilla-nep, vanilla-pc (default: all enabled in config)")
    
    # Single prompt mode parameters
    parser.add_argument("--min-tokens", type=int,
                       help="Min context size for single-prompt tests")
    parser.add_argument("--max-tokens", type=int,
                       help="Max context size for single-prompt tests")
    parser.add_argument("--output-tokens", type=int,
                       help="Output tokens to generate")
    parser.add_argument("--num-iterations", type=int,
                       help="Number of iterations per test")
    
    # Cache rate mode parameters
    parser.add_argument("--input-tokens", type=int,
                       help="Input tokens for cache-rate/working-set tests")
    
    # Working set mode parameters
    parser.add_argument("--cache-hit-rate", type=int,
                       help="Cache hit rate for working-set tests")
    parser.add_argument("--min-working-set-gb", type=int,
                       help="Minimum working set size in GB")
    parser.add_argument("--max-working-set-gb", type=int,
                       help="Maximum working set size in GB")
    parser.add_argument("--step-gb", type=int,
                       help="Working set step size in GB")
    
    # Utility options
    parser.add_argument("--dry-run", action="store_true",
                       help="Print commands without executing")
    parser.add_argument("--print-config", action="store_true",
                       help="Print current configuration and exit")
    parser.add_argument("--debug-logs", type=str,
                       help="Directory to save server debug logs (default: from config)")
    
    args = parser.parse_args()
    
    # Load configuration
    config = ConfigManager(args.config_file)
    
    # Print config and exit if requested
    if args.print_config:
        config.print_config()
        return
    
    # Override config with CLI args
    model = args.model or config.get("model.default_model")
    max_model_len = args.max_model_len or 0
    tokenizer = args.tokenizer or model
    port = args.port or config.get("server.default_port")
    
    # Generate configs
    builder = ServerConfigBuilder(config)
    all_configs = builder.get_all_configs(
        model=model,
        max_model_len=max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        port=port,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    
    # Filter configs if specified
    if args.configs:
        config_map = {
            "dynamo-cpu": "Dynamo-CPU",
            "dynamo-weka": "Dynamo-WEKA",
            "lmcache-dram": "LMCache-DRAM",
            "lmcache-weka": "LMCache-WEKA",
            "lmcache-nixl": "LMCache-NIXL",
            "vanilla-nep": "Vanilla-vLLM-nep",
            "vanilla-pc": "Vanilla-vLLM-pc"
        }
        selected_names = [config_map[c] for c in args.configs]
        all_configs = [c for c in all_configs if c.name in selected_names]
    else:
        # Use enabled configs from config file
        enabled = config.get("backends.enabled", [])
        if enabled:
            config_map = {
                "dynamo-cpu": "Dynamo-CPU",
                "dynamo-weka": "Dynamo-WEKA",
                "lmcache-dram": "LMCache-DRAM",
                "lmcache-weka": "LMCache-WEKA",
                "lmcache-nixl": "LMCache-NIXL",
                "vanilla-nep": "Vanilla-vLLM-nep",
                "vanilla-pc": "Vanilla-vLLM-pc"
            }
            enabled_names = [config_map.get(c) for c in enabled if config_map.get(c)]
            all_configs = [c for c in all_configs if c.name in enabled_names]
    
    logger.info(f"\n{Colors.HEADER}Testing {len(all_configs)} configurations:{Colors.ENDC}")
    for cfg in all_configs:
        logger.info(f"  - {cfg.name}")
    
    # Dry run mode
    if args.dry_run:
        logger.info("")
        logger.info(f"{Colors.HEADER}{'='*80}{Colors.ENDC}")
        logger.info(f"{Colors.HEADER}DRY RUN MODE - {args.mode.upper()} MODE{Colors.ENDC}")
        logger.info(f"{Colors.HEADER}{'='*80}{Colors.ENDC}")
        logger.info("")
        
        # Determine tester script
        tester_map = {
            "single-prompt": "single_prompt_tester.py",
            "cache-rate": "cache_rate_tester.py",
            "working-set": "working_set_tester.py",
        }
        tester_script = tester_map.get(args.mode, "single_prompt_tester.py")
        
        for i, cfg in enumerate(all_configs):
            if i > 0:
                logger.info(f"\n{Colors.OKBLUE}{'='*80}{Colors.ENDC}\n")
            logger.info(f"{Colors.HEADER}>>> Configuration: {cfg.name}{Colors.ENDC}")
            logger.info(f"  Backend: {cfg.backend} | Cache: {cfg.cache_type}")
            logger.info(f"  Venv: {cfg.venv_path}")
            logger.info(f"\n{Colors.OKBLUE}  Environment variables:{Colors.ENDC}")
            for key, val in cfg.env_vars.items():
                logger.info(f"    export {key}={val}")
            
            server_cmd_parts = [f"{cfg.venv_path}/bin/python", "-m", "vllm.entrypoints.openai.api_server"] + cfg.server_args
            logger.info(f"\n{Colors.OKBLUE}  Server command:{Colors.ENDC}\n    {' '.join(server_cmd_parts)}")
            
            # Show tester command
            tester_cmd_parts = [f"{cfg.venv_path}/bin/python", tester_script, 
                              "--api-endpoint", f"http://localhost:{port}",
                              "--tokenizer", tokenizer,
                              "--output-dir", "<results_dir>"]
            
            if args.mode == "single-prompt":
                min_tok = args.min_tokens or config.get("testing.single_prompt.min_tokens")
                max_tok = args.max_tokens or config.get("testing.single_prompt.max_tokens")
                # If max_tokens not specified but max_model_len is set, use that
                if max_tok is None and max_model_len > 0:
                    max_tok = max_model_len
                out_tok = args.output_tokens or config.get("testing.single_prompt.output_tokens")
                iters = args.num_iterations or config.get("testing.single_prompt.num_iterations")
                tester_cmd_parts.extend(["--min-tokens", str(min_tok),
                                        "--output-tokens", str(out_tok), "--num-iterations", str(iters)])
                if max_tok is not None:
                    tester_cmd_parts.extend(["--max-tokens", str(max_tok)])
            elif args.mode == "cache-rate":
                in_tok = args.input_tokens or config.get("testing.cache_rate.input_tokens")
                out_tok = args.output_tokens or config.get("testing.cache_rate.output_tokens")
                iters = args.num_iterations or config.get("testing.cache_rate.num_iterations")
                tester_cmd_parts.extend(["--input-tokens", str(in_tok), "--output-tokens", str(out_tok),
                                        "--num-iterations", str(iters)])
            elif args.mode == "working-set":
                chr = args.cache_hit_rate or config.get("testing.working_set.cache_hit_rate")
                min_ws = args.min_working_set_gb or config.get("testing.working_set.min_working_set_gb")
                max_ws = args.max_working_set_gb or config.get("testing.working_set.max_working_set_gb")
                step = args.step_gb or config.get("testing.working_set.step_gb")
                in_tok = args.input_tokens or config.get("testing.working_set.input_tokens")
                out_tok = args.output_tokens or config.get("testing.working_set.output_tokens")
                iters = args.num_iterations or config.get("testing.working_set.num_iterations")
                tester_cmd_parts.extend(["--cache-hit-rate", str(chr), "--min-working-set-gb", str(min_ws),
                                        "--max-working-set-gb", str(max_ws), "--step-gb", str(step),
                                        "--input-tokens", str(in_tok), "--output-tokens", str(out_tok),
                                        "--num-iterations", str(iters)])
            
            logger.info(f"\n{Colors.OKBLUE}  Tester command:{Colors.ENDC}\n    {' '.join(tester_cmd_parts)}\n")
        
        return
    
    # Override debug logs directory if specified
    if args.debug_logs:
        config.set("output.debug_logs_dir", args.debug_logs)
    
    # Run benchmarks
    benchmark = BenchmarkRunner(
        configs=all_configs,
        config_manager=config,
        mode=args.mode,
        model=model,
        port=port,
        script_dir=Path(__file__).parent,
    )
    
    # Build kwargs from CLI args
    test_kwargs = {}
    if args.min_tokens:
        test_kwargs["min_tokens"] = args.min_tokens
    if args.max_tokens:
        test_kwargs["max_tokens"] = args.max_tokens
    elif max_model_len > 0:
        # If max_tokens not specified but max_model_len is set, use that
        test_kwargs["max_tokens"] = max_model_len
    if args.output_tokens:
        test_kwargs["output_tokens"] = args.output_tokens
    if args.num_iterations:
        test_kwargs["num_iterations"] = args.num_iterations
    if args.input_tokens:
        test_kwargs["input_tokens"] = args.input_tokens
    if args.cache_hit_rate:
        test_kwargs["cache_hit_rate"] = args.cache_hit_rate
    if args.min_working_set_gb:
        test_kwargs["min_working_set_gb"] = args.min_working_set_gb
    if args.max_working_set_gb:
        test_kwargs["max_working_set_gb"] = args.max_working_set_gb
    if args.step_gb:
        test_kwargs["step_gb"] = args.step_gb
    
    benchmark.run_benchmarks(tokenizer=tokenizer, **test_kwargs)
    
    logger.info(f"\n{Colors.OKGREEN}{'='*80}{Colors.ENDC}")
    logger.info(f"{Colors.OKGREEN}✓ All benchmarks complete!{Colors.ENDC}")
    logger.info(f"{Colors.OKGREEN}Results saved to: {benchmark.output_dir}{Colors.ENDC}")
    logger.info(f"{Colors.OKGREEN}{'='*80}{Colors.ENDC}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
