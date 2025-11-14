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
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

try:
    import requests
    from config_manager import ConfigManager
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Please install: pip install requests pyyaml")
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


@dataclass
class ServerConfig:
    """Configuration for a vLLM server instance"""
    name: str
    backend: str
    cache_type: str
    env_vars: Dict[str, str]
    server_args: List[str]
    venv_path: str
    
    def to_dict(self) -> dict:
        return asdict(self)


class ServerConfigBuilder:
    """Builds server configurations for different backends using ConfigManager"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
    
    def get_all_configs(
        self,
        model: str,
        max_model_len: int,
        gpu_memory_utilization: float = None,
        port: int = None,
        tensor_parallel_size: int = None,
    ) -> List[ServerConfig]:
        """Generate all server configurations using ConfigManager"""
        
        # Get defaults from config
        if gpu_memory_utilization is None:
            gpu_memory_utilization = self.config.get("server.gpu_memory_utilization", 0.80)
        if port is None:
            port = self.config.get("server.default_port", 8002)
        if tensor_parallel_size is None:
            tensor_parallel_size = self.config.get("server.tensor_parallel_size", 1)
        
        # Get paths and cache sizes from config
        dynamo_venv = self.config.get("venvs.dynamo")
        lmcache_venv = self.config.get("venvs.lmcache")
        hf_home = self.config.get("model.hf_home")
        
        dynamo_cpu_cache_gb = self.config.get("cache.dynamo.cpu_cache_gb", 100)
        dynamo_weka_cache_gb = self.config.get("cache.dynamo.disk_cache_gb", 100)
        
        lmcache_chunk_size = self.config.get("cache.lmcache.chunk_size", 256)
        lmcache_weka_path = self.config.get("cache.lmcache.weka_path")
        cufile_env_path = self.config.get("cache.lmcache.cufile_env_path")
        
        configs = []
        
        # Base server args (common to all configs)
        base_args = [
            "--port", str(port),
            "--gpu-memory-utilization", str(gpu_memory_utilization),
            "--disable-log-requests",
        ]
        
        # Dynamo-specific base args (with enforce-eager)
        dynamo_base_args = base_args + ["--enforce-eager"]
        
        # Add max-model-len if specified
        if max_model_len > 0:
            base_args.extend(["--max-model-len", str(max_model_len)])
            dynamo_base_args.extend(["--max-model-len", str(max_model_len)])
        
        # Add tensor parallel if > 1
        if tensor_parallel_size > 1:
            base_args.extend(["--tensor-parallel-size", str(tensor_parallel_size)])
            dynamo_base_args.extend(["--tensor-parallel-size", str(tensor_parallel_size)])
        
        # 1. Dynamo with CPU cache
        configs.append(ServerConfig(
            name="Dynamo-CPU",
            backend="dynamo",
            cache_type="cpu",
            env_vars={
                "DYN_KVBM_CPU_CACHE_GB": str(dynamo_cpu_cache_gb),
                "HF_HOME": hf_home,
            },
            server_args=dynamo_base_args + [
                "--kv-transfer-config",
                json.dumps({
                    "kv_connector": "DynamoConnector",
                    "kv_role": "kv_both",
                    "kv_connector_module_path": "kvbm.vllm_integration.connector"
                }),
                "--no-enable-prefix-caching",
                "--model", model
            ],
            venv_path=dynamo_venv
        ))
        
        # 2. Dynamo with WEKA cache
        configs.append(ServerConfig(
            name="Dynamo-WEKA",
            backend="dynamo",
            cache_type="weka",
            env_vars={
                "DYN_KVBM_CPU_CACHE_GB": "0",
                "DYN_KVBM_DISK_CACHE_GB": str(dynamo_weka_cache_gb),
                "HF_HOME": hf_home,
            },
            server_args=dynamo_base_args + [
                "--kv-transfer-config",
                json.dumps({
                    "kv_connector": "DynamoConnector",
                    "kv_role": "kv_both",
                    "kv_connector_module_path": "kvbm.vllm_integration.connector"
                }),
                "--no-enable-prefix-caching",
                "--model", model
            ],
            venv_path=dynamo_venv
        ))
        
        # 3. LMCache with DRAM
        configs.append(ServerConfig(
            name="LMCache-DRAM",
            backend="lmcache",
            cache_type="dram",
            env_vars={
                "LMCACHE_CONFIG_FILE": "",
                "LMCACHE_CHUNK_SIZE": str(lmcache_chunk_size),
                "LMCACHE_LOCAL_CPU": "true",
                "LMCACHE_SAVE_DECODE_CACHE": "true",
                "HF_HOME": hf_home,
                "PROMETHEUS_MULTIPROC_DIR": "/tmp/lmcache_prometheus",
                "PYTHONHASHSEED": "0",
            },
            server_args=base_args + [
                "--kv-transfer-config",
                json.dumps({
                    "kv_connector": "LMCacheConnectorV1",
                    "kv_role": "kv_both",
                    "kv_connector_extra_config": {}
                }),
                "--no-enable-prefix-caching",
                "--model", model
            ],
            venv_path=lmcache_venv
        ))
        
        # 4. LMCache with WEKA (GDS)
        configs.append(ServerConfig(
            name="LMCache-WEKA",
            backend="lmcache",
            cache_type="weka",
            env_vars={
                "LMCACHE_CONFIG_FILE": "",
                "LMCACHE_WEKA_PATH": lmcache_weka_path,
                "LMCACHE_CHUNK_SIZE": str(lmcache_chunk_size),
                "LMCACHE_EXTRA_CONFIG": json.dumps({"gds_io_threads": 32}),
                "LMCACHE_CUFILE_BUFFER_SIZE": "8192",
                "LMCACHE_LOCAL_CPU": "false",
                "LMCACHE_SAVE_DECODE_CACHE": "true",
                "HF_HOME": hf_home,
                "CUFILE_ENV_PATH_JSON": cufile_env_path,
                "PROMETHEUS_MULTIPROC_DIR": "/tmp/lmcache_prometheus",
                "PYTHONHASHSEED": "0",
            },
            server_args=base_args + [
                "--kv-transfer-config",
                json.dumps({
                    "kv_connector": "LMCacheConnectorV1",
                    "kv_role": "kv_both",
                    "kv_connector_extra_config": {}
                }),
                "--no-enable-prefix-caching",
                "--model", model
            ],
            venv_path=lmcache_venv
        ))
        
        # 5. LMCache with NIXL
        configs.append(ServerConfig(
            name="LMCache-NIXL",
            backend="lmcache",
            cache_type="nixl",
            env_vars={
                "LMCACHE_CONFIG_FILE": "",
                "LMCACHE_WEKA_PATH": "",
                "LMCACHE_LOCAL_DISK": "",
                "LMCACHE_CHUNK_SIZE": str(lmcache_chunk_size),
                "LMCACHE_LOCAL_CPU": "false",
                "LMCACHE_SAVE_DECODE_CACHE": "true",
                "LMCACHE_NIXL_BUFFER_SIZE": "5368709120",
                "LMCACHE_NIXL_BUFFER_DEVICE": "cuda",
                "LMCACHE_EXTRA_CONFIG": json.dumps({
                    "enable_nixl_storage": True,
                    "nixl_backend": "GDS_MT",
                    "nixl_file_pool_size": 64,
                    "nixl_path": lmcache_weka_path or "/mnt/weka/cache"
                }),
                "CUFILE_ENV_PATH_JSON": cufile_env_path,
                "HF_HOME": hf_home,
                "PROMETHEUS_MULTIPROC_DIR": "/tmp/lmcache_prometheus",
                "PYTHONHASHSEED": "0",
            },
            server_args=base_args + [
                "--kv-transfer-config",
                json.dumps({
                    "kv_connector": "LMCacheConnectorV1",
                    "kv_role": "kv_both",
                    "kv_connector_extra_config": {}
                }),
                "--no-enable-prefix-caching",
                "--model", model
            ],
            venv_path=lmcache_venv
        ))
        
        # 6. Vanilla vLLM (no KV cache transfer, no prefix caching)
        configs.append(ServerConfig(
            name="Vanilla-vLLM-nep",
            backend="vanilla",
            cache_type="none",
            env_vars={
                "HF_HOME": hf_home,
            },
            server_args=base_args + ["--no-enable-prefix-caching", "--model", model],
            venv_path=lmcache_venv
        ))
        
        # 7. Vanilla vLLM with prefix caching enabled
        configs.append(ServerConfig(
            name="Vanilla-vLLM-pc",
            backend="vanilla",
            cache_type="prefix-cache",
            env_vars={
                "HF_HOME": hf_home,
            },
            server_args=base_args + ["--model", model],
            venv_path=lmcache_venv
        ))
        
        return configs


class VLLMServerRunner:
    """Runs vLLM server with different configurations"""
    
    def __init__(self, config: ServerConfig, port: int, startup_timeout: int, debug_log_dir: Optional[str] = None):
        self.config = config
        self.port = port
        self.startup_timeout = startup_timeout
        self.process: Optional[subprocess.Popen] = None
        self.api_endpoint = f"http://localhost:{port}"
        self.debug_log_dir = debug_log_dir
        
    def start_server(self) -> bool:
        """Start the vLLM server"""
        logger.info(f"\n{Colors.HEADER}{'='*80}{Colors.ENDC}")
        logger.info(f"{Colors.HEADER}Starting Server: {self.config.name}{Colors.ENDC}")
        logger.info(f"{Colors.HEADER}Backend: {self.config.backend} | Cache: {self.config.cache_type}{Colors.ENDC}")
        logger.info(f"{Colors.HEADER}Venv: {self.config.venv_path}{Colors.ENDC}")
        logger.info(f"{Colors.HEADER}{'='*80}{Colors.ENDC}")
        
        venv_path = os.path.expanduser(self.config.venv_path)
        python_bin = os.path.join(venv_path, "bin", "python")
        
        if not os.path.exists(python_bin):
            logger.error(f"{Colors.FAIL}✗ Python not found in venv: {python_bin}{Colors.ENDC}")
            return False
        
        cmd = [python_bin, "-m", "vllm.entrypoints.openai.api_server"] + self.config.server_args
        
        logger.info(f"{Colors.OKCYAN}Command: {' '.join(cmd)}{Colors.ENDC}")
        logger.info(f"{Colors.OKCYAN}Environment variables:{Colors.ENDC}")
        for key, value in self.config.env_vars.items():
            logger.info(f"  {key}={value}")
        
        try:
            env = os.environ.copy()
            
            for key, value in self.config.env_vars.items():
                if value == "":
                    env.pop(key, None)
                else:
                    env[key] = value
            
            if 'CUDA_VISIBLE_DEVICES' in os.environ:
                env['CUDA_VISIBLE_DEVICES'] = os.environ['CUDA_VISIBLE_DEVICES']
                logger.info(f"  CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")
            
            if self.debug_log_dir:
                debug_log_path = Path(self.debug_log_dir)
                debug_log_path.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file = debug_log_path / f"{self.config.name}_{timestamp}.log"
                logger.info(f"  Server output will be written to: {log_file}")
                stdout_handle = open(log_file, 'w')
            else:
                stdout_handle = subprocess.PIPE
            
            self.process = subprocess.Popen(
                cmd,
                env=env,
                stdout=stdout_handle,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            logger.info(f"{Colors.OKCYAN}Server process started (PID: {self.process.pid}), waiting for ready...{Colors.ENDC}")
            
            if not self._wait_for_server(timeout=self.startup_timeout):
                logger.error(f"{Colors.FAIL}✗ Server failed to start{Colors.ENDC}")
                self.shutdown()
                return False
            
            logger.info(f"{Colors.OKGREEN}✓ Server is ready at {self.api_endpoint}{Colors.ENDC}\n")
            return True
            
        except Exception as e:
            logger.error(f"{Colors.FAIL}✗ Failed to start server: {e}{Colors.ENDC}")
            import traceback
            traceback.print_exc()
            return False
    
    def _wait_for_server(self, timeout: int) -> bool:
        """Wait for server to be ready"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.api_endpoint}/health", timeout=2)
                if response.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                pass
            
            if self.process and self.process.poll() is not None:
                logger.error(f"{Colors.FAIL}Server process died{Colors.ENDC}")
                return False
            
            time.sleep(2)
        
        return False
    
    def shutdown(self):
        """Shutdown the server"""
        logger.info(f"{Colors.WARNING}Shutting down server: {self.config.name}{Colors.ENDC}")
        
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning(f"{Colors.WARNING}Server didn't terminate gracefully, killing...{Colors.ENDC}")
                self.process.kill()
                self.process.wait()
            self.process = None


class BenchmarkRunner:
    """Runs tests against all server configurations"""
    
    def __init__(
        self,
        configs: List[ServerConfig],
        config_manager: ConfigManager,
        mode: str,
        model: str,
        port: int,
        script_dir: Path,
    ):
        self.configs = configs
        self.config_manager = config_manager
        self.mode = mode
        self.model = model
        self.port = port
        self.script_dir = script_dir
        
        # Get mode-specific output directory (with automatic fallback)
        self.output_dir = config_manager.get_output_dir(mode)
        self.debug_log_dir = config_manager.get_expanded_path("output.debug_logs_dir")
        self.shutdown_delay = config_manager.get("server.shutdown_delay", 10)
        self.startup_timeout = config_manager.get("server.startup_timeout", 600)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_config_dir_name(self, config: ServerConfig) -> str:
        """Generate directory name for a config"""
        config_name_map = {
            "Dynamo-CPU": "dynamo-cpu-nep",
            "Dynamo-WEKA": "dynamo-weka-nep",
            "LMCache-DRAM": "lmcache-dram-nep",
            "LMCache-WEKA": "lmcache-weka-nep",
            "LMCache-NIXL": "lmcache-nixl-nep",
            "Vanilla-vLLM-nep": "vanilla-nep",
            "Vanilla-vLLM-pc": "vanilla-pc"
        }
        return config_name_map.get(config.name, config.name.lower().replace(" ", "-"))
    
    def _get_model_dir_name(self) -> str:
        """Generate directory name from model name"""
        return self.model.replace("/", "--")
    
    def _get_tester_script(self) -> str:
        """Get the appropriate tester script based on mode"""
        tester_map = {
            "single-prompt": "single_prompt_tester.py",
            "cache-rate": "cache_rate_tester.py",
            "working-set": "working_set_tester.py",
        }
        return tester_map.get(self.mode, "single_prompt_tester.py")
    
    def _build_tester_command(
        self,
        python_bin: str,
        config_output_dir: Path,
        tokenizer: str,
        **kwargs
    ) -> List[str]:
        """Build command for the appropriate tester based on mode"""
        tester_script = self.script_dir / self._get_tester_script()
        
        base_cmd = [
            python_bin,
            str(tester_script),
            "--api-endpoint", f"http://localhost:{self.port}",
            "--output-dir", str(config_output_dir),
            "--tokenizer", tokenizer
        ]
        
        if self.mode == "single-prompt":
            # Get params from config or kwargs
            min_tokens = kwargs.get("min_tokens") or self.config_manager.get("testing.single_prompt.min_tokens")
            max_tokens = kwargs.get("max_tokens") or self.config_manager.get("testing.single_prompt.max_tokens")
            output_tokens = kwargs.get("output_tokens") or self.config_manager.get("testing.single_prompt.output_tokens")
            num_iterations = kwargs.get("num_iterations") or self.config_manager.get("testing.single_prompt.num_iterations")
            
            base_cmd.extend([
                "--min-tokens", str(min_tokens),
                "--max-tokens", str(max_tokens),
                "--output-tokens", str(output_tokens),
                "--num-iterations", str(num_iterations),
            ])
        
        elif self.mode == "cache-rate":
            input_tokens = kwargs.get("input_tokens") or self.config_manager.get("testing.cache_rate.input_tokens")
            output_tokens = kwargs.get("output_tokens") or self.config_manager.get("testing.cache_rate.output_tokens")
            num_iterations = kwargs.get("num_iterations") or self.config_manager.get("testing.cache_rate.num_iterations")
            
            base_cmd.extend([
                "--input-tokens", str(input_tokens),
                "--output-tokens", str(output_tokens),
                "--num-iterations", str(num_iterations),
            ])
        
        elif self.mode == "working-set":
            cache_hit_rate = kwargs.get("cache_hit_rate") or self.config_manager.get("testing.working_set.cache_hit_rate")
            min_ws = kwargs.get("min_working_set_gb") or self.config_manager.get("testing.working_set.min_working_set_gb")
            max_ws = kwargs.get("max_working_set_gb") or self.config_manager.get("testing.working_set.max_working_set_gb")
            step_gb = kwargs.get("step_gb") or self.config_manager.get("testing.working_set.step_gb")
            input_tokens = kwargs.get("input_tokens") or self.config_manager.get("testing.working_set.input_tokens")
            output_tokens = kwargs.get("output_tokens") or self.config_manager.get("testing.working_set.output_tokens")
            num_iterations = kwargs.get("num_iterations") or self.config_manager.get("testing.working_set.num_iterations")
            
            base_cmd.extend([
                "--cache-hit-rate", str(cache_hit_rate),
                "--min-working-set-gb", str(min_ws),
                "--max-working-set-gb", str(max_ws),
                "--step-gb", str(step_gb),
                "--input-tokens", str(input_tokens),
                "--output-tokens", str(output_tokens),
                "--num-iterations", str(num_iterations),
            ])
        
        return base_cmd
    
    def run_benchmarks(self, tokenizer: str, **kwargs):
        """Run benchmarks across all configurations"""
        logger.info(f"\n{Colors.BOLD}{'='*80}{Colors.ENDC}")
        logger.info(f"{Colors.BOLD}Running {self.mode.upper()} Benchmarks{Colors.ENDC}")
        logger.info(f"{Colors.BOLD}{'='*80}{Colors.ENDC}\n")
        
        for config in self.configs:
            server = VLLMServerRunner(
                config, 
                self.port,
                self.startup_timeout,
                str(self.debug_log_dir) if self.debug_log_dir else None
            )
            
            try:
                if not server.start_server():
                    logger.error(f"{Colors.FAIL}✗ Failed to start {config.name}, skipping...{Colors.ENDC}")
                    continue
                
                logger.info(f"{Colors.OKCYAN}Running {self.mode} test for {config.name}...{Colors.ENDC}")
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_dir = self._get_model_dir_name()
                config_dir = self._get_config_dir_name(config)
                config_output_dir = self.output_dir / model_dir / config_dir / timestamp
                config_output_dir.mkdir(parents=True, exist_ok=True)
                
                logger.info(f"{Colors.OKCYAN}Results will be saved to: {config_output_dir}{Colors.ENDC}")
                
                venv_path = os.path.expanduser(config.venv_path)
                python_bin = os.path.join(venv_path, "bin", "python")
                
                tester_cmd = self._build_tester_command(
                    python_bin,
                    config_output_dir,
                    tokenizer,
                    **kwargs
                )
                
                logger.info(f"  Command: {' '.join(tester_cmd)}")
                
                result = subprocess.run(tester_cmd, capture_output=False, text=True)
                
                if result.returncode == 0:
                    logger.info(f"{Colors.OKGREEN}✓ Testing complete for {config.name}{Colors.ENDC}")
                    logger.info(f"{Colors.OKGREEN}  Results saved to: {config_output_dir}{Colors.ENDC}\n")
                else:
                    logger.error(f"{Colors.FAIL}✗ Testing failed for {config.name}{Colors.ENDC}\n")
                
            except Exception as e:
                logger.error(f"{Colors.FAIL}Error testing {config.name}: {e}{Colors.ENDC}\n")
                
            finally:
                server.shutdown()
                logger.info(f"{Colors.OKCYAN}Waiting {self.shutdown_delay} seconds before next configuration...{Colors.ENDC}\n")
                time.sleep(self.shutdown_delay)


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
                       choices=["dynamo-cpu", "dynamo-weka", "lmcache-dram", 
                               "lmcache-weka", "lmcache-nixl", "vanilla-nep", "vanilla-pc"],
                       help="Specific configs to test (default: all enabled in config)")
    
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
        logger.info(f"\n{Colors.OKCYAN}{'='*80}{Colors.ENDC}")
        logger.info(f"{Colors.OKCYAN}DRY RUN MODE{Colors.ENDC}")
        logger.info(f"{Colors.OKCYAN}{'='*80}{Colors.ENDC}\n")
        
        for cfg in all_configs:
            logger.info(f"{Colors.BOLD}Configuration: {cfg.name}{Colors.ENDC}")
            logger.info(f"  Backend: {cfg.backend} | Cache: {cfg.cache_type}")
            logger.info(f"  Venv: {cfg.venv_path}")
            logger.info(f"\n  Environment variables:")
            for key, val in cfg.env_vars.items():
                logger.info(f"    export {key}={val}")
            
            cmd_parts = [f"{cfg.venv_path}/bin/python", "-m", "vllm.entrypoints.openai.api_server"] + cfg.server_args
            logger.info(f"\n  Command:\n    {' '.join(cmd_parts)}\n")
        
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
