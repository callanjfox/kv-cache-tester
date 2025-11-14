#!/usr/bin/env python3
"""
Benchmark Runner

Orchestrates running tests across different server configurations.
"""

import logging
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import List

from launcher.config_manager import ConfigManager
from launcher.server_configs import ServerConfig
from launcher.server_runner import VllmServer


class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


logger = logging.getLogger(__name__)


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
        self.config = config_manager
        self.mode = mode
        self.model = model
        self.port = port
        self.script_dir = script_dir
        
        # Get output dir and shutdown delay from config
        output_base = self.config.get("output.base_dir", "~/dynamo/results")
        self.output_dir = Path(os.path.expanduser(output_base)) / mode
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.shutdown_delay = self.config.get("server.shutdown_delay", 10)
        self.debug_log_dir = self.config.get("output.debug_logs_dir")
    
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
        model_name = self.model.replace("/", "--")
        return model_name
    
    def _build_tester_command(self, python_bin: str, output_dir: Path, tokenizer: str, **kwargs) -> List[str]:
        """Build the tester command based on mode"""
        tester_map = {
            "single-prompt": "single_prompt_tester.py",
            "cache-rate": "cache_rate_tester.py",
            "working-set": "working_set_tester.py",
        }
        tester_script = tester_map.get(self.mode, "single_prompt_tester.py")
        
        cmd = [
            python_bin,
            str(self.script_dir.parent / tester_script),
            "--api-endpoint", f"http://localhost:{self.port}",
            "--tokenizer", tokenizer,
            "--output-dir", str(output_dir),
        ]
        
        # Add mode-specific arguments
        if self.mode == "single-prompt":
            min_tok = kwargs.get("min_tokens") or self.config.get("testing.single_prompt.min_tokens", 1000)
            max_tok = kwargs.get("max_tokens") or self.config.get("testing.single_prompt.max_tokens")
            out_tok = kwargs.get("output_tokens") or self.config.get("testing.single_prompt.output_tokens", 256)
            iters = kwargs.get("num_iterations") or self.config.get("testing.single_prompt.num_iterations", 5)
            
            cmd.extend(["--min-tokens", str(min_tok)])
            if max_tok is not None:
                cmd.extend(["--max-tokens", str(max_tok)])
            cmd.extend(["--output-tokens", str(out_tok), "--num-iterations", str(iters)])
            
        elif self.mode == "cache-rate":
            in_tok = kwargs.get("input_tokens") or self.config.get("testing.cache_rate.input_tokens", 32768)
            out_tok = kwargs.get("output_tokens") or self.config.get("testing.cache_rate.output_tokens", 256)
            iters = kwargs.get("num_iterations") or self.config.get("testing.cache_rate.num_iterations", 100)
            
            cmd.extend(["--input-tokens", str(in_tok), "--output-tokens", str(out_tok),
                       "--num-iterations", str(iters)])
            
        elif self.mode == "working-set":
            chr = kwargs.get("cache_hit_rate") or self.config.get("testing.working_set.cache_hit_rate", 90)
            min_ws = kwargs.get("min_working_set_gb") or self.config.get("testing.working_set.min_working_set_gb", 1)
            max_ws = kwargs.get("max_working_set_gb") or self.config.get("testing.working_set.max_working_set_gb", 100)
            step = kwargs.get("step_gb") or self.config.get("testing.working_set.step_gb", 10)
            in_tok = kwargs.get("input_tokens") or self.config.get("testing.working_set.input_tokens", 32768)
            out_tok = kwargs.get("output_tokens") or self.config.get("testing.working_set.output_tokens", 256)
            iters = kwargs.get("num_iterations") or self.config.get("testing.working_set.num_iterations", 10)
            
            cmd.extend(["--cache-hit-rate", str(chr), "--min-working-set-gb", str(min_ws),
                       "--max-working-set-gb", str(max_ws), "--step-gb", str(step),
                       "--input-tokens", str(in_tok), "--output-tokens", str(out_tok),
                       "--num-iterations", str(iters)])
        
        return cmd
    
    def run_benchmarks(self, tokenizer: str, **kwargs):
        """Run benchmarks across all configurations"""
        logger.info(f"\n{Colors.BOLD}{'='*80}{Colors.ENDC}")
        logger.info(f"{Colors.BOLD}Running {self.mode.upper()} Benchmarks{Colors.ENDC}")
        logger.info(f"{Colors.BOLD}{'='*80}{Colors.ENDC}\n")
        
        for config in self.configs:
            server = VllmServer(config, self.port, self.debug_log_dir)
            
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
