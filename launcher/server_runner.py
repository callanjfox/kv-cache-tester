#!/usr/bin/env python3
"""
vLLM Server Runner

Manages vLLM server lifecycle - starting, health checking, and shutdown.
"""

import logging
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import requests
except ImportError:
    print("Missing required dependency: requests")
    import sys
    sys.exit(1)

from launcher.server_configs import ServerConfig


class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'


logger = logging.getLogger(__name__)


class VllmServer:
    """Manages a vLLM server instance"""
    
    def __init__(self, config: ServerConfig, port: int, debug_log_dir: Optional[str] = None):
        self.config = config
        self.port = port
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
        
        # Build command
        cmd = [python_bin, "-m", "vllm.entrypoints.openai.api_server"] + self.config.server_args
        
        logger.info(f"{Colors.OKCYAN}Command: {' '.join(cmd)}{Colors.ENDC}")
        logger.info(f"{Colors.OKCYAN}Environment variables:{Colors.ENDC}")
        for key, value in self.config.env_vars.items():
            logger.info(f"  {key}={value}")
        
        # Start server process
        try:
            env = os.environ.copy()
            
            # Remove env vars that should be unset (empty string values)
            for key, value in self.config.env_vars.items():
                if value == "":
                    env.pop(key, None)
                else:
                    env[key] = value
            
            # Ensure CUDA_VISIBLE_DEVICES is preserved
            if 'CUDA_VISIBLE_DEVICES' in os.environ:
                env['CUDA_VISIBLE_DEVICES'] = os.environ['CUDA_VISIBLE_DEVICES']
                logger.info(f"  CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")
            
            # Setup output handling
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
            
            # Wait for server to be ready (longer timeout for large models with TP)
            if not self._wait_for_server(timeout=600):
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
    
    def _wait_for_server(self, timeout: int = 600) -> bool:
        """Wait for server to be ready"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.api_endpoint}/health", timeout=2)
                if response.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                pass
            
            # Check if process died
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
