#!/usr/bin/env python3
"""
Configuration Manager for KV Cache Tester

Handles loading configuration from:
1. Default values (hardcoded)
2. YAML config file (if provided)
3. Environment variables (highest priority)
4. Command-line arguments (override everything)

Environment variable naming convention:
- KVTEST_OUTPUT_BASE_DIR
- KVTEST_VENV_DYNAMO
- KVTEST_CACHE_DYNAMO_CPU_GB
- etc.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional
import logging

try:
    import yaml
except ImportError:
    yaml = None

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages configuration from multiple sources with priority"""
    
    # Default configuration (fallback values - generic, no machine-specific paths)
    DEFAULTS = {
        "output": {
            "base_dir": "./results",
            "single_prompt_dir": None,  # Will use base_dir/single_prompt
            "cache_rate_dir": None,      # Will use base_dir/cache_rate
            "working_set_dir": None,     # Will use base_dir/working_set
            "debug_logs_dir": None,
        },
        "venvs": {
            "dynamo": None,   # Must be set via config file, env var, or CLI
            "lmcache": None,  # Must be set via config file, env var, or CLI
        },
        "cache": {
            "dynamo": {
                "cpu_cache_gb": 100,
                "disk_cache_gb": 100,
            },
            "lmcache": {
                "chunk_size": 256,
                "weka_path": None,        # Set if using WEKA/GDS
                "cufile_env_path": None,  # Set if using GDS
            },
        },
        "model": {
            "hf_home": None,  # Will use HF default if not set
            "default_model": "Qwen/Qwen2.5-7B",
        },
        "server": {
            "default_port": 8002,
            "gpu_memory_utilization": 0.80,
            "tensor_parallel_size": 1,
            "startup_timeout": 600,
            "shutdown_delay": 10,
        },
        "testing": {
            "single_prompt": {
                "min_tokens": 1000,
                "max_tokens": None,
                "output_tokens": 256,
                "num_iterations": 5,
            },
            "cache_rate": {
                "input_tokens": 32768,
                "output_tokens": 256,
                "num_iterations": 10,
                "cache_hit_rates": [0, 25, 50, 75, 100],
            },
            "working_set": {
                "cache_hit_rate": 50,
                "min_working_set_gb": 1,
                "max_working_set_gb": 100,
                "step_gb": 10,
                "input_tokens": 32768,
                "output_tokens": 256,
                "num_iterations": 5,
            },
        },
        "backends": {
            "enabled": [
                "dynamo-cpu",
                "dynamo-weka",
                "lmcache-dram",
                "lmcache-weka",
                "lmcache-nixl",
                "vanilla-nep",
                "vanilla-pc",
            ],
        },
    }
    
    # Mapping of environment variables to config paths
    ENV_VAR_MAP = {
        "KVTEST_OUTPUT_BASE_DIR": "output.base_dir",
        "KVTEST_OUTPUT_SINGLE_PROMPT_DIR": "output.single_prompt_dir",
        "KVTEST_OUTPUT_CACHE_RATE_DIR": "output.cache_rate_dir",
        "KVTEST_OUTPUT_WORKING_SET_DIR": "output.working_set_dir",
        "KVTEST_OUTPUT_DEBUG_LOGS_DIR": "output.debug_logs_dir",
        "KVTEST_VENV_DYNAMO": "venvs.dynamo",
        "KVTEST_VENV_LMCACHE": "venvs.lmcache",
        "KVTEST_CACHE_DYNAMO_CPU_GB": "cache.dynamo.cpu_cache_gb",
        "KVTEST_CACHE_DYNAMO_DISK_GB": "cache.dynamo.disk_cache_gb",
        "KVTEST_CACHE_LMCACHE_CHUNK_SIZE": "cache.lmcache.chunk_size",
        "KVTEST_CACHE_LMCACHE_WEKA_PATH": "cache.lmcache.weka_path",
        "KVTEST_CACHE_LMCACHE_CUFILE_PATH": "cache.lmcache.cufile_env_path",
        "KVTEST_MODEL_HF_HOME": "model.hf_home",
        "KVTEST_MODEL_DEFAULT": "model.default_model",
        "KVTEST_SERVER_PORT": "server.default_port",
        "KVTEST_SERVER_GPU_MEM_UTIL": "server.gpu_memory_utilization",
        "KVTEST_SERVER_TP_SIZE": "server.tensor_parallel_size",
        "KVTEST_SERVER_STARTUP_TIMEOUT": "server.startup_timeout",
        "KVTEST_SERVER_SHUTDOWN_DELAY": "server.shutdown_delay",
    }
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager
        
        Args:
            config_file: Path to YAML config file (optional)
        """
        self.config = self._deep_copy(self.DEFAULTS)
        
        # Load from config file if provided
        if config_file:
            self._load_from_file(config_file)
        
        # Override with environment variables
        self._load_from_env()
    
    def _deep_copy(self, obj: Any) -> Any:
        """Deep copy a nested dict/list structure"""
        if isinstance(obj, dict):
            return {k: self._deep_copy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._deep_copy(item) for item in obj]
        else:
            return obj
    
    def _load_from_file(self, config_file: str):
        """Load configuration from YAML file"""
        if yaml is None:
            logger.warning("PyYAML not installed, skipping config file loading")
            return
        
        config_path = Path(config_file).expanduser()
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}")
            return
        
        try:
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
            
            if file_config:
                self._merge_configs(self.config, file_config)
                logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading config file {config_path}: {e}")
    
    def _merge_configs(self, base: Dict, override: Dict):
        """Recursively merge override config into base config"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_configs(base[key], value)
            else:
                base[key] = value
    
    def _load_from_env(self):
        """Load configuration from environment variables"""
        for env_var, config_path in self.ENV_VAR_MAP.items():
            value = os.environ.get(env_var)
            if value is not None:
                self._set_nested_value(config_path, value)
                logger.debug(f"Set {config_path} from {env_var}={value}")
    
    def _set_nested_value(self, path: str, value: str):
        """Set a nested dictionary value using dot notation"""
        keys = path.split('.')
        current = self.config
        
        # Navigate to the parent dict
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set the value with type conversion
        final_key = keys[-1]
        current[final_key] = self._convert_type(value, current.get(final_key))
    
    def _convert_type(self, value: str, reference: Any) -> Any:
        """Convert string value to appropriate type based on reference value"""
        if reference is None:
            # Try to infer type
            if value.lower() in ('true', 'false'):
                return value.lower() == 'true'
            try:
                return int(value)
            except ValueError:
                try:
                    return float(value)
                except ValueError:
                    return value
        elif isinstance(reference, bool):
            return value.lower() in ('true', '1', 'yes')
        elif isinstance(reference, int):
            return int(value)
        elif isinstance(reference, float):
            return float(value)
        elif isinstance(reference, list):
            # Parse comma-separated list
            return [item.strip() for item in value.split(',')]
        else:
            return value
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation
        
        Args:
            path: Dot-separated path (e.g., "output.base_dir")
            default: Default value if path not found
            
        Returns:
            Configuration value
        """
        keys = path.split('.')
        current = self.config
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        
        return current
    
    def set(self, path: str, value: Any):
        """
        Set a configuration value using dot notation
        
        Args:
            path: Dot-separated path (e.g., "output.base_dir")
            value: Value to set
        """
        keys = path.split('.')
        current = self.config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def get_expanded_path(self, path: str, default: Any = None) -> Optional[Path]:
        """
        Get a path configuration value with ~ expansion
        
        Args:
            path: Dot-separated path to config value
            default: Default value if path not found
            
        Returns:
            Expanded Path object or None
        """
        value = self.get(path, default)
        if value is None:
            return None
        return Path(value).expanduser()
    
    def get_output_dir(self, mode: str) -> Path:
        """
        Get output directory for a specific test mode, with automatic fallback
        
        Args:
            mode: Test mode ("single-prompt", "cache-rate", or "working-set")
            
        Returns:
            Expanded Path object
        """
        # Map mode to config key
        mode_key = mode.replace("-", "_")
        specific_dir = self.get_expanded_path(f"output.{mode_key}_dir")
        
        # If specific directory is set, use it
        if specific_dir:
            return specific_dir
        
        # Otherwise, use base_dir/mode
        base_dir = self.get_expanded_path("output.base_dir", "./results")
        return base_dir / mode_key.replace("_", "-")
    
    def to_dict(self) -> Dict:
        """Return the full configuration as a dictionary"""
        return self._deep_copy(self.config)
    
    def print_config(self, section: Optional[str] = None):
        """Print current configuration (for debugging)"""
        import json
        if section:
            config_to_print = self.get(section, {})
        else:
            config_to_print = self.config
        
        print(json.dumps(config_to_print, indent=2, default=str))
