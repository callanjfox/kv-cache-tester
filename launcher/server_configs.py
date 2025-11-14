#!/usr/bin/env python3
"""
Server Configuration Builder

Builds vLLM server configurations for different KV cache backends:
- Dynamo (CPU and WEKA disk caches)
- LMCache (DRAM, WEKA/GDS, and NIXL backends)
- Vanilla vLLM (with and without prefix caching)
"""

import json
from typing import List
from dataclasses import dataclass, asdict
from launcher.config_manager import ConfigManager


@dataclass
class ServerConfig:
    """Configuration for a vLLM server instance"""
    name: str
    backend: str
    cache_type: str
    env_vars: dict[str, str]
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
