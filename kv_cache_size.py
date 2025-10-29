#!/usr/bin/env python3
"""
KV Cache Size Calculator

A simple tool to calculate KV cache memory requirements for LLM models.
Downloads model config from HuggingFace and computes cache size for various token counts.

Usage:
    python kv_cache_size.py meta-llama/Llama-3.3-70B-Instruct
    python kv_cache_size.py Qwen/Qwen2.5-Coder-32B-Instruct
    python kv_cache_size.py --precision fp16 meta-llama/Llama-3.3-70B-Instruct

Version: 1.0
Date: 2025-01-17
"""

import argparse
import sys
from typing import Dict, Optional

try:
    from transformers import AutoConfig
except ImportError:
    print("Error: transformers library not found")
    print("Please install: pip install transformers")
    sys.exit(1)


def format_bytes(bytes_size: int) -> str:
    """Format bytes as human-readable string"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


def get_model_config(model_id: str) -> Dict:
    """
    Download and parse model config from HuggingFace

    Args:
        model_id: HuggingFace model identifier

    Returns:
        Dictionary with model configuration parameters
    """
    print(f"📥 Downloading model config: {model_id}")

    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        print(f"❌ Failed to download model config: {e}")
        print(f"\nPlease verify:")
        print(f"  1. Model ID is correct: {model_id}")
        print(f"  2. You have internet connection")
        print(f"  3. Model exists on HuggingFace Hub")
        sys.exit(1)

    # Extract relevant parameters
    model_config = {
        'model_id': model_id,
        'num_layers': None,
        'num_kv_heads': None,
        'head_dim': None,
        'hidden_size': None,
        'num_attention_heads': None,
    }

    # Try different config key names (different models use different names)
    if hasattr(config, 'num_hidden_layers'):
        model_config['num_layers'] = config.num_hidden_layers
    elif hasattr(config, 'n_layer'):
        model_config['num_layers'] = config.n_layer

    if hasattr(config, 'num_key_value_heads'):
        model_config['num_kv_heads'] = config.num_key_value_heads
    elif hasattr(config, 'num_kv_heads'):
        model_config['num_kv_heads'] = config.num_kv_heads
    elif hasattr(config, 'n_head_kv'):
        model_config['num_kv_heads'] = config.n_head_kv
    else:
        # If not specified, assume same as attention heads (no GQA)
        if hasattr(config, 'num_attention_heads'):
            model_config['num_kv_heads'] = config.num_attention_heads
        elif hasattr(config, 'n_head'):
            model_config['num_kv_heads'] = config.n_head

    if hasattr(config, 'num_attention_heads'):
        model_config['num_attention_heads'] = config.num_attention_heads
    elif hasattr(config, 'n_head'):
        model_config['num_attention_heads'] = config.n_head

    if hasattr(config, 'hidden_size'):
        model_config['hidden_size'] = config.hidden_size
    elif hasattr(config, 'd_model'):
        model_config['hidden_size'] = config.d_model

    # Use head_dim from config (trust the model's config file)
    if hasattr(config, 'head_dim'):
        model_config['head_dim'] = config.head_dim
    elif model_config['hidden_size'] and model_config['num_attention_heads']:
        # Calculate as fallback if not in config
        model_config['head_dim'] = model_config['hidden_size'] // model_config['num_attention_heads']

    # Validate we got all required parameters
    if not all([model_config['num_layers'], model_config['num_kv_heads'], model_config['head_dim']]):
        print(f"❌ Could not extract all required parameters from model config")
        print(f"\nExtracted config:")
        for key, value in model_config.items():
            if value is not None:
                print(f"  {key}: {value}")
        print(f"\nMissing parameters - please check model config manually")
        sys.exit(1)

    print(f"✅ Model config loaded successfully\n")
    return model_config


def calculate_kv_cache_size(
    num_tokens: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    precision: str = 'fp8',
    batch_size: int = 1
) -> Dict:
    """
    Calculate KV cache memory requirements

    Formula: 2 (K and V) × batch_size × num_kv_heads × seq_length × head_dim × bytes_per_element × num_layers

    Args:
        num_tokens: Sequence length in tokens
        num_layers: Number of transformer layers
        num_kv_heads: Number of KV heads (GQA/MQA)
        head_dim: Dimension of each attention head
        precision: 'fp8' (1 byte) or 'fp16' (2 bytes)
        batch_size: Batch size (default: 1)

    Returns:
        Dictionary with memory breakdown
    """
    bytes_per_element = 1 if precision == 'fp8' else 2

    # Per-layer KV cache size
    per_layer_bytes = (
        2 *  # K and V
        batch_size *
        num_kv_heads *
        num_tokens *
        head_dim *
        bytes_per_element
    )

    # Total across all layers
    total_bytes = per_layer_bytes * num_layers

    return {
        'num_tokens': num_tokens,
        'precision': precision,
        'bytes_per_element': bytes_per_element,
        'per_layer_bytes': per_layer_bytes,
        'per_layer_mb': per_layer_bytes / (1024 ** 2),
        'per_layer_gb': per_layer_bytes / (1024 ** 3),
        'total_bytes': total_bytes,
        'total_mb': total_bytes / (1024 ** 2),
        'total_gb': total_bytes / (1024 ** 3),
        'num_layers': num_layers,
        'num_kv_heads': num_kv_heads,
        'head_dim': head_dim,
        'batch_size': batch_size
    }


def main():
    parser = argparse.ArgumentParser(
        description="Calculate KV cache memory requirements for LLM models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python kv_cache_size.py meta-llama/Llama-3.3-70B-Instruct
  python kv_cache_size.py Qwen/Qwen2.5-Coder-32B-Instruct
  python kv_cache_size.py --precision fp16 meta-llama/Llama-3.3-70B-Instruct
  python kv_cache_size.py --batch-size 4 Qwen/Qwen2.5-Coder-32B-Instruct
        """
    )

    parser.add_argument(
        'model_id',
        type=str,
        help='HuggingFace model identifier (e.g., meta-llama/Llama-3.3-70B-Instruct)'
    )
    parser.add_argument(
        '--precision',
        type=str,
        choices=['fp8', 'fp16'],
        default='fp8',
        help='KV cache precision: fp8 (1 byte) or fp16 (2 bytes). Default: fp8'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size for calculation. Default: 1'
    )
    parser.add_argument(
        '--custom-tokens',
        type=int,
        nargs='+',
        help='Custom token counts to calculate (e.g., --custom-tokens 5000 50000 500000)'
    )

    args = parser.parse_args()

    # Header
    print("=" * 80)
    print("KV Cache Size Calculator")
    print("=" * 80)
    print()

    # Download and parse model config
    config = get_model_config(args.model_id)

    # Display model architecture
    print("📊 Model Architecture")
    print("-" * 80)
    print(f"Model: {config['model_id']}")
    print(f"Layers: {config['num_layers']}")
    print(f"KV Heads: {config['num_kv_heads']}")
    if config['num_attention_heads']:
        print(f"Attention Heads: {config['num_attention_heads']}")
        if config['num_kv_heads'] < config['num_attention_heads']:
            print(f"GQA Ratio: {config['num_attention_heads'] // config['num_kv_heads']}:1 (Grouped Query Attention)")
    print(f"Head Dimension: {config['head_dim']}")
    if config['hidden_size']:
        print(f"Hidden Size: {config['hidden_size']}")
    print(f"Precision: {args.precision.upper()}")
    print(f"Batch Size: {args.batch_size}")
    print()

    # Calculate for standard token counts
    token_counts = args.custom_tokens if args.custom_tokens else [1, 1000, 10000, 100000]

    print("💾 KV Cache Memory Requirements")
    print("-" * 80)
    print()

    for num_tokens in token_counts:
        result = calculate_kv_cache_size(
            num_tokens=num_tokens,
            num_layers=config['num_layers'],
            num_kv_heads=config['num_kv_heads'],
            head_dim=config['head_dim'],
            precision=args.precision,
            batch_size=args.batch_size
        )

        # Format output based on size
        if result['total_bytes'] < 1024 * 1024:  # < 1 MB
            size_str = f"{result['total_bytes']:,.0f} bytes"
        elif result['total_bytes'] < 1024 * 1024 * 1024:  # < 1 GB
            size_str = f"{result['total_mb']:.2f} MB"
        else:
            size_str = f"{result['total_gb']:.3f} GB ({result['total_mb']:.2f} MB)"

        print(f"📏 {num_tokens:,} tokens:")
        print(f"   Per-layer: {format_bytes(result['per_layer_bytes'])}")
        print(f"   Total:     {size_str}")

        # Calculate bytes per token for convenience
        bytes_per_token = result['total_bytes'] / num_tokens
        print(f"   Per-token: {format_bytes(int(bytes_per_token))}")
        print()

    # Summary
    print("=" * 80)
    print("💡 Notes:")
    print("   • KV cache grows linearly with sequence length")
    print("   • Cache is per request (multiply by concurrent requests for total)")
    print("   • Actual cache size depends on server configuration")
    print(f"   • FP8 uses half the memory of FP16 ({args.precision.upper()} selected)")
    if config['num_kv_heads'] < config.get('num_attention_heads', config['num_kv_heads']):
        print(f"   • Model uses GQA (Grouped Query Attention) for memory efficiency")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
