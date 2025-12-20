#!/bin/bash
# Cache Rate Tester - Run command
# Generated: 2025-12-19 20:45:23
# To re-run this exact test configuration, execute:
#   bash run_command_20251219_204523.sh
# or
#   bash run_command_20251219_204523.sh --force-restart  # to restart from beginning

python cache_rate_tester.py \
  --api-endpoint http://localhost:8125 \
  --context-sizes 64000 \
  --working-set-size 1000000 \
  --output-tokens 1 \
  --max-ttft 5.0 \
  --output-dir ./cacherate_sustained--22 \
  --reinit-strategy per_cache_rate \
  --random-working-set-selection \
  --cache-hit-rates 95 $@
