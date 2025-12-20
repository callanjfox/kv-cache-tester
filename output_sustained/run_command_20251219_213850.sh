#!/bin/bash
# Cache Rate Tester - Run command
# Generated: 2025-12-19 21:38:50
# To re-run this exact test configuration, execute:
#   bash run_command_20251219_213850.sh
# or
#   bash run_command_20251219_213850.sh --force-restart  # to restart from beginning

python cache_rate_tester.py \
  --api-endpoint http://localhost:8125 \
  --context-sizes 8000 \
  --working-set-size 500000 \
  --max-ttft 5.0 \
  --output-dir ./output_sustained \
  --test-duration 30 \
  --cache-hit-rates 50 \
  --skip-graphs $@
