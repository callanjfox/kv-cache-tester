#!/bin/bash
# Cache Rate Tester - Run command
# Generated: 2025-12-19 21:40:24
# To re-run this exact test configuration, execute:
#   bash run_command_20251219_214024.sh
# or
#   bash run_command_20251219_214024.sh --force-restart  # to restart from beginning

python cache_rate_tester.py \
  --api-endpoint http://localhost:8125 \
  --context-sizes 8000 \
  --working-set-size 500000 \
  --max-ttft 5.0 \
  --output-dir ./output_fixed \
  --test-duration 30 \
  --ramp-duration 10 \
  --cache-hit-rates 50 \
  --skip-graphs $@
