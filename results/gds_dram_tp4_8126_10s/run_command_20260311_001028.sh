#!/bin/bash
# Working Set Tester - Run command
# Generated: 2026-03-11 00:10:28
# To re-run this exact test configuration, execute:
#   bash run_command_20260311_001028.sh
# or
#   bash run_command_20260311_001028.sh --force-restart  # to restart from beginning

python working_set_tester.py \
  --api-endpoint http://localhost:8126 \
  --context-sizes 64000 \
  --min-working-set-size 600000 \
  --max-working-set-size 5000000 \
  --working-set-increments 20 \
  --output-tokens 1 \
  --max-ttft 10.0 \
  --output-dir results/gds_dram_tp4_8126_10s \
  --tokenizer Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
  --test-duration 1800 \
  --cache-hit-rates 96 \
  --random-working-set-selection \
  --num-retries 0 \
  --start-concurrency 1 $@
