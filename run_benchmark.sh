#!/bin/bash
export OPENAI_API_KEY=maei-local
export OPENAI_API_BASE=http://localhost:8317/v1
export ELENCHUS_MODEL_FAST=openai/gemini-2.5-flash
export ELENCHUS_MODEL_CAPABLE=openai/gemini-2.5-pro

# Print config for debugging
echo "Starting benchmark with:"
echo "  FAST: $ELENCHUS_MODEL_FAST"
echo "  CAPABLE: $ELENCHUS_MODEL_CAPABLE"
echo "  BASE: $OPENAI_API_BASE"

# Run benchmark
.venv/bin/python scripts/benchmark_probe.py \
    --concurrency 2 \
    --output benchmark_instruct.json \
    --limit 50 \
    > benchmark_instruct.log 2>&1
