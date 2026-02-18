#!/bin/bash

# Source user's .env for API keys
if [ -f /home/leonb/maei/.env ]; then
    export $(grep -v '^#' /home/leonb/maei/.env | xargs)
fi

# Configure models to use OpenRouter
export ELENCHUS_MODEL_FAST=openrouter/qwen/qwen3-32b
export ELENCHUS_MODEL_CAPABLE=openrouter/deepseek/deepseek-r1-0528

# Ensure OpenRouter key is available to LiteLLM
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "Error: OPENROUTER_API_KEY not found in /home/leonb/maei/.env"
    exit 1
fi

echo "Starting benchmark with OpenRouter:"
echo "  FAST: $ELENCHUS_MODEL_FAST"
echo "  CAPABLE: $ELENCHUS_MODEL_CAPABLE"

# Run benchmark
.venv/bin/python scripts/benchmark_probe.py \
    --concurrency 2 \
    --output benchmark_instruct.json \
    --limit 50 \
    > benchmark_instruct.log 2>&1
