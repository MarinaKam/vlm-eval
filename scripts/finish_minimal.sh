#!/bin/zsh
# Minimal top-up so every capability has a measured value for both models, then stop.
# ~2 hours. Run after interrupting the full script; everything already computed is reused.
#   caffeinate -i zsh scripts/finish_minimal.sh 2>&1 | tee -a runs/local_run.log
set -e
cd "$(dirname "$0")/.."
VE=.venv/bin/vlm-eval
OLLAMA=http://localhost:11434/v1

echo "=== fetching listing images (needed for the multi-image summary) ==="
$VE download

Q3=(--model qwen3-vl-8b-ollama --served-name qwen3-vl:8b --base-url $OLLAMA --flavor ollama)
echo "=== qwen3-vl-8b: captions (100) ==="
$VE run "${Q3[@]}" --task captions --limit 100
echo "=== qwen3-vl-8b: grounding (100) ==="
$VE run "${Q3[@]}" --task grounding --limit 100 --coords norm1000
echo "=== qwen3-vl-8b: all-questions-in-one-call (60) ==="
$VE run "${Q3[@]}" --task tagging --no-logprobs --chunk 0 --limit 60
echo "=== qwen3-vl-8b: property summary ==="
$VE run "${Q3[@]}" --task summary
$VE metrics --model qwen3-vl-8b-ollama

# Last: switching models makes Ollama reload weights, so keep it to one switch at the end.
Q25=(--model qwen2.5vl-7b-ollama --served-name qwen2.5vl:7b --base-url $OLLAMA --flavor ollama)
echo "=== qwen2.5vl-7b: property summary (redone with the full image sets) ==="
rm -f runs/qwen2.5vl-7b-ollama/summary.jsonl
$VE run "${Q25[@]}" --task summary
$VE metrics --model qwen2.5vl-7b-ollama

echo "MINIMAL RUN COMPLETE"
