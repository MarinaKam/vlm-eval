#!/bin/zsh
# Full local runs on Apple silicon via Ollama. Resumable: re-run after any interruption.
# Usage:  caffeinate -i zsh scripts/run_all_local.sh 2>&1 | tee -a runs/local_run.log
set -e
cd "$(dirname "$0")/.."
VE=.venv/bin/vlm-eval
OLLAMA=http://localhost:11434/v1

run_model () {  # $1 = run name, $2 = ollama model tag, $3 = bbox coords (abs for Qwen2.5-VL, norm1000 for Qwen3-VL)
  echo "=== $1 : tagging chunk15 full (1000) ==="
  $VE run --model "$1" --served-name "$2" --base-url $OLLAMA --flavor ollama --task tagging --no-logprobs
  echo "=== $1 : consistency x3 (first 100) ==="
  $VE run --model "$1" --served-name "$2" --base-url $OLLAMA --flavor ollama --task tagging --no-logprobs \
      --repeats 3 --limit 100
  echo "=== $1 : all-questions-in-one-call (first 300) ==="
  $VE run --model "$1" --served-name "$2" --base-url $OLLAMA --flavor ollama --task tagging --no-logprobs \
      --chunk 0 --limit 300
  echo "=== $1 : captions (1000) ==="
  $VE run --model "$1" --served-name "$2" --base-url $OLLAMA --flavor ollama --task captions
  echo "=== $1 : grounding (first 300) ==="
  $VE run --model "$1" --served-name "$2" --base-url $OLLAMA --flavor ollama --task grounding --limit 300 --coords "$3"
  echo "=== $1 : property summary (5 listings) ==="
  $VE run --model "$1" --served-name "$2" --base-url $OLLAMA --flavor ollama --task summary
  $VE metrics --model "$1"
}

run_model qwen2.5vl-7b-ollama qwen2.5vl:7b abs
run_model qwen3-vl-8b-ollama  qwen3-vl:8b norm1000
echo "ALL DONE"
