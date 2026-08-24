#!/bin/zsh
# Florence-2 / InternVL3.5 / PaliGemma2 local runs (transformers on MPS). Run AFTER the Ollama
# script or in parallel (separate compute paths, but both are heavy — sequential is safer).
# One-time setup:   uv pip install -e ".[hf]"
# PaliGemma extra:  accept the Gemma licence at https://huggingface.co/google/paligemma2-3b-mix-448
#                   then `export HF_TOKEN=...` (read token) before running.
# Resumable; re-run after interruption.
set -e
cd "$(dirname "$0")/.."
VE=.venv/bin/vlm-eval

echo "=== Florence-2-large: captions (300) ==="
$VE florence --task captions --limit 300
echo "=== Florence-2-large: grounding (300) ==="
$VE florence --task grounding --limit 300
echo "=== Florence-2-large: tagging via OVD (100) ==="
$VE florence --task tagging --limit 100
$VE metrics --model florence-2-large

echo "=== InternVL3.5-8B: tagging (150) ==="
$VE hf --backend internvl --task tagging --limit 150
echo "=== InternVL3.5-8B: captions (150) ==="
$VE hf --backend internvl --task captions --limit 150
echo "=== InternVL3.5-8B: summary (5 listings) ==="
$VE hf --backend internvl --task summary

echo "=== PaliGemma2-3b-mix: tagging (150) ==="
$VE hf --backend paligemma --task tagging --limit 150
echo "=== PaliGemma2-3b-mix: captions (150) ==="
$VE hf --backend paligemma --task captions --limit 150
echo "ALL HF MODELS DONE"
