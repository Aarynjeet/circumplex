#!/usr/bin/env bash
# Assemble a self-contained Hugging Face Space folder from a trained run.
# Usage: bash scripts/make_space.sh runs/roberta_lora_s42 space_out
set -euo pipefail
RUN=${1:?run directory, e.g. runs/roberta_lora_s42}
OUT=${2:-space_out}
rm -rf "$OUT" && mkdir -p "$OUT/model" "$OUT/data"
cp demo/app.py "$OUT/app.py"
cp demo/requirements.txt "$OUT/requirements.txt"
cp -r src "$OUT/src"
cp -r data/goemotions "$OUT/data/goemotions"
[ -f data/emotion_vad.csv ] && cp data/emotion_vad.csv "$OUT/data/"
cp "$RUN/config.yaml" "$RUN/best.pt" "$RUN/metrics.json" "$OUT/model/"
cp -r "$RUN/tokenizer" "$OUT/model/tokenizer"
cat > "$OUT/README.md" <<'MD'
---
title: Circumplex Emotion Demo
emoji: 🎭
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
---
Research demo of multi-label emotion detection trained on GoEmotions.
Not a mental-health or diagnostic tool.
MD
echo "Space folder ready in $OUT ($(du -sh "$OUT" | cut -f1))"
