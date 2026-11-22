#!/usr/bin/env bash
# Single-video inference: ./run.sh <video.mp4> <out.json>
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
python3 "$DIR/code/vsum.py" "$1" "$2"
