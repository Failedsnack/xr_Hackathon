#!/usr/bin/env bash
# Single-video inference:  ./run.sh <video.mp4> <out.json>
set -euo pipefail
if [ "$#" -ne 2 ]; then
  echo "usage: ./run.sh <video.mp4> <out.json>" >&2
  exit 2
fi
DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$DIR/code${PYTHONPATH:+:$PYTHONPATH}"
exec python3 -m vsum "$1" "$2"
