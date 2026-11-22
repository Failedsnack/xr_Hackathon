#!/usr/bin/env bash
# One-time setup (macOS or Linux): virtual env, dependencies, CLIP weights, EPIC downloader.
#   ./tools/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."

command -v ffmpeg  >/dev/null || { echo "ffmpeg is missing. macOS: brew install ffmpeg"; exit 1; }
command -v ffprobe >/dev/null || { echo "ffprobe is missing (it ships with ffmpeg)"; exit 1; }

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-tools.txt

if [ ! -d third_party/epic-kitchens-download-scripts ]; then
  git clone --depth 1 https://github.com/epic-kitchens/epic-kitchens-download-scripts.git \
      third_party/epic-kitchens-download-scripts
fi

# downloads the CLIP checkpoint into weights/ and caches all text embeddings
PYTHONPATH=code python -m vsum.warmup

echo
echo "Ready. Next:"
echo "  source .venv/bin/activate"
echo "  python tools/run_all.py --dry-run"
