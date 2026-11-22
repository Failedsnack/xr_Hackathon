# Reproducible environment for the Track 04 submission.
#
#   docker build -t xr-vsum .
#   docker run --rm -v "$PWD/videos":/in:ro -v "$PWD/out":/out xr-vsum /in/<video>.mp4 /out/<video>.json
#
# The CLIP checkpoint and the text-embedding cache are downloaded at *build* time, so running the
# container needs no network access.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch first (the default Linux wheel pulls ~2 GB of CUDA libraries), then the rest.
COPY requirements.txt .
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

COPY code/ code/
COPY data/ data/
COPY weights/epic_pair_prior.json weights/epic_pair_prior.json
COPY run.sh team_id.txt ./
RUN chmod +x run.sh

# Bake the pretrained weights and all text embeddings into the image.
RUN PYTHONPATH=code python -m vsum.warmup

ENTRYPOINT ["./run.sh"]
