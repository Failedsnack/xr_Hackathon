"""Video probing and chunked frame decoding through ffmpeg (full-resolution frames are never stored)."""
from __future__ import annotations

import json
import subprocess
import tempfile
from typing import Iterator

import numpy as np

from . import config as C


def probe_duration(path: str) -> float:
    """Container duration in seconds (falls back to the video stream duration)."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    info = json.loads(out)
    dur = info.get("format", {}).get("duration")
    if dur is None:
        for s in info.get("streams", []):
            if s.get("codec_type") == "video" and s.get("duration"):
                dur = s["duration"]
                break
    if dur is None:
        raise RuntimeError(f"could not read the duration of {path}")
    return float(dur)


def _spawn(path: str, fps: float, size: int, hwaccel: bool, err):
    cmd = ["ffmpeg", "-v", "error", "-nostdin"]
    if hwaccel:
        cmd += ["-hwaccel", "auto"]
    cmd += ["-i", str(path), "-an", "-sn", "-dn",
            "-vf", f"fps={fps},scale={size}:{size}:flags=bilinear",
            "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err)


def _read_exact(stream, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        b = stream.read(n - len(buf))
        if not b:
            break
        buf += b
    return bytes(buf)


def iter_frames(path: str, fps: float = C.FPS, size: int = C.FRAME, chunk: int = C.CHUNK) -> Iterator[np.ndarray]:
    """Yield uint8 arrays [k, size, size, 3] (k <= chunk) of frames sampled at `fps`.

    Frame g covers [g/fps, (g+1)/fps). Hardware decoding is tried first and dropped if it fails.
    """
    fsz = size * size * 3
    with tempfile.TemporaryFile() as err:
        for hw in (True, False):
            err.seek(0)
            err.truncate()
            proc = _spawn(path, fps, size, hw, err)
            try:
                data = _read_exact(proc.stdout, fsz * chunk)
                if not data:
                    proc.wait()
                    if hw:
                        continue  # retry in software
                    err.seek(0)
                    raise RuntimeError("ffmpeg decoded no frames: " + err.read().decode(errors="replace")[-400:])
                while data:
                    k = len(data) // fsz
                    if k:
                        yield np.frombuffer(data, np.uint8, count=k * fsz).reshape(k, size, size, 3)
                    if len(data) < fsz * chunk:
                        break
                    data = _read_exact(proc.stdout, fsz * chunk)
                return
            finally:
                if proc.poll() is None:
                    proc.kill()
                proc.stdout.close()
                proc.wait()
