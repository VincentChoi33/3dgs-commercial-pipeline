"""Step 1: Video frame extraction and blur filtering."""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from pipeline.ingest import extract_frames as ingest_extract_frames
from pipeline.ingest import is_video

log = logging.getLogger("pipeline.preprocess")


def compute_blur_score(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def extract_frames(video_path: Path, output_dir: Path, fps: int = 2) -> Path:
    extracted_dir = ingest_extract_frames(video_path, output_dir, fps=fps)
    n = len(list(extracted_dir.glob("frame_*.jpg")))
    log.info(f"Extracted {n} frames → {extracted_dir}")
    return extracted_dir


def filter_blurry_frames(images_dir: Path, percentile: int = 20) -> list[Path]:
    files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    if not files:
        raise FileNotFoundError(f"No images in {images_dir}")

    scores = []
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            raise ValueError(f"Unable to read image: {f}")
        scores.append((f, compute_blur_score(img)))

    threshold = np.percentile([s for _, s in scores], percentile)
    kept = [f for f, s in scores if s >= threshold]
    removed = len(files) - len(kept)

    for f, s in scores:
        if s < threshold:
            f.unlink()

    log.info(f"Blur filter: {len(files)} → {len(kept)} frames ({removed} removed, threshold={threshold:.1f})")
    return kept


def run_preprocess(input_path: Path, output_dir: Path, cfg: dict) -> Path:
    from pipeline.ingest import run_ingest
    from pipeline.select import run_select

    run_ingest(input_path, output_dir, cfg)
    return run_select(output_dir, cfg)
