from __future__ import annotations

import json
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

import jiwer
import numpy as np


def normalize_japanese(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return "".join(
        char
        for char in normalized
        if not char.isspace() and not unicodedata.category(char).startswith("P")
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def score_file(prediction_file: Path, run_metadata: dict[str, Any]) -> dict[str, Any]:
    rows = _read_jsonl(prediction_file)
    if not rows:
        raise ValueError(f"No predictions in {prediction_file}")
    references = [normalize_japanese(row["reference"]) for row in rows]
    hypotheses = [normalize_japanese(row["hypothesis"]) for row in rows]
    if any(not reference for reference in references):
        raise ValueError(f"Empty normalized reference in {prediction_file}")
    latencies = np.asarray([row["latency_seconds"] for row in rows], dtype=float)
    audio_seconds = sum(float(row["audio_seconds"]) for row in rows)
    inference_seconds = sum(float(row["latency_seconds"]) for row in rows)
    return {
        "dataset": run_metadata["dataset"],
        "model": run_metadata["model"],
        "num_examples": len(rows),
        "audio_hours": audio_seconds / 3600,
        "cer": jiwer.cer(references, hypotheses),
        "model_load_seconds": run_metadata.get("model_load_seconds"),
        "inference_seconds": inference_seconds,
        "rtf": inference_seconds / audio_seconds if audio_seconds else None,
        "latency_p50_seconds": float(np.percentile(latencies, 50)),
        "latency_p95_seconds": float(np.percentile(latencies, 95)),
        "peak_rss_mb": run_metadata.get("peak_rss_mb"),
        "peak_gpu_memory_mb": run_metadata.get("peak_gpu_memory_mb"),
        "device": run_metadata.get("device"),
        "model_revision": run_metadata.get("model_revision"),
    }


def score_results(results_root: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for prediction_file in sorted(results_root.glob("*/*/predictions.jsonl")):
        metadata_file = prediction_file.parent / "run.json"
        if not metadata_file.is_file():
            continue
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        if not metadata.get("complete") or metadata.get("limited"):
            continue
        summaries.append(score_file(prediction_file, metadata))
    if not summaries:
        raise FileNotFoundError(f"No completed runs found under {results_root}")
    output_path = results_root / "summary.json"
    output_path.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summaries
