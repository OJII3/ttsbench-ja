from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import statistics
import threading
import time
from pathlib import Path
from typing import Any, Callable

import psutil
import yaml

from .data import PROJECT_ROOT, manifest_path, read_manifest


class ResourceSampler:
    def __init__(self, device: str):
        self.device = device
        self.process = psutil.Process()
        self.stop_event = threading.Event()
        self.peak_rss = 0
        self.peak_gpu = 0
        self.nvml = None
        self.gpu_handle = None
        if device == "cuda":
            try:
                import pynvml

                pynvml.nvmlInit()
                self.nvml = pynvml
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                self.nvml = None

    def _sample(self) -> None:
        self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
        if self.nvml is None or self.gpu_handle is None:
            return
        try:
            for process in self.nvml.nvmlDeviceGetComputeRunningProcesses(self.gpu_handle):
                if process.pid == os.getpid() and process.usedGpuMemory:
                    self.peak_gpu = max(self.peak_gpu, int(process.usedGpuMemory))
        except Exception:
            pass

    def _run(self) -> None:
        while not self.stop_event.wait(0.2):
            self._sample()

    def start(self) -> None:
        self._sample()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def finish(self) -> tuple[float, float | None]:
        self.stop_event.set()
        self.thread.join()
        self._sample()
        if self.nvml is not None:
            try:
                self.nvml.nvmlShutdown()
            except Exception:
                pass
        return (
            self.peak_rss / (1024**2),
            self.peak_gpu / (1024**2) if self.peak_gpu else None,
        )


def _versions() -> dict[str, str | None]:
    versions = {}
    for package in (
        "ctranslate2",
        "datasets",
        "faster-whisper",
        "huggingface-hub",
        "jiwer",
        "nemo-toolkit",
        "nvidia-ml-py",
        "pytorch-lightning",
        "torch",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _load_model(
    model_key: str,
    model_config: dict[str, Any],
    device: str,
    settings: dict[str, Any],
    model_path: Path,
) -> Callable[[str], str]:
    if model_config["backend"] == "nemo":
        import torch
        from nemo.collections.asr.models import ASRModel

        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        checkpoint = model_path / model_config["checkpoint"]
        model = ASRModel.restore_from(str(checkpoint), map_location=device)
        model.eval()
        if model_config.get("decoder") == "ctc" and hasattr(model, "cur_decoder"):
            model.change_decoding_strategy(decoder_type="ctc")

        def transcribe(audio_path: str) -> str:
            with torch.inference_mode():
                output = model.transcribe(
                    [audio_path], batch_size=1, return_hypotheses=False, verbose=False
                )[0]
            return output.text if hasattr(output, "text") else str(output)

        return transcribe

    if model_config["backend"] == "faster_whisper":
        from faster_whisper import WhisperModel
        compute_type = "float16" if device == "cuda" else "int8"
        model = WhisperModel(
            str(model_path),
            device=device,
            compute_type=compute_type,
            cpu_threads=1,
            num_workers=1,
        )

        def transcribe(audio_path: str) -> str:
            segments, _ = model.transcribe(
                audio_path,
                language=settings["language"],
                task=settings["task"],
                beam_size=settings["faster_whisper_beam_size"],
                temperature=0.0,
                vad_filter=False,
                initial_prompt=None,
                hotwords=None,
            )
            return "".join(segment.text for segment in segments)

        return transcribe

    raise ValueError(f"Unknown model backend: {model_config['backend']}")


def run_model(
    dataset: str,
    model_key: str,
    data_root: Path,
    results_root: Path,
    device_arg: str = "auto",
    limit: int | None = None,
) -> Path:
    with (PROJECT_ROOT / "benchmarks" / "config.yaml").open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    model_config = config["models"][model_key]
    model_settings = config["inference"]
    manifest = manifest_path(dataset, data_root)
    if not manifest.is_file():
        raise FileNotFoundError(f"Run prepare for {dataset!r} first: {manifest}")
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    device = device_arg
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and model_config["backend"] == "faster_whisper":
        compute_type = "float16"
    else:
        compute_type = model_settings["precision"]["nemo"] if model_config["backend"] == "nemo" else "int8"
    settings_sha256 = hashlib.sha256(
        json.dumps(
            {"model": model_config, "inference": model_settings},
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    output_dir = results_root / dataset / model_key
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_file = output_dir / "predictions.jsonl"
    metadata_file = output_dir / "run.json"
    expected_identity = {
        "dataset": dataset,
        "model": model_key,
        "model_repo": model_config["repo_id"],
        "model_revision": model_config["revision"],
        "device": device,
        "compute_type": compute_type,
        "manifest_sha256": manifest_hash,
        "settings_sha256": settings_sha256,
        "limit": limit,
    }
    if metadata_file.exists():
        existing = json.loads(metadata_file.read_text(encoding="utf-8"))
        for key, value in expected_identity.items():
            if existing.get(key) != value:
                raise ValueError(f"Existing run differs at {key}; use a new results directory")

    from huggingface_hub import snapshot_download

    download_start = time.perf_counter()
    download_kwargs = {
        "repo_id": model_config["repo_id"],
        "revision": model_config["revision"],
    }
    if model_config["backend"] == "nemo":
        download_kwargs["allow_patterns"] = [model_config["checkpoint"]]
    model_path = Path(snapshot_download(**download_kwargs))
    model_download_seconds = time.perf_counter() - download_start

    sampler = ResourceSampler(device)
    sampler.start()
    load_start = time.perf_counter()
    transcribe = _load_model(model_key, model_config, device, model_settings, model_path)
    model_load_seconds = time.perf_counter() - load_start
    records = list(read_manifest(manifest))
    if limit is not None:
        records = records[:limit]

    metadata = {
        **expected_identity,
        "backend": model_config["backend"],
        "decoder": model_config.get("decoder"),
        "compute_type": compute_type,
        "batch_size": model_settings["batch_size"],
        "inference_settings": model_settings,
        "versions": _versions(),
        "model_download_seconds": model_download_seconds,
        "model_load_seconds": model_load_seconds,
        "complete": False,
        "limited": limit is not None,
    }
    metadata_file.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    completed: set[str] = set()
    if prediction_file.exists():
        with prediction_file.open(encoding="utf-8") as f:
            completed = {json.loads(line)["record_id"] for line in f if line.strip()}
    pending = [record for record in records if record["record_id"] not in completed]
    warmup_examples = int(model_settings["warmup_examples"])
    for record in records[:warmup_examples]:
        transcribe(str(PROJECT_ROOT / record["audio_path"]))

    started = time.perf_counter()
    latencies: list[float] = []
    with prediction_file.open("a", encoding="utf-8") as output:
        for record in pending:
            audio_path = PROJECT_ROOT / record["audio_path"]
            before = time.perf_counter()
            hypothesis = transcribe(str(audio_path))
            latency = time.perf_counter() - before
            latencies.append(latency)
            result = {
                "record_id": record["record_id"],
                "dataset": dataset,
                "model": model_key,
                "audio_seconds": record["audio_seconds"],
                "latency_seconds": latency,
                "reference": record["reference"],
                "hypothesis": hypothesis,
            }
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
    inference_seconds = time.perf_counter() - started
    peak_rss_mb, peak_gpu_memory_mb = sampler.finish()
    completed_count = len(completed) + len(pending)
    if metadata_file.exists():
        previous_metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    else:
        previous_metadata = {}
    metadata.update(
        {
            "complete": completed_count >= len(records),
            "completed_examples": completed_count,
            "manifest_examples": len(records),
            "inference_seconds_this_run": inference_seconds,
            "model_load_seconds": model_load_seconds,
            "peak_rss_mb": max(peak_rss_mb, previous_metadata.get("peak_rss_mb") or 0),
            "peak_gpu_memory_mb": max(
                (peak_gpu_memory_mb or 0),
                (previous_metadata.get("peak_gpu_memory_mb") or 0),
            )
            or None,
            "latency_p50_this_run": statistics.median(latencies) if latencies else None,
        }
    )
    metadata_file.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return prediction_file
