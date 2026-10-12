from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable

import soundfile as sf
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config() -> dict[str, Any]:
    with (PROJECT_ROOT / "benchmarks" / "config.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def manifest_path(dataset: str, data_root: Path) -> Path:
    return data_root / "manifests" / f"{dataset}.jsonl"


def _safe_name(dataset: str, source_id: str) -> str:
    digest = hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:16]
    readable = re.sub(r"[^A-Za-z0-9_-]+", "_", source_id).strip("_")[:40]
    return f"{dataset}-{readable or 'record'}-{digest}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ffmpeg_convert(source: str | Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-map_metadata",
            "-1",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(destination),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def _audio_to_temp_file(audio: Any, temp_dir: Path) -> str | Path:
    if isinstance(audio, list):
        if len(audio) != 1:
            raise ValueError(f"Expected one audio channel/entry, got {len(audio)}")
        audio = audio[0]
    if isinstance(audio, dict):
        path = audio.get("path")
        if path and Path(path).is_file():
            return Path(path)
        source_url = audio.get("src")
        if source_url:
            return source_url
        payload = audio.get("bytes")
        if payload:
            path = temp_dir / "source-audio"
            path.write_bytes(payload)
            return path
        if "array" in audio:
            path = temp_dir / "source-audio.wav"
            sf.write(path, audio["array"], int(audio["sampling_rate"]), subtype="PCM_16")
            return path
    raise ValueError("Dataset row has no readable audio payload")


def _write_record(
    dataset: str,
    index: int,
    source_id: str,
    reference: str,
    source: str | Path,
    audio_dir: Path,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record_id = _safe_name(dataset, source_id)
    audio_path = audio_dir / f"{record_id}.wav"
    if not audio_path.exists():
        _ffmpeg_convert(source, audio_path)
    info = sf.info(audio_path)
    if info.samplerate != 16000 or info.channels != 1 or info.subtype != "PCM_16":
        raise ValueError(f"Unexpected normalized audio format: {audio_path}")
    try:
        stored_audio_path = str(audio_path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        stored_audio_path = str(audio_path.resolve())
    record = {
        "record_id": record_id,
        "dataset": dataset,
        "index": index,
        "audio_path": stored_audio_path,
        "audio_sha256": _sha256_file(audio_path),
        "audio_seconds": info.duration,
        "reference": reference,
        "source_id": source_id,
    }
    if extra:
        record.update(extra)
    return record


def _prepare_huggingface(
    dataset: str,
    dataset_config: dict[str, Any],
    audio_dir: Path,
    manifest_file,
) -> int:
    from datasets import Audio, load_dataset

    kwargs: dict[str, Any] = {
        "path": dataset_config["repo_id"],
        "split": dataset_config["split"],
        "revision": dataset_config["revision"],
        "streaming": True,
    }
    if dataset_config.get("config"):
        kwargs["name"] = dataset_config["config"]
    rows = load_dataset(**kwargs)
    if "audio" in rows.features:
        rows = rows.cast_column("audio", Audio(decode=False))
    count = 0
    for index, row in enumerate(rows):
        reference = row.get("transcription") or row.get("sentence")
        if not reference:
            raise ValueError(f"No transcription in {dataset} row {index}")
        source_id = str(row.get("id") or row.get("path") or index)
        with tempfile.TemporaryDirectory(prefix="ttsbench-audio-") as temp:
            source = _audio_to_temp_file(row["audio"], Path(temp))
            record = _write_record(
                dataset,
                index,
                source_id,
                str(reference),
                source,
                audio_dir,
                {"source_repo": dataset_config["repo_id"], "source_revision": dataset_config["revision"]},
            )
        manifest_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        count += 1
    return count


def _prepare_common_voice(
    dataset: str,
    dataset_config: dict[str, Any],
    source_root: Path,
    audio_dir: Path,
    manifest_file,
) -> int:
    split_file = source_root / dataset_config["split_file"]
    clips_dir = source_root / "clips"
    if not split_file.is_file() or not clips_dir.is_dir():
        raise FileNotFoundError(f"Expected {split_file} and {clips_dir}")
    count = 0
    with split_file.open(encoding="utf-8", newline="") as f:
        for index, row in enumerate(csv.DictReader(f, delimiter="\t")):
            clip_path = (clips_dir / row["path"]).resolve()
            if not clip_path.is_relative_to(clips_dir.resolve()):
                raise ValueError(f"Clip path escapes dataset root: {row['path']}")
            record = _write_record(
                dataset,
                index,
                row["path"],
                row["sentence"],
                clip_path,
                audio_dir,
                {"source_version": dataset_config["version"]},
            )
            manifest_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def _read_kaldi_text(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            key, separator, value = line.rstrip("\n").partition(" ")
            if separator:
                values[key] = value
    return values


def _prepare_tedx(
    dataset: str,
    dataset_config: dict[str, Any],
    source_root: Path,
    audio_dir: Path,
    manifest_file,
) -> int:
    texts = _read_kaldi_text(source_root / "text")
    wavs: dict[str, str] = {}
    with (source_root / "wavlist.txt").open(encoding="utf-8") as f:
        for line in f:
            fields = line.split()
            if len(fields) >= 2:
                wavs[fields[0]] = fields[1]

    count = 0
    with (source_root / "segments").open(encoding="utf-8") as f:
        for index, line in enumerate(f):
            fields = line.split(maxsplit=3)
            if len(fields) != 4:
                continue
            utterance_id, recording_id, start_text, end_text = fields
            if utterance_id not in texts or recording_id not in wavs:
                continue
            start, end = float(start_text), float(end_text)
            source = source_root / "wav" / wavs[recording_id]
            audio_path = audio_dir / f"{_safe_name(dataset, utterance_id)}.wav"
            if not audio_path.exists():
                info = sf.info(source)
                if info.samplerate != 16000:
                    raise ValueError(f"TEDxJP-10K v1.1 source must be 16 kHz: {source}")
                with sf.SoundFile(source) as audio_file:
                    audio_file.seek(round(start * info.samplerate))
                    samples = audio_file.read(
                        max(1, round((end - start) * info.samplerate)),
                        dtype="float32",
                        always_2d=True,
                    )
                mono = samples.mean(axis=1) if samples.shape[1] > 1 else samples[:, 0]
                audio_path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(audio_path, mono, 16000, subtype="PCM_16")
            info = sf.info(audio_path)
            record = {
                "record_id": _safe_name(dataset, utterance_id),
                "dataset": dataset,
                "index": index,
                "audio_path": str(audio_path.resolve().relative_to(PROJECT_ROOT))
                if audio_path.resolve().is_relative_to(PROJECT_ROOT)
                else str(audio_path.resolve()),
                "audio_sha256": _sha256_file(audio_path),
                "audio_seconds": info.duration,
                "reference": texts[utterance_id],
                "source_id": utterance_id,
                "source_version": dataset_config["version"],
            }
            manifest_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def prepare_dataset(
    dataset: str,
    data_root: Path,
    source_override: Path | None = None,
    force: bool = False,
) -> tuple[Path, int]:
    config = load_config()["datasets"]
    if dataset not in config:
        raise ValueError(f"Unknown dataset {dataset!r}; choose from {', '.join(config)}")
    dataset_config = config[dataset]
    target_manifest = manifest_path(dataset, data_root)
    audio_dir = data_root / "prepared" / dataset
    if target_manifest.exists() and not force:
        raise FileExistsError(f"{target_manifest} exists; pass --force to rebuild it")
    if force:
        target_manifest.unlink(missing_ok=True)
        shutil.rmtree(audio_dir, ignore_errors=True)
    target_manifest.parent.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    temp_manifest = target_manifest.with_suffix(".jsonl.tmp")
    try:
        with temp_manifest.open("w", encoding="utf-8") as f:
            source = dataset_config["source"]
            if source == "huggingface":
                count = _prepare_huggingface(dataset, dataset_config, audio_dir, f)
            elif source == "local_common_voice":
                root = source_override or (PROJECT_ROOT / dataset_config["required_path"])
                count = _prepare_common_voice(dataset, dataset_config, root, audio_dir, f)
            elif source == "local_tedxjp10k":
                root = source_override or (PROJECT_ROOT / dataset_config["required_path"])
                count = _prepare_tedx(dataset, dataset_config, root, audio_dir, f)
            else:
                raise ValueError(f"Unknown dataset source {source!r}")
        temp_manifest.replace(target_manifest)
    except Exception:
        temp_manifest.unlink(missing_ok=True)
        raise
    return target_manifest, count


def read_manifest(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
