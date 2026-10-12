from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .data import PROJECT_ROOT, prepare_dataset
from .inference import run_model
from .metrics import score_results


def _choices(section: str) -> list[str]:
    with (PROJECT_ROOT / "benchmarks" / "config.yaml").open(encoding="utf-8") as f:
        return list(yaml.safe_load(f)[section])


def _root(value: str | None, default: str) -> Path:
    path = Path(value or default)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _add_roots(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", help="ローカルデータ保存先 (default: data)")
    parser.add_argument("--results-root", help="予測・結果の保存先 (default: results)")


def main() -> None:
    parser = argparse.ArgumentParser(prog="asrbench-ja")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="音声を共通形式にして manifest を作る")
    prepare.add_argument("dataset", choices=_choices("datasets"))
    prepare.add_argument("--source-path", help="Common Voice / TEDxJP-10K の入力ディレクトリ")
    prepare.add_argument("--force", action="store_true", help="既存の manifest と正規化音声を作り直す")
    prepare.add_argument("--data-root", help="ローカルデータ保存先 (default: data)")

    run = commands.add_parser("run", help="指定モデルで推論する")
    run.add_argument("--dataset", required=True, choices=_choices("datasets"))
    run.add_argument("--model", required=True, choices=["all", *_choices("models")])
    run.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    run.add_argument("--limit", type=int, help="先頭 N 件だけ実行 (動作確認用)")
    _add_roots(run)

    score = commands.add_parser("score", help="完了した全件ベンチマークを採点する")
    score.add_argument("--results-root", help="予測・結果の保存先 (default: results)")

    args = parser.parse_args()
    if args.command == "prepare":
        data_root = _root(args.data_root, "data")
        source_path = Path(args.source_path).expanduser().resolve() if args.source_path else None
        manifest, count = prepare_dataset(args.dataset, data_root, source_path, args.force)
        print(f"{args.dataset}: {count} 件を準備しました ({manifest})")
        return

    if args.command == "run":
        if args.limit is not None and args.limit < 1:
            parser.error("--limit は 1 以上を指定してください")
        data_root = _root(args.data_root, "data")
        results_root = _root(args.results_root, "results")
        models = _choices("models") if args.model == "all" else [args.model]
        for model in models:
            output = run_model(
                args.dataset,
                model,
                data_root,
                results_root,
                args.device,
                args.limit,
            )
            print(f"{model}: {output}")
        return

    summaries = score_results(_root(args.results_root, "results"))
    print("dataset\tmodel\tCER (%)\tRTF\tp50 latency (s)\tpeak VRAM (MB)")
    for row in summaries:
        print(
            f"{row['dataset']}\t{row['model']}\t{row['cer'] * 100:.3f}\t"
            f"{row['rtf']:.4f}\t{row['latency_p50_seconds']:.3f}\t"
            f"{row['peak_gpu_memory_mb'] if row['peak_gpu_memory_mb'] is not None else 'n/a'}"
        )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
