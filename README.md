# ttsbench-ja

再現可能な日本語 ASR ベンチマークです。Parakeet の日本語モデルと faster-whisper の `large-v3-turbo` を、固定した音声、モデル revision、前処理、採点条件で比較します。

## 比較条件

- データセット: FLEURS `ja_jp/test`、JSUT Basic5000、Common Voice Japanese 27.0 `test`、TEDxJP-10K v1.1
- 全音声を FFmpeg で mono / 16 kHz / PCM S16LE に変換し、変換済み音声を全モデルで共有
- 初期プロンプト、hotword、外部言語モデル、VAD は使わない
- 1 音声ずつ推論。NeMo はモデル既定のgreedyデコード・FP32、faster-whisper は CUDA FP16・beam size 5
- Parakeet TDT-CTC 0.6B は TDT と CTC の両デコーダーを別々に測定
- 主指標は Unicode NFKC 後に空白と句読点を除いた CER。漢数字とアラビア数字は変換しない
- 速度はモデル読み込み時間と音声ごとの推論時間を分けて保存。RTF は推論秒数 / 音声秒数

各データセットとモデルの Hugging Face revision、split、評価設定は [`benchmarks/config.yaml`](benchmarks/config.yaml) に固定しています。音声、予測文、測定結果はローカルの `data/` と `results/` に保存し、Git には追加しません。

## 環境

```sh
nix develop
```

Nix 開発シェルに入った後、そこで環境を作ります。

```sh
uv sync
```

NeMo 2.0.0rc0 と Python 3.10 を使うため、`uv` が Python を管理します。CUDA 12系PyTorchを使い、CUDAが使えるNVIDIA GPUではNeMoをFP32、faster-whisperをFP16で実行します。CPUだけの環境では `--device cpu` を指定してください。Nix開発シェルはfaster-whisper用CUDAライブラリの検索パスも設定します。

## データの準備

以下のコマンドもNix開発シェル内で実行します。

FLEURS と JSUT は Hugging Face Hub から固定 revision を取得します。

```sh
uv run asrbench-ja prepare fleurs
uv run asrbench-ja prepare jsut_basic5000
```

Common Voice は [Mozilla Data Collective](https://mozilladatacollective.com/datasets/cmu5xbhnf00fmnq07z9p3aa70) の Japanese 27.0 を取得し、`test.tsv` と `clips/` を次に配置してください。

```text
data/raw/common-voice/cv-corpus-27.0-2026-09-11/ja/test.tsv
data/raw/common-voice/cv-corpus-27.0-2026-09-11/ja/clips/...
```

```sh
uv run asrbench-ja prepare common_voice_ja
```

TEDxJP-10Kは公式URLリストの動画からv1.1を再構築します。公式手順では約44GBの素材が必要です。次は公式ソースを固定revisionで取得し、動画音声と日本語字幕を保存する例です。

```sh
git clone https://github.com/laboroai/TEDxJP-10K.git data/raw/TEDxJP-10K-source
git -C data/raw/TEDxJP-10K-source checkout 6147e131b165219947861da957789097a713407c
mkdir -p data/raw/TEDxJP-10K-source/temp/raw
while IFS= read -r url; do
  yt-dlp --extract-audio --audio-format wav --write-subs --sub-format vtt --sub-langs ja \
    --output 'data/raw/TEDxJP-10K-source/temp/raw/%(id)s.%(ext)s' "$url"
  sleep 10
done < data/raw/TEDxJP-10K-source/data/tedx-jp_urls.txt
```

公式スクリプトでv1.1を再構築してから準備します。

```sh
uv run python data/raw/TEDxJP-10K-source/compose_tedxjp10k.py \
  data/raw/TEDxJP-10K-source/temp/raw \
  --dst_dir data/raw/TEDxJP-10K_v1.1 \
  --diff_data_dir data/raw/TEDxJP-10K-source/data/1.1
uv run asrbench-ja prepare tedxjp10k
```

YouTube字幕の更新により、公式の約10,000件より少なく再構築される場合があります。音源や推論結果は公開リポジトリへ入れません。

音源は各データ提供元の条件に従って扱ってください。Common Voice は再ホスト禁止のため、データも予測文もこの公開リポジトリには含めません。

## 実行

```sh
uv run asrbench-ja run --dataset fleurs --model all
uv run asrbench-ja run --dataset jsut_basic5000 --model all
uv run asrbench-ja run --dataset common_voice_ja --model all
uv run asrbench-ja run --dataset tedxjp10k --model all
uv run asrbench-ja score
```

全モデルを順番に実行し、途中で止めても既存の予測 JSONL から再開します。`--limit 20` は動作確認用で、正式な比較値には使いません。
