# ttsbench-ja

再現可能な日本語 ASR ベンチマークです。FLEURS、JSUT Basic5000、TEDxJP-10K v1.1 で Parakeet 0.6B と faster-whisper `large-v3-turbo` を比較します。Parakeet 1.1B CTC も測定しましたが、今回の3データセットで0.6Bに精度・速度とも及ばなかったため、現在の実行対象から外しています。

## 比較条件

- 計測データセット: FLEURS `ja_jp/test`、JSUT Basic5000、TEDxJP-10K v1.1
- 全音声を FFmpeg で mono / 16 kHz / PCM S16LE に変換し、変換済み音声を全モデルで共有
- 初期プロンプト、hotword、外部言語モデル、VAD は使わない
- 1 音声ずつ推論。NeMo はモデル既定のgreedyデコード・FP32、faster-whisper は CUDA FP16・beam size 5
- Parakeet TDT-CTC 0.6B は TDT と CTC の両デコーダーを別々に測定
- 主指標は Unicode NFKC 後に空白と句読点を除いた CER。漢数字とアラビア数字は変換しない
- 速度はモデル読み込み時間と音声ごとの推論時間を分けて保存。RTF は推論秒数 / 音声秒数

各データセットと現行モデルの Hugging Face revision、split、評価設定は [`benchmarks/config.yaml`](benchmarks/config.yaml) に固定しています。音声と予測文はローカルの `data/` と `results/` に保存し、Git には追加しません。今回の測定条件と集計値は下記に記録しています。

## ベンチマーク結果

CERは低いほど認識精度が高く、RTFは低いほど高速です。p50は音声1件あたりの推論時間中央値、Peak VRAMは推論時のGPUメモリ最大使用量です。測定環境はNVIDIA RTX 3060 12GB、CUDAです。

| データセット | 件数 | モデル | CER (%) | RTF | p50 (秒) | Peak VRAM (MB) |
|---|---:|---|---:|---:|---:|---:|
| FLEURS | 650 | faster-whisper large-v3-turbo | **4.995** | 0.0379 | 0.489 | 6,952 |
| FLEURS | 650 | Parakeet 1.1B CTC（測定済・現行対象外） | 7.791 | 0.0080 | 0.103 | 4,762 |
| FLEURS | 650 | Parakeet 0.6B CTC | 5.678 | **0.0044** | 0.057 | 3,068 |
| FLEURS | 650 | Parakeet 0.6B TDT | 5.736 | 0.0053 | 0.067 | 3,068 |
| JSUT Basic5000 | 5,000 | faster-whisper large-v3-turbo | 7.267 | 0.0907 | 0.432 | 6,622 |
| JSUT Basic5000 | 5,000 | Parakeet 1.1B CTC（測定済・現行対象外） | 9.047 | 0.0138 | 0.062 | 4,432 |
| JSUT Basic5000 | 5,000 | Parakeet 0.6B CTC | 6.701 | **0.0075** | **0.033** | 2,746 |
| JSUT Basic5000 | 5,000 | Parakeet 0.6B TDT | **6.629** | 0.0087 | 0.038 | **2,646** |
| TEDxJP-10K v1.1 | 8,949 | faster-whisper large-v3-turbo | 10.648 | 0.1297 | 0.413 | 6,592 |
| TEDxJP-10K v1.1 | 8,949 | Parakeet 1.1B CTC（測定済・現行対象外） | 16.128 | 0.0183 | 0.060 | 4,402 |
| TEDxJP-10K v1.1 | 8,949 | Parakeet 0.6B CTC | 9.638 | **0.0098** | **0.032** | 2,734 |
| TEDxJP-10K v1.1 | 8,949 | Parakeet 0.6B TDT | **9.590** | 0.0110 | 0.036 | **2,734** |

FLEURSではfaster-whisper、JSUTとTEDxではParakeet 0.6B TDTが最低CERでした。Parakeet 0.6BのTDTとCTCの精度差は最大でも0.072ポイントで、CTCは3データセットすべてでRTFが低くなりました。1.1B CTCは3データセットすべてで0.6BよりCERが高く、RTFも大きかったため、現行の `--model all` から外しています。

TEDxJP-10KはYouTubeの273 URL中244本を取得し、公式手順で8,949発話（約7.94時間）に再構築しました。29本の取得失敗と字幕差分により、公式の約1万件より少なくなっています。Common Voice Japanese 27.0はMozilla Data Collectiveの認証が必要で、この環境では未取得・未計測です。

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
