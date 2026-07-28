# video-clip-poc

人物検出・自動リフレーム（横動画 → 9:16縦動画）の PoC。

MediaPipe Face Detection で動画中の顔位置を検出し、移動平均で座標を平滑化した上で
ffmpeg の `crop` フィルタにより 9:16 の縦型動画を生成します。

## 構成

```
extract_frames.py    入力動画から0.5秒間隔でフレームを抽出し frames/ に保存
detect_face.py        各フレームの顔バウンディングボックスを検出し coordinates.json に出力
smooth_and_crop.py    座標を平滑化し、9:16クロップ窓を計算して output_vertical.mp4 を生成
measure.py             上記一連の処理を実行し、処理時間（元動画1分あたりの秒数）を計測
```

## セットアップ

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

MediaPipe の Face Detector タスク用モデルファイル（`models/blaze_face_full_range.tflite`）が必要です。
未取得の場合は以下から取得してください。

```powershell
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_full_range/float16/1/blaze_face_full_range.tflite" -OutFile "models\blaze_face_full_range.tflite"
```

`blaze_face_short_range`（近距離・正面向き向け）と比較検証した結果、話者が横向き・奥にいる場面が多い
ウェビナー録画では `blaze_face_full_range` の方が検出率が大きく高かったため、こちらを採用しています。

ffmpeg / ffprobe が PATH 上で実行できる必要があります。

## サンプル動画

`samples/sample.mp4` はダミーの検証用動画です（1280x720, 6秒, 30fps, 単色背景+テキスト）。
以下のコマンドで再生成できます。

```powershell
ffmpeg -y -f lavfi -i "color=c=navy:s=1280x720:d=6:r=30" `
  -vf "drawtext=text='PoC Sample Video':fontfile='C\:/Windows/Fonts/arial.ttf':fontcolor=white:fontsize=60:x=(w-text_w)/2:y=(h-text_h)/2,drawbox=x=550:y=200:w=180:h=180:color=white@0.8:t=fill" `
  -c:v libx264 -pix_fmt yuv420p samples\sample.mp4
```

実際の顔が映っていないダミー動画のため、`detect_face.py` は全フレームで検出失敗となり、
`smooth_and_crop.py` のフォールバック処理（3フレーム以上連続失敗時に直前座標を維持）を
確認する用途に使えます。実際の人物が映った動画を `samples/` に配置して同様に実行することで、
本来の顔追従クロップの動作を確認できます。

## 実行方法

個別実行:

```powershell
.\venv\Scripts\python.exe extract_frames.py samples\sample.mp4
.\venv\Scripts\python.exe detect_face.py
.\venv\Scripts\python.exe smooth_and_crop.py samples\sample.mp4
```

一括実行＋処理時間計測:

```powershell
.\venv\Scripts\python.exe measure.py samples\sample.mp4
```

## 検証結果（ダミー動画 samples/sample.mp4 に対する実行結果）

コマンド: `.\venv\Scripts\python.exe measure.py samples\sample.mp4`

```
[extract_frames] fps=30.00 step=15 frames_saved=12 -> frames
[detect_face] frames=12 detected=0 -> coordinates.json
[smooth_and_crop] frames=12 fallback_frames(streak>=3)=10
[smooth_and_crop] source=1280x720 crop=404x720 samples=12
[smooth_and_crop] wrote output_vertical.mp4
[measure] --- stage timing (sec) ---
[measure] extract_frames : 0.336
[measure] detect_face    : 4.874
[measure] smooth_and_crop: 0.221
[measure] total          : 5.432
[measure] source video duration: 6.000 sec
[measure] processing time per 1 min of source video: 54.317 sec/min
```

- エラー: なし
- 出力ファイル: `output_vertical.mp4`（404x720, 9:16比率, 6秒, 30fps）— 生成を確認済み
- 抽出フレーム数: 12（frames/ フォルダ内）
- 顔検出: ダミー動画のため0件（想定通り）。3フレーム以上連続で検出失敗した12フレーム中10フレームで
  フォールバック（直前座標維持）が発動したことを確認
- 処理時間: 元動画1分あたり約54.3秒（`detect_face.py` が全体の約9割を占める。実顔映像でも
  フレーム数・解像度が同程度であれば同様のオーダーになる見込み）

## 注意事項

- 本PoCは検証目的のみであり、DB接続や本番環境への影響はありません。
- `venv/`, `frames/`, `*.mp4`, `__pycache__/` は `.gitignore` で除外しています
  （`output_vertical.mp4` や `samples/sample.mp4` もリポジトリには含まれません。再生成してください）。
