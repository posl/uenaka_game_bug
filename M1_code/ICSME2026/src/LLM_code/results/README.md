# Results

このディレクトリには `6_run_vision_prompts.py` の実行結果が格納される。
各ファイルは JSONL 形式（1行1JSONオブジェクト）で、シナリオごとに分かれている。

## ファイル一覧

| ファイル名 | シナリオ | 説明 |
|---|---|---|
| `all_vision_results.jsonl` | all | 全カテゴリのバグ検出（二値分類） |
| `oob_vision_results.jsonl` | oob | Out of Bounds バグ検出 |
| `fg_vision_results.jsonl` | fg | Floating Glitch バグ検出 |

## スキーマ

各行は以下のフィールドを持つ JSON オブジェクトである。

| フィールド | 型 | 説明 |
|---|---|---|
| `frame` | string | フレームファイル名（例: `"267613.jpg"`） |
| `label` | int | 正解ラベル（`1` = バグ, `0` = 非バグ） |
| `scenario` | string | シナリオ名（`"all"`, `"oob"`, `"fg"`） |
| `prediction` | int | モデルの予測（`1` = バグ, `0` = 非バグ） |
| `reason` | string | モデルによる判断理由の説明文 |
| `response_id` | string | OpenAI Responses API のレスポンスID |
| `model` | string | 使用モデル（例: `"gpt-4.1-mini-2025-04-14"`） |
| `usage` | object | トークン使用量の内訳 |
| `usage.input_tokens` | int | 入力トークン数 |
| `usage.output_tokens` | int | 出力トークン数 |
| `usage.total_tokens` | int | 合計トークン数（入力 + 出力） |
| `cost` | float | このリクエストの推定コスト（USD） |

## 出力例

```json
{
  "frame": "267613.jpg",
  "label": 1,
  "scenario": "oob",
  "prediction": 1,
  "reason": "The character appears to be inside a solid wall or terrain, which is not a normal position and indicates passing through collision boundaries, a typical Out of Bounds bug.",
  "response_id": "resp_0652ef99c812739300699f14ccb6708195acdfab4622ffca82",
  "model": "gpt-4.1-mini-2025-04-14",
  "usage": {
    "input_tokens": 8607,
    "output_tokens": 44,
    "total_tokens": 8651
  },
  "cost": 0.0035132
}
```

## フレームIDと予測結果の紐づけ

各行の `frame` フィールドがフレームの一意な識別子となる。
`frame` をキーにして、正解ラベル (`label`) とモデル予測 (`prediction`) を比較すればよい。

例えば、pandas で読み込む場合：

```python
import pandas as pd

df = pd.read_json("results/oob_vision_results.jsonl", lines=True)
# frame ごとの正解と予測を比較
print(df[["frame", "label", "prediction", "reason"]].head())
```

## 備考

- モデルの返答が有効な JSON でなかった場合、`prediction` には生のテキストが入り、`reason` は空文字列になる。
- コストはデフォルトで standard 料金で計算される。`6_run_vision_prompts.py` 実行時に `--pricing-tier batch` を指定すると batch 料金で計算される。
- スクリプトはレジューム対応：同じシナリオで再実行すると、処理済みフレームはスキップされる。
