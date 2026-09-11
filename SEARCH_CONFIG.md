# 探索設定の切り替え

リポジトリ直下の `search_config.py` の1行を変更します。

```python
SEARCH_MODE: SearchMode = "shot"
```

`"shot_origin"` に変更すると、次の設定一式に切り替わります。

| SEARCH_MODE | 行動数 | 探索回数 | 初期候補数 | 行動の分割構成 |
| --- | ---: | ---: | ---: | --- |
| `shot` | 1600 | 1022 | 256（policy 上位） | default：32×25×2 |
| `shot_origin` | 3584 | 14320 | 3584（全行動） | high_resolution：32×56×2 |

初期値は `shot` です。各方式の設定値は同じファイルの `SEARCH_PROFILES` にまとまっています。
`transformer/params.py` の行動構成、ネットワークの policy 出力数、特徴量・行動変換、
各探索の既定回数・候補数は、この定義を参照します。
`shot/params.py` の shot15 専用名は互換用の別名で、独立した設定ではありません。
shot15 でも回数の上書きはありません。

探索 API に `max_simulations` を明示した場合、その値を使用します。
通常の生成コマンドと学習ループは中央設定の回数を使用します。
教師生成では時間上限を無効にする既存の動作を維持します。

## 単体実行

中央設定に対応する既存スクリプトを実行します。

```powershell
# SEARCH_MODE = "shot"
python transformer/shot_generator.py --chunk_start 0 --chunk_end 3 --num_workers 4

# SEARCH_MODE = "shot_origin"
python transformer/shot_origin_generator.py --start 0 --end 3
```

違う方式のスクリプトや行動構成を指定すると、生成・探索の開始前にエラーになります。
単体実行の保存形式やその他の設定は各既存スクリプトのままです。

## 学習ループ

方式にかかわらず共通のコマンドです。

```powershell
python transformer/shot_pipeline.py --chunk_start 0 --chunk_end 3
```

中央設定で、探索方式・行動数・探索回数・初期候補数と初期モデルを切り替えます。
既定の初期モデルは、shot が従来の CNN、shot_origin が既存の
`transformer-supervised-model-AdamW-vy56.bin` です。`--base-model` で変更できますが、
選択した方式に合う構造のモデルが必要です。

どちらの方式でも、同じ end の次 shot の学習済みモデルを評価に使用します。
各 end の shot15 は既存の実得点・勝率テーブルで評価します。
ループのデータ保存形式、得点差拡張 OFF、枝刈り、学習条件は維持します。
詳細は [学習ループの手順](transformer/SHOT_PIPELINE.md) を参照してください。

設定は起動時に読み込みます。実行を終了してから切り替え、新しいプロセスで起動してください。
行動数の異なるデータやモデルを混ぜないよう、方式を変える際は空の出力先を用意し、
必要なら `--program-dir` で出力ルートを指定してください。
