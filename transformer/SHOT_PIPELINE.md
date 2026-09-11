# SHOT の生成・学習ループ

`search_config.py` の `SEARCH_MODE` で探索方式を選択します。
`"shot"` と `"shot_origin"` のどちらでも、以下の共通コマンドでループを実行できます。

リポジトリ直下から実行します。以下はチャンク0～3を各局面のデータ生成に使用する例です。

```powershell
python transformer/shot_pipeline.py --chunk_start 0 --chunk_end 3
```

`end=9, shot=15` から `shot` を1ずつ減らし、`shot=0` の学習後に
`end` を1減らして `shot=15` に戻ります。最後は `end=0, shot=0` です。
各局面について、データ生成が終了してから学習を行い、モデル保存後に次へ進みます。

## 局面ごとのモデル

- `end=9, shot=15` のモデルは、`end=9, shot=14` のデータ生成で1投先の評価に使用します。
- `end=9, shot=14` のモデルは、`end=9, shot=13` のデータ生成で1投先の評価に使用します。
- 各 end の `shot=15` は、既存の実得点・勝率テーブルで評価します。次 end の `shot=0` モデルを参照しません。
- `shot` は CNN の policy 上位256候補から探索します。
- `shot_origin` は policy で候補を絞らず、全3584行動から探索します。
- 学習は既存の `transformer.learn.train()` を使い、局面ごとに新規初期化したモデルを学習します。

## 出力

| 内容 | 保存先の例 |
| --- | --- |
| 学習データ | `data/end9/shot15/sl_data_chunk0_0.npz` |
| モデル | `model/shot-end9-shot15.bin` |
| 学習履歴 | `record/shot-end9-shot15.json` |

`shot_origin` のモデル・履歴には `shot_origin-end9-shot15` のように方式名を付けます。
データはどちらも `data/end{end}/shot{shot}/` に保存します。
ループでは共通の生成・保存処理から選択した探索関数を呼び、同じ学習関数に渡します。
単体の `shot_origin_generator.py` にある raw 保存・pack 処理は、ループでは使用しません。

対象局面のデータやモデルが既にある場合は、生成前に停止します。
`--program-dir` で `data/`・`model/`・`record/` の出力ルートを変更できます。
処理に失敗した場合はそこで停止し、生成済みのファイルは残します。

## 設定

| オプション | 既定値 |
| --- | --- |
| `--chunk_start`, `--chunk_end` | 必須。終了チャンクを含む |
| `--num_workers` | 生成4並列 |
| `--inference_batch_size` | 64。SHOT のバッチ推論に使用。shot_origin は既存の逐次推論 |
| `--simulation_seed` | 0 |
| `--batch-size` | `learning_param.BATCH_SIZE`（現在1024） |
| `--epochs` | `learning_param.EPOCHS`（現在30） |
| `--log-path` | リポジトリの `LearnLog/all` |
| `--base-model` | 中央設定の方式に対応する既存モデル。shot は CNN、shot_origin は vy56 Transformer |
| `--program-dir` | リポジトリ直下 |
| `--start-end`, `--start-shot` | 9、15 |
| `--stop-end`, `--stop-shot` | 0、0 |

生成・学習とも GPU を使用します。CUDA が利用できない場合は停止します。
チャンクサイズと生成の端数処理は、既存の `learning_param.BATCH_SIZE` を使用します。
`--batch-size` は学習時のバッチサイズを指定します。

生成設定は現在の `shot_generator.py` の通常実行に合わせています。
得点差拡張は **OFF**、end 拡張は ON、入力の shuffle seed は12345です。
探索回数は中央設定に従い、`shot` は全 shot で1022回、`shot_origin` は全 shot で14320回です。
枝刈り、教師分布の作り方、value の得点分布予測、optimizer・学習率・損失関数も既存処理を使用します。

範囲を限定する例：

```powershell
python transformer/shot_pipeline.py --chunk_start 0 --chunk_end 3 --start-end 8 --start-shot 15 --stop-end 8 --stop-shot 14
```

途中の shot から開始する場合は、同じ end の次 shot のモデルが上記の保存先に必要です。
各 end の shot 15 から開始する場合、他の end の学習済みモデルは不要です。
