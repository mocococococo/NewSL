# Transformer Feature Specification

この文書は `transformer/feature.py` の入力特徴量仕様をまとめる。

## 目的

NewSL の CNN 版 `nn/feature.py::generate_input_planes()` と同じ raw state を受け取り、Transformer 版 `network.py` に渡すための単一局面特徴を作る。

## 関数

```python
generate_input_features(
    stones,
    end,
    shot,
    hammer,
    score_diff_for_team0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]
```

## 入力

- `stones`: 長さ16の `list[dict | None]`
- `stones[i]` が `None` なら、その石は盤上に存在しない
- `stones[i]` が `dict` なら、CNN版と同じく `stone["position"]["x"]`, `stone["position"]["y"]` を持つ
- `i < 8` は team0、`i >= 8` は team1 の石
- `end`: 0-origin
- `shot`: 0-origin
- `hammer`: そのエンドの後攻チーム。`0` または `1`
- `score_diff_for_team0`: team0 から見たスコア差

## 出力

単一局面なので batch 次元は持たない。

```text
stones_feature: (16, 5), float32
game_feature:   (4,), float32
stone_mask:     (16,), bool
```

`stone_mask` は PyTorch の key padding mask に合わせる。

```text
False = 有効な石
True  = padding
```

## Stone Feature

存在する石だけを tee に近い順で前に詰め、残りは zero padding にする。

```text
[
    x_feature,
    y_feature,
    is_own_stone,
    dist_feature,
    in_house,
]
```

座標は CNN 版と同じ盤面範囲に clamp してから特徴化する。

```text
x = clamp(x, X_MIN, X_MAX)
y = clamp(y, Y_MIN, Y_MAX)
scale = Y_MAX - Y_MIN

x_feature = x / scale
y_feature = (y - Y_MIN) / scale
dist_feature = dist_to_tee / scale
```

`dist_to_tee` は次で計算する。

```text
sqrt(x^2 + (y - Y_TEE)^2)
```

`in_house` は CNN 版と同じく、石半径を含める。

```text
dist_to_tee <= R_HOUSE + STONE_RADIUS
```

`is_own_stone` は現在投げるチームから見て自分の石なら `1.0`、相手の石なら `0.0`。

## Game Feature

```text
[
    end_norm,
    shot_norm,
    has_hammer,
    score_diff_norm,
]
```

`shot_team` は CNN 版と同じ計算にする。

```text
shot_team = hammer if (shot % 2) == 1 else 1 - hammer
```

`has_hammer` は、現在投げるチームが後攻なら `1.0`、先攻なら `0.0`。

```text
has_hammer = 1.0 if shot_team == hammer else 0.0
```

スコア差は現在投げるチーム視点に変換する。

```text
score_diff_for_current_team =
    score_diff_for_team0      if shot_team == 0
    -score_diff_for_team0     if shot_team == 1
```

正規化は次の通り。

```text
end_norm = clamp(end, 0, 9) / 9
shot_norm = clamp(shot, 0, 15) / 15
score_diff_norm = clamp(score_diff_for_current_team, -8, 8) / 8
```

## 不正入力

- `len(stones) != 16` は `ValueError`
- `hammer` が `0` または `1` でない場合は `ValueError`
- `stone` が `None` でなく、`position.x` / `position.y` を持たない場合は例外
