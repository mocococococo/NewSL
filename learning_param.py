"""学習用の各種ハイパーパラメータの設定。
"""

# 教師あり学習実行時の学習率
SL_LEARNING_RATE = 0.01

# ミニバッチサイズ
BATCH_SIZE = 256

# 学習器のモーメンタムパラメータ
MOMENTUM=0.9

# L2正則化の重み
WEIGHT_DECAY = 1e-4

EPOCHS = 15

# 学習率を変更するエポック数と変更後の学習率
LEARNING_SCHEDULE = {
    "learning_rate": {
        5: 0.001,
        8: 0.0001,
        10: 0.00001,
    }
}

# npzファイル1つに格納するデータの個数
DATA_SET_SIZE = BATCH_SIZE * 4

# Policyのlossに対するValueのlossの重み比率
"""Valueは使用しない"""
SL_VALUE_WEIGHT = 0
