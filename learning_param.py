"""学習用の各種ハイパーパラメータの設定。
"""

# 教師あり学習実行時の学習率
SL_LEARNING_RATE_SGD = 0.01
SL_LEARNING_RATE_ADAMW = 0.0003

# ミニバッチサイズ
BATCH_SIZE = 1024

# AdamWオプティマイザのパラメータ
OPTIMIZER_NAME = "adamw"
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8

# 学習器のモーメンタムパラメータ
MOMENTUM=0.9

# L2正則化の重み
WEIGHT_DECAY = 1e-4

EPOCHS = 50

# 学習率を変更するエポック数と変更後の学習率
LEARNING_SCHEDULE_SGD = {
    "learning_rate": {
        5: 0.001,
        8: 0.0001,
        10: 0.00001,
    }
}

LEARNING_SCHEDULE_ADAMW = {
    "learning_rate": {
        5: 0.0001,
        8: 0.00003,
        10: 0.00001,
    }
}

SL_LEARNING_RATE = SL_LEARNING_RATE_SGD
LEARNING_SCHEDULE = LEARNING_SCHEDULE_SGD

# npzファイル1つに格納するデータの個数
DATA_SET_SIZE = BATCH_SIZE * 100

# Policyのlossに対するValueのlossの重み比率
"""Valueは使用しない"""
SL_VALUE_WEIGHT = 0.0625
