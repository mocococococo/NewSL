- まず Token から
CNN では、入力はだいたいこうでした。

(B, C, H, W)
たとえば curling の盤面を CNN に入れるなら、32x32 の盤面画像みたいにして、各セルに特徴チャンネルを持たせます。

B = batch size
C = feature plane 数
H = 盤面の高さ
W = 盤面の幅
CNN はこの H x W のマス目を前提にして、「近くのマス同士には意味がある」という仮定で畳み込みます。

一方 Transformer は、入力をこう見ます。

(B, S, D)
ここで重要なのが S です。

B = batch size
S = token の数
D = 各 token の特徴次元
この token は、かなり雑に言うと「Transformer が一つの単位として見るもの」です。

自然言語なら、

"I love curling"
を

["I", "love", "curling"]
のように分けて、各単語を token として扱います。

curling 盤面なら、候補としてはこう考えられます。

stone_0
stone_1
stone_2
...
stone_15
game_info
つまり「石1個」を1 token、「試合全体の情報」をもう1 token として扱う、という発想です。

CNN との対応で言うと、

CNN:
盤面の各マスが処理単位

Transformer:
石やゲーム情報など、意味のあるオブジェクトが処理単位
という違いです。

stone_feat:
石1個の生特徴の次元数
例: x, y, team, dist, in_house なら 5

game_feat:
試合情報の生特徴の次元数
例: score_diff, end, shot なら 3

d_model:
Transformer 内部の特徴次元
CNN の channels に近い
例: 128, 256

num_layers:
TransformerEncoderLayer を何層積むか
CNN の residual block 数に近い
例: 2, 4, 6

nhead:
Attention の見方を何種類に分けるか
CNN の複数 filter/channel に少し近い
例: 4, 8
d_model が nhead で割り切れる必要がある

dim_feedforward:
EncoderLayer 内の MLP の中間次元
CNN block 内の拡張 channel みたいなもの
例: d_model * 4

dropout:
過学習対策

action_dim:
policy の出力次元
速度bin × 回転方向

value_dim:
score 分布なら 17
