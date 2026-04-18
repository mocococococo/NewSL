"""カーリング局面用 Transformer Encoder policy-value network。

入力仕様は意図的に厳密にする。

    stones:     (B, max_stones, stone_feat_dim)
    game:       (B, game_feat_dim)
    stone_mask: (B, max_stones)

`stone_mask` は False が有効な石、True が padding の石を表す。
network 内部で game token を末尾に追加し、mask にも game token 用の
False を追加して TransformerEncoder に渡す。
"""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn
import torch.nn.functional as F

from transformer.params import ActivationName, TransformerNetworkConfig


def _make_activation(name: ActivationName) -> nn.Module:
    """設定名から活性化関数 module を作る。"""

    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


class MLP(nn.Module):
    """embedding と head で使う 2 層 MLP。"""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float,
        activation: ActivationName,
        use_layer_norm: bool,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = [
            nn.Linear(in_dim, hidden_dim),
            _make_activation(activation),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, out_dim))

        self.net = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(out_dim) if use_layer_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.net(x))


class TransformerNetwork(nn.Module):
    """盤面全体の policy / value logits を返す Transformer Encoder。"""

    def __init__(
        self,
        config: TransformerNetworkConfig | None = None,
        **overrides: object,
    ) -> None:
        super().__init__()

        # ハイパーパラメータ本体は transformer/params.py 側で管理する。
        if config is None:
            config = TransformerNetworkConfig()
        if overrides:
            config = replace(config, **overrides)
        self.config = config

        if config.d_model % config.nhead != 0:
            raise ValueError(
                f"d_model ({config.d_model}) must be divisible by nhead ({config.nhead})"
            )
        if config.max_stones <= 0:
            raise ValueError("max_stones must be positive")

        embedding_hidden_dim = config.embedding_hidden_dim or config.d_model
        head_hidden_dim = config.head_hidden_dim or config.d_model

        # 石 1 個の特徴量を Transformer 内部次元 d_model に変換する。
        self.stone_embedding = MLP(
            in_dim=config.stone_feat_dim,
            hidden_dim=embedding_hidden_dim,
            out_dim=config.d_model,
            dropout=config.dropout,
            activation=config.activation,
            use_layer_norm=True,
        )

        # 試合情報を game token 用の d_model 次元に変換する。
        self.game_embedding = MLP(
            in_dim=config.game_feat_dim,
            hidden_dim=embedding_hidden_dim,
            out_dim=config.d_model,
            dropout=config.dropout,
            activation=config.activation,
            use_layer_norm=True,
        )

        # token の役割を区別する埋め込み。0: stone token, 1: game token。
        self.type_embedding = (
            nn.Embedding(2, config.d_model) if config.use_type_embedding else None
        )

        # 通常は使わないが、比較実験用に位置埋め込みを有効化できる。
        self.positional_embedding = (
            nn.Parameter(torch.zeros(1, config.max_stones + 1, config.d_model))
            if config.use_positional_embedding
            else None
        )

        # CNN の common block 群に相当する Transformer trunk。
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.num_layers,
            norm=nn.LayerNorm(config.d_model),
        )

        # game token の最終表現から policy logits を作る。
        self.policy_head = MLP(
            in_dim=config.d_model,
            hidden_dim=head_hidden_dim,
            out_dim=config.action_dim,
            dropout=config.dropout,
            activation=config.activation,
            use_layer_norm=False,
        )

        # game token の最終表現から score/value logits を作る。
        self.value_head = MLP(
            in_dim=config.d_model,
            hidden_dim=head_hidden_dim,
            out_dim=config.value_dim,
            dropout=config.dropout,
            activation=config.activation,
            use_layer_norm=False,
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """追加した embedding parameter を初期化する。"""

        if self.type_embedding is not None:
            nn.init.normal_(self.type_embedding.weight, mean=0.0, std=0.02)
        if self.positional_embedding is not None:
            nn.init.trunc_normal_(self.positional_embedding, std=0.02)

    def forward(
        self,
        stones: torch.Tensor,
        game: torch.Tensor,
        stone_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """stones と game から policy/value logits を計算する。"""

        self._validate_stones(stones)
        stones = stones.to(dtype=torch.float32)
        batch_size, stone_count, _ = stones.shape

        game = self._validate_game(game, batch_size, stones.device)

        # (B, max_stones, stone_feat_dim) -> (B, max_stones, d_model)
        stone_tokens = self.stone_embedding(stones)

        # (B, game_feat_dim) -> (B, 1, d_model)
        game_token = self.game_embedding(game).unsqueeze(1)

        # 石 token の末尾に game token を追加する。
        tokens = torch.cat([stone_tokens, game_token], dim=1)

        tokens = self._add_optional_embeddings(tokens, stone_count)
        key_padding_mask = self._build_key_padding_mask(
            stone_mask=stone_mask,
            batch_size=batch_size,
            stone_count=stone_count,
            device=stones.device,
        )

        encoded = self.transformer_encoder(
            tokens,
            src_key_padding_mask=key_padding_mask,
        )

        # 最後の token は game token。これを盤面全体の代表特徴として使う。
        board_feature = encoded[:, -1, :]

        policy_logits = self.policy_head(board_feature)
        value_logits = self.value_head(board_feature)
        return policy_logits, value_logits

    def forward_for_sl(
        self,
        stones: torch.Tensor,
        game: torch.Tensor,
        stone_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """教師あり学習用。softmax 前の logits を返す。"""

        return self.forward(stones, game, stone_mask)

    def forward_with_softmax(
        self,
        stones: torch.Tensor,
        game: torch.Tensor,
        stone_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """policy/value を確率分布として返す。"""

        policy_logits, value_logits = self.forward(stones, game, stone_mask)
        return F.softmax(policy_logits, dim=-1), F.softmax(value_logits, dim=-1)

    @torch.no_grad()
    def inference(
        self,
        stones: torch.Tensor,
        game: torch.Tensor,
        stone_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """推論用。eval mode で softmax 後の分布を返す。"""

        was_training = self.training
        self.eval()
        policy, value = self.forward_with_softmax(stones, game, stone_mask)
        if was_training:
            self.train()
        return policy, value

    def _validate_stones(self, stones: torch.Tensor) -> None:
        """stones が (B, max_stones, stone_feat_dim) であることを検証する。"""

        if not torch.is_tensor(stones):
            raise TypeError(f"stones must be a torch.Tensor, got {type(stones)}")
        if stones.dim() != 3:
            raise ValueError(f"stones must have shape (B, S, F), got {tuple(stones.shape)}")
        if stones.shape[1] != self.config.max_stones:
            raise ValueError(
                f"stone count must be max_stones ({self.config.max_stones}), "
                f"got {stones.shape[1]}"
            )
        if stones.shape[2] != self.config.stone_feat_dim:
            raise ValueError(
                "stone feature dim mismatch: "
                f"{stones.shape[2]} != {self.config.stone_feat_dim}"
            )

    def _validate_game(
        self,
        game: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """game が (B, game_feat_dim) であることを検証して dtype/device を揃える。"""

        if not torch.is_tensor(game):
            raise TypeError(f"game must be a torch.Tensor, got {type(game)}")

        game = game.to(device=device, dtype=torch.float32)
        if game.dim() != 2:
            raise ValueError(f"game must have shape (B, G), got {tuple(game.shape)}")
        if game.shape[0] != batch_size:
            raise ValueError(f"game batch size mismatch: {game.shape[0]} != {batch_size}")
        if game.shape[1] != self.config.game_feat_dim:
            raise ValueError(
                f"game feature dim mismatch: {game.shape[1]} != {self.config.game_feat_dim}"
            )
        return game

    def _add_optional_embeddings(
        self,
        tokens: torch.Tensor,
        stone_count: int,
    ) -> torch.Tensor:
        """type embedding と、必要なら positional embedding を足す。"""

        batch_size, seq_len, _ = tokens.shape

        if self.type_embedding is not None:
            stone_types = torch.zeros(
                (batch_size, stone_count),
                dtype=torch.long,
                device=tokens.device,
            )
            game_type = torch.ones(
                (batch_size, 1),
                dtype=torch.long,
                device=tokens.device,
            )
            type_ids = torch.cat([stone_types, game_type], dim=1)
            tokens = tokens + self.type_embedding(type_ids)

        if self.positional_embedding is not None:
            tokens = tokens + self.positional_embedding[:, :seq_len, :]

        return tokens

    def _build_key_padding_mask(
        self,
        stone_mask: torch.Tensor,
        batch_size: int,
        stone_count: int,
        device: torch.device,
    ) -> torch.Tensor:
        """stone_mask を検証し、game token 用 False を足して key_padding_mask を作る。"""

        if not torch.is_tensor(stone_mask):
            raise TypeError(f"stone_mask must be a torch.Tensor, got {type(stone_mask)}")

        stone_mask = stone_mask.to(device=device, dtype=torch.bool)
        expected_shape = (batch_size, stone_count)
        if tuple(stone_mask.shape) != expected_shape:
            raise ValueError(
                f"stone_mask must have shape {expected_shape}, "
                f"got {tuple(stone_mask.shape)}"
            )

        game_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)
        return torch.cat([stone_mask, game_mask], dim=1)


CurlingTransformer = TransformerNetwork


__all__ = [
    "TransformerNetwork",
    "CurlingTransformer",
]
