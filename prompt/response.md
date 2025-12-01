from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple


# ==========
# 型・ユーティリティ
# ==========

@dataclass
class Shot:
    """
    ショットの物理パラメータを表すデータ構造。

    Attributes:
        vx (float): x方向初速ベクトル。
        vy (float): y方向初速ベクトル。
        spin (int): 回転方向。0=cw, 1=ccw を想定。
    """
    vx: float
    vy: float
    spin: int


@dataclass
class State:
    """
    局面（ノードが保持するゲーム状態）を表すデータ構造。

    Attributes:
        stones (List[Tuple[float, float]]): シート上の全ストーン座標（単位・座標系は実装側に依存）。
        hammer (int): 先攻/後攻を表すフラグ。0=先攻, 1=後攻。
        throw_index (int): このエンド内の投番（0〜15 を想定）。
        end_index (int): 現在のエンド番号（0始まりを想定）。
        cum_score_diff (int): これまでのエンドで生まれた得点差（to-play視点: プラス=有利）。
        to_play (int): この局面で投げるチーム（0/1）。手番交代に用いる。
    """
    stones: List[Tuple[float, float]]
    hammer: int
    throw_index: int
    end_index: int
    cum_score_diff: int
    to_play: int  # 0 or 1


# ==========
# Action 空間（32×32×2 の離散化）
# ==========

class ActionSpace:
    """
    32×32×2 に離散化されたショット空間の変換ユーティリティ。

    - vx ∈ [-0.25, +0.25] を 32 等分
    - vy ∈ [ +2.2,  +3.5] を 32 等分
    - spin ∈ {0(cw), 1(ccw)}
    """

    VX_MIN = -0.25
    VX_MAX = +0.25
    VY_MIN = +2.2
    VY_MAX = +3.5
    VX_BINS = 32
    VY_BINS = 32
    SPIN_BINS = 2
    ACTION_SIZE = VX_BINS * VY_BINS * SPIN_BINS

    @classmethod
    def _linspace(cls, lo: float, hi: float, bins: int) -> List[float]:
        """閉区間 [lo, hi] を bins 等分した中心値リストを返す。"""
        if bins == 1:
            return [(lo + hi) / 2.0]
        step = (hi - lo) / (bins - 1)
        return [lo + i * step for i in range(bins)]

    @classmethod
    def decode(cls, action_id: int) -> Shot:
        """
        action_id から (vx, vy, spin) を復元する。

        Args:
            action_id (int): 0〜2047 のアクションID。

        Returns:
            Shot: 復元されたショット。
        """
        assert 0 <= action_id < cls.ACTION_SIZE
        spin = action_id % 2
        rem = action_id // 2
        iy = rem % cls.VY_BINS
        ix = rem // cls.VY_BINS

        vx_vals = cls._linspace(cls.VX_MIN, cls.VX_MAX, cls.VX_BINS)
        vy_vals = cls._linspace(cls.VY_MIN, cls.VY_MAX, cls.VY_BINS)
        return Shot(vx=vx_vals[ix], vy=vy_vals[iy], spin=spin)

    @classmethod
    def encode(cls, vx: float, vy: float, spin: int) -> int:
        """
        連続値 (vx, vy, spin) を最も近い離散インデックスに丸めて action_id を返す。

        Args:
            vx (float): x方向初速。
            vy (float): y方向初速。
            spin (int): 0 または 1。

        Returns:
            int: 0〜2047 のアクションID。
        """
        def nearest_idx(vals: List[float], x: float) -> int:
            best = 0
            best_diff = float("inf")
            for i, v in enumerate(vals):
                d = abs(v - x)
                if d < best_diff:
                    best = i
                    best_diff = d
            return best

        vx_vals = cls._linspace(cls.VX_MIN, cls.VX_MAX, cls.VX_BINS)
        vy_vals = cls._linspace(cls.VY_MIN, cls.VY_MAX, cls.VY_BINS)

        ix = nearest_idx(vx_vals, vx)
        iy = nearest_idx(vy_vals, vy)
        assert spin in (0, 1)
        return ((ix * cls.VY_BINS) + iy) * 2 + spin


# ==========
# NN/Simulator/Rules のインタフェース（実装者置き換え前提）
# ==========

class PolicyNetwork:
    """
    ポリシーネットワークのインタフェース。

    備考:
        2048 次元の確率分布（softmax後）を返すこと。
        非法手の再正規化は MCTS 内で行う前提とする（ここでは未調整の分布を返す）。
    """
    def evaluate(self, state: State) -> List[float]:
        """
        局面を入力として、長さ 2048 の確率分布を返す。

        Args:
            state (State): 評価したい局面。

        Returns:
            List[float]: 長さ 2048 の確率分布。合計はおおむね 1 を想定。
        """
        # ---- 実装者が差し替える想定 ----
        # とりあえず一様分布（デモ用）
        p = 1.0 / ActionSpace.ACTION_SIZE
        return [p] * ActionSpace.ACTION_SIZE


class Simulator:
    """
    1手のショットを適用する高速シミュレータのインタフェース。

    仕様:
        - ルール違反（フリーガードゾーンなど）は simulator 側で判定することを推奨。
        - もし不可能な場合は「ショット前後の状態一致＋ε内判定」を別途実装する。
    """
    def step(self, state: State, shot: Shot) -> Tuple[State, bool]:
        """
        1手だけ状態遷移を行う。

        Args:
            state (State): 現在の局面。
            shot  (Shot): 実行するショット。

        Returns:
            Tuple[State, bool]: (遷移後の局面, 合法フラグ) を返す。
        """
        # ---- 実装者が差し替える想定 ----
        # ダミー：投番+1 と手番トグルだけを進める（物理遷移は未実装）
        next_state = State(
            stones=list(state.stones),
            hammer=state.hammer,
            throw_index=state.throw_index + 1,
            end_index=state.end_index,
            cum_score_diff=state.cum_score_diff,
            to_play=1 - state.to_play,
        )
        legal = True
        return next_state, legal


class EndRules:
    """
    エンド終了判定とスコア計算のインタフェース。
    """
    def is_end_and_score(self, state: State) -> Tuple[bool, int]:
        """
        現局面が終端かどうかと、終端ならエンドスコア（to-play視点でなく客観値）を返す。

        Args:
            state (State): 評価対象の局面。

        Returns:
            Tuple[bool, int]: (終端か, スコア[-8..+8])。
        """
        # ---- 実装者が差し替える想定 ----
        # ダミー：投番==15 で終端、スコア0固定
        terminal = (state.throw_index >= 15)
        score = 0
        return terminal, score


# ==========
# ノード／エッジ
# ==========

@dataclass
class Edge:
    """
    親ノードから子ノードへの遷移（ショット）を表す。

    Attributes:
        action_id (int): 0〜2047 のアクションID。
        prior (float): NNが出力した事前確率（非法手除外・再正規化後）。
        shot (Shot): 実行するショット。
        child (Optional[Node]): 遷移先ノード。未展開なら None。
        N (int): 訪問回数。
        W (float): 累積価値。
        Q (float): 平均価値 (= W / N)。
        illegal (bool): 非法手フラグ。True の場合、候補から除外。
    """
    action_id: int
    prior: float
    shot: Shot
    child: Optional["Node"] = None
    N: int = 0
    W: float = 0.0
    Q: float = 0.0
    illegal: bool = False

    def update_stats(self, v: float) -> None:
        """
        バックアップ段階でこのエッジの統計量を更新する。

        Args:
            v (float): このエッジ視点の価値（[-1, 1]）。
        """
        self.N += 1
        self.W += v
        self.Q = self.W / self.N


@dataclass
class Node:
    """
    MCTS のノード（局面）を表す。

    Attributes:
        state (State): 局面データ。
        P (List[float]): 長さ 2048 の NN 出力確率（非法手除外・再正規化“後”を想定）。
        policy_order (List[int]): P 降順の action_id リスト（非法手は含めない）。
        children (Dict[int, Edge]): アクションID -> Edge。
        unlocked_actions (List[int]): progressive widening で解放済みのアクションID。
        N (int): ノード訪問回数。
    """
    state: State
    P: List[float] = field(default_factory=list)
    policy_order: List[int] = field(default_factory=list)
    children: Dict[int, Edge] = field(default_factory=dict)
    unlocked_actions: List[int] = field(default_factory=list)
    N: int = 0

    def ensure_child(self, action_id: int) -> Edge:
        """
        指定アクションの Edge を存在させて返す（なければ生成）。

        Args:
            action_id (int): アクションID。

        Returns:
            Edge: 対応するエッジ。
        """
        if action_id not in self.children:
            shot = ActionSpace.decode(action_id)
            prior = self.P[action_id] if self.P else 0.0
            self.children[action_id] = Edge(action_id=action_id, prior=prior, shot=shot)
        return self.children[action_id]


# ==========
# MCTS/PUCT
# ==========

@dataclass
class PUCTConfig:
    """
    PUCT と Progressive Widening の設定。

    Attributes:
        cpuct (float): 探索バイアスの強さ。
        widen_c0 (int): 最低解放数 c0（M(N)=max(c0, floor(k*N^a))）。
        widen_k (float): k。
        widen_a (float): a。
        root_decision (str): ルートの手の確定規則。"N" or "Q"。
    """
    cpuct: float = 1.5
    widen_c0: int = 4
    widen_k: float = 1.0
    widen_a: float = 0.5
    root_decision: str = "N"  # or "Q"


class PUCTSearch:
    """
    Valueヘッド無しのポリシー単独PUCTを用いた1エンド限定の探索器。

    使い方:
        search = PUCTSearch(nn, sim, rules, cfg)
        best_action, root = search.search(root_state, n_simulations=1000)
    """

    def __init__(self, nn: PolicyNetwork, sim: Simulator, rules: EndRules, config: PUCTConfig):
        """
        構築子。

        Args:
            nn (PolicyNetwork): ポリシーネットワーク。
            sim (Simulator): 高速シミュレータ。
            rules (EndRules): 終端判定・スコア計算。
            config (PUCTConfig): パラメータ。
        """
        self.nn = nn
        self.sim = sim
        self.rules = rules
        self.cfg = config

    # ---- 公開API ---------------------------------------------------------

    def search(self, root_state: State, n_simulations: int) -> Tuple[int, Node]:
        """
        ルート状態から n_simulations 回のシミュレーションを実行し、最善手とルートノードを返す。

        Args:
            root_state (State): ルート局面。
            n_simulations (int): シミュレーション回数。

        Returns:
            Tuple[int, Node]: (選択アクションID, ルートノード)。
        """
        root = Node(state=root_state)
        self._initialize_node(root)

        for _ in range(n_simulations):
            self._simulate(root)

        best = self._select_root_move(root)
        return best, root

    # ---- 内部ロジック -----------------------------------------------------

    def _initialize_node(self, node: Node) -> None:
        """
        ノード初期化。NNでPを得て並びを決め、初期解放を行う。

        Args:
            node (Node): 初期化対象ノード。
        """
        # 端末かチェック
        terminal, score = self.rules.is_end_and_score(node.state)
        if terminal:
            node.P = [0.0] * ActionSpace.ACTION_SIZE
            node.policy_order = []
            node.unlocked_actions = []
            return

        raw_P = self.nn.evaluate(node.state)  # 長さ2048
        # ここでは非法手の情報が無いので、いったんそのまま保持（展開時に除外＆再正規化）
        # 初期の policy_order は NNの降順（暫定）
        order = list(range(ActionSpace.ACTION_SIZE))
        order.sort(key=lambda a: raw_P[a], reverse=True)

        node.P = list(raw_P)
        node.policy_order = order
        node.unlocked_actions = []  # progressive widening で段階解放

        # 初回の解放
        self._progressive_widen(node)

    def _simulate(self, root: Node) -> None:
        """
        1回の MCTS シミュレーション（Select→Expand→Backup）を実行する。

        Args:
            root (Node): ルートノード。
        """
        path: List[Tuple[Node, Edge]] = []
        node = root

        # 選択フェーズ
        while True:
            terminal, score = self.rules.is_end_and_score(node.state)
            if terminal:
                v_leaf = self._normalize_score(score)  # 葉（終端）の客観スコアを[-1,1]に
                break

            # progressive widening で解放数を更新
            self._progressive_widen(node)

            # 候補の中から PUCT で1手選ぶ
            edge = self._select_edge_via_puct(node)
            if edge is None:
                # 解放候補が全て非法で尽きた場合など。保険として終了扱い（価値0）
                v_leaf = 0.0
                break

            # 未展開なら展開
            if edge.child is None:
                next_state, legal = self.sim.step(node.state, edge.shot)
                if not legal:
                    # 非法なら以後除外。親ノードで再選択（探索手数は消費しない扱い）
                    edge.illegal = True
                    # 非法除外後の確率再正規化
                    self._renormalize_P_excluding_illegal(node)
                    continue

                child = Node(state=next_state)
                edge.child = child

                t, s = self.rules.is_end_and_score(child.state)
                if t:
                    v_leaf = self._normalize_score(s)
                    path.append((node, edge))
                    node = child
                    break
                else:
                    # 子ノード初期化（NN評価）
                    self._initialize_node(child)
                    path.append((node, edge))
                    node = child
                    # 続行して次のループで選択へ
            else:
                # 既展開なら次へ降りる
                path.append((node, edge))
                node = edge.child

        # バックアップ
        self._backup(path, node_value=v_leaf)

    def _select_edge_via_puct(self, node: Node) -> Optional[Edge]:
        """
        解放済み（かつ合法）の候補の中から PUCT の (Q+U) 最大のエッジを返す。

        Args:
            node (Node): 対象ノード。

        Returns:
            Optional[Edge]: 選ばれたエッジ。候補が無ければ None。
        """
        candidates = []
        for aid in node.unlocked_actions:
            edge = node.ensure_child(aid)
            if not edge.illegal:
                candidates.append(edge)

        if not candidates:
            return None

        best_score = -float("inf")
        best_edges: List[Edge] = []
        sqrtN = math.sqrt(max(1, node.N))
        for e in candidates:
            U = self.cfg.cpuct * e.prior * sqrtN / (1 + e.N)
            score = e.Q + U
            if score > best_score:
                best_score = score
                best_edges = [e]
            elif score == best_score:
                best_edges.append(e)

        # タイブレーク：prior の大きい方、さらに同値ならランダム
        if len(best_edges) == 1:
            return best_edges[0]
        best_edges.sort(key=lambda x: x.prior, reverse=True)
        top_prior = best_edges[0].prior
        tied = [e for e in best_edges if e.prior == top_prior]
        return random.choice(tied)

    def _progressive_widen(self, node: Node) -> None:
        """
        親ノード訪問回数に応じて解放する候補数 M(N) を増やす。

        Args:
            node (Node): 対象ノード。
        """
        target = max(self.cfg.widen_c0, int(self.cfg.widen_k * (max(1, node.N) ** self.cfg.widen_a)))
        # 既解放数が target 未満なら、policy_order の上位から追加
        i = 0
        unlocked_set = set(node.unlocked_actions)
        while len(unlocked_set) < target and i < len(node.policy_order):
            aid = node.policy_order[i]
            i += 1
            # 既に非法がわかっている場合はスキップ（Edge作成時に分かる）
            if aid in unlocked_set:
                continue
            unlocked_set.add(aid)
        node.unlocked_actions = list(unlocked_set)

    def _renormalize_P_excluding_illegal(self, node: Node) -> None:
        """
        非法手を除外した後で、P を再正規化する（候補の前提確率を整える）。

        Args:
            node (Node): 対象ノード。
        """
        # 非法手を 0 にして合計で割り直す
        total = 0.0
        mask_illegal = set()
        for aid, edge in node.children.items():
            if edge.illegal:
                mask_illegal.add(aid)

        if not node.P:
            return

        for i, p in enumerate(node.P):
            if i in mask_illegal:
                continue
            total += p

        if total <= 0:
            # すべて非法等の異常時は解放済み候補を一様に（保険）
            eq = 1.0 / max(1, len(node.unlocked_actions))
            for aid in range(len(node.P)):
                node.P[aid] = 0.0
            for aid in node.unlocked_actions:
                node.P[aid] = eq
        else:
            for i in range(len(node.P)):
                if i in mask_illegal:
                    node.P[i] = 0.0
                else:
                    node.P[i] /= total

        # policy_order も P の降順で再構築
        order = list(range(ActionSpace.ACTION_SIZE))
        order.sort(key=lambda a: node.P[a], reverse=True)
        node.policy_order = order

        # 既存の Edge に prior を反映
        for aid, edge in node.children.items():
            edge.prior = node.P[aid]

    def _backup(self, path: List[Tuple[Node, Edge]], node_value: float) -> None:
        """
        葉の価値 node_value を、to-play 視点の符号を交替させながら根に向かってバックアップする。

        Args:
            path (List[Tuple[Node, Edge]]): ルートから葉直前までの (Node, Edge) の列。
            node_value (float): 葉（終端）での正規化スコア（[-1,1]、客観値）。
        """
        # 葉直前ノードの to-play から見た値に変換して反映していく
        # ルート→…→葉直前 の順に path が積まれている
        # 深さ d 手戻るごとに符号が反転（to-play が交替）する。
        for depth_from_leaf, (node, edge) in enumerate(reversed(path), start=1):
            v = node_value if (depth_from_leaf % 2 == 1) else -node_value
            # エッジ統計更新
            edge.update_stats(v)
            # ノード訪問回数
            node.N += 1

    def _normalize_score(self, score: int) -> float:
        """
        エンドスコアを [-1, 1] に線形正規化する。

        Args:
            score (int): [-8, +8] のスコア（客観値）。

        Returns:
            float: [-1, 1] の値。
        """
        return max(-1.0, min(1.0, score / 8.0))

    def _select_root_move(self, root: Node) -> int:
        """
        探索終了後のルート手を選択する。

        Args:
            root (Node): ルートノード。

        Returns:
            int: 採用するアクションID。
        """
        edges = [root.ensure_child(aid) for aid in root.children.keys()]
        edges = [e for e in edges if not e.illegal]

        if not edges:
            # 解が無い場合の保険：解放済みから prior 最大
            if root.unlocked_actions:
                best_a = max(root.unlocked_actions, key=lambda aid: root.P[aid] if root.P else 0.0)
                return best_a
            return 0  # どうにもならない場合のフォールバック

        if self.cfg.root_decision.upper() == "Q":
            best = max(edges, key=lambda e: e.Q)
        else:
            best = max(edges, key=lambda e: e.N)
        return best.action_id


# ==========
# 使い方（参考）
# ==========

def example_usage() -> None:
    """
    デモ用の簡易実行例。
    """
    nn = PolicyNetwork()
    sim = Simulator()
    rules = EndRules()
    cfg = PUCTConfig(cpuct=1.5, widen_c0=4, widen_k=1.0, widen_a=0.5, root_decision="N")

    root_state = State(
        stones=[],
        hammer=1,
        throw_index=0,
        end_index=0,
        cum_score_diff=0,
        to_play=0,
    )

    search = PUCTSearch(nn, sim, rules, cfg)
    best_action, root = search.search(root_state, n_simulations=128)

    shot = ActionSpace.decode(best_action)
    print("best_action:", best_action, "=>", shot)


# ==========
# 不足情報のプレースホルダ（この仕様で必要なAPIの“名前だけ”を記す）
# ==========

# - is_states_equal_with_epsilon(state_a, state_b, eps)  # 状態一致（浮動小数の許容誤差つき）を判定する
# - compute_end_score(stones, hammer)  # ストーン配置からエンドスコア（客観）を計算する
# - encode_state_for_nn(state)  # NN入力の前処理（座標正規化・特徴量化）を行う
# - PolicyNetwork.evaluate(state)  # NNの推論（2048次元softmax出力）
# - Simulator.step(state, shot)  # 1手の物理シミュレーションと合法判定
# - EndRules.is_end_and_score(state)  # 投番==15時にスコア確定を返す（未満ではFalse）
