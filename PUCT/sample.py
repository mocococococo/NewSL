from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import random


# =========================================================
#  不足情報のため、以下のように仕様のみを明示（実装は未確定）
# =========================================================
# Rules.is_end(next_state: "GameState") -> Tuple[bool, int]  # 終端判定とスコア（スコアは[-8,+8]の整数想定）
# PolicyNetwork.evaluate(input_state: "GameState") -> List[float]  # 長さ2048のポリシー確率分布を返す
# Simulator.step(prev_state: "GameState", shot_info: "Shot") -> Tuple["GameState", bool]  # 1手進めた状態と合法フラグ
# Geometry.equal_sheet(a_sheet: List[Tuple[float,float]], b_sheet: List[Tuple[float,float]], eps: float) -> bool  # 近傍一致
# Encoding.encode_for_cnn(input_state: "GameState") -> "Any"  # NN入力エンコード
# =========================================================


# =========================================================
#  アクション空間（32×32×2 の離散化と相互変換）
# =========================================================

class ActionSpace:
    """アクション（vx, vy, spin）を 32×32×2 に離散化し、IDと相互変換するユーティリティ。
    
    - vx は [-0.25, +0.25] の範囲を32分割
    - vy は [ +2.20, +3.50] の範囲を32分割
    - spin は {0:cw, 1:ccw} の2値
    """

    VX_MIN = -0.25
    VX_MAX = +0.25
    VY_MIN = +2.20
    VY_MAX = +3.50
    VX_BINS = 32
    VY_BINS = 32
    SPIN_BINS = 2
    ACTION_SIZE = VX_BINS * VY_BINS * SPIN_BINS

    @staticmethod
    def index_to_params(index_value: int) -> Tuple[float, float, int, int, int]:
        """アクションIDから (vx, vy, spin, ix, iy) を取得する。
        
        Args:
            index_value: 0..(2047) のアクションID
        
        Returns:
            (vx, vy, spin, ix, iy)
        """
        assert 0 <= index_value < ActionSpace.ACTION_SIZE
        spin_id = index_value % ActionSpace.SPIN_BINS
        temp_q = index_value // ActionSpace.SPIN_BINS
        iy_id = temp_q % ActionSpace.VY_BINS
        ix_id = temp_q // ActionSpace.VY_BINS

        vx = ActionSpace.VX_MIN + (ActionSpace.VX_MAX - ActionSpace.VX_MIN) * (ix_id / (ActionSpace.VX_BINS - 1))
        vy = ActionSpace.VY_MIN + (ActionSpace.VY_MAX - ActionSpace.VY_MIN) * (iy_id / (ActionSpace.VY_BINS - 1))
        return vx, vy, spin_id, ix_id, iy_id

    @staticmethod
    def params_to_index(ix_id: int, iy_id: int, spin_id: int) -> int:
        """離散インデックス (ix, iy, spin) からアクションIDを取得する。
        
        Args:
            ix_id: vxのビン番号（0..31）
            iy_id: vyのビン番号（0..31）
            spin_id: 0(cw) or 1(ccw)
        
        Returns:
            アクションID（0..2047）
        """
        assert 0 <= ix_id < ActionSpace.VX_BINS
        assert 0 <= iy_id < ActionSpace.VY_BINS
        assert 0 <= spin_id < ActionSpace.SPIN_BINS
        return (ix_id * ActionSpace.VY_BINS + iy_id) * ActionSpace.SPIN_BINS + spin_id


# =========================================================
#  ドメインデータ
# =========================================================

@dataclass
class Shot:
    """1回のショットを表すデータ構造。
    
    Attributes:
        shot_vx: x方向初速
        shot_vy: y方向初速
        shot_spin: 0(cw) or 1(ccw)
    """
    shot_vx: float
    shot_vy: float
    shot_spin: int


@dataclass
class GameState:
    """カーリングの局面情報を表すデータ構造（to-play視点で保持）。
    
    Attributes:
        state_sheet: ストーン座標のリスト（[(x,y), ...]）
        state_hammer: 先攻/後攻情報（boolやenumなど、Trueならハンマー所持など任意だが一貫）
        state_throw_index: 投番（0..15）
        state_end_index: 現在のエンド番号（0起算想定）
        state_score_diff: これまでのエンドの得点差（to-play視点）
    """
    state_sheet: List[Tuple<float, float]]
    state_hammer: bool
    state_throw_index: int
    state_end_index: int
    state_score_diff: int


# =========================================================
#  エッジ／ノード（PUCT用）
# =========================================================

@dataclass
class SearchEdge:
    """MCTSのエッジ（手）を表す。
    
    Attributes:
        edge_action_id: 0..2047 のアクションID
        edge_shot: Shot構造体
        edge_prior_p: 事前確率P
        edge_visit_count: 訪問回数N
        edge_value_sum: 累積価値W
        edge_q_value: 平均価値Q
        edge_child_ref: 遷移先ノード参照
        edge_is_illegal: 非合法手フラグ
    """
    edge_action_id: int
    edge_shot: Shot
    edge_prior_p: float
    edge_visit_count: int = 0
    edge_value_sum: float = 0.0
    edge_q_value: float = 0.0
    edge_child_ref: Optional["SearchNode"] = None
    edge_is_illegal: bool = False

    def update_backup(self, incoming_value: float) -> None:
        """バックアップ用：訪問回数と累積価値を更新してQを再計算する。
        
        Args:
            incoming_value: このエッジから見た価値（to-play視点で符号処理済み）
        """
        self.edge_visit_count += 1
        self.edge_value_sum += incoming_value
        self.edge_q_value = self.edge_value_sum / self.edge_visit_count


@dataclass
class SearchNode:
    """MCTSのノード（局面）を表す。
    
    Attributes:
        node_state: ゲーム状態
        node_policy: 2048次元の事前確率（非法手除外後に再正規化想定）
        node_children: action_id -> SearchEdge
        node_visit_total: ノードの訪問回数
        node_policy_order: P降順のアクションIDの並び
        node_unlocked_set: Progressive Wideningで解放済みの集合
    """
    node_state: GameState
    node_policy: List[float] = field(default_factory=list)
    node_children: Dict[int, SearchEdge] = field(default_factory=dict)
    node_visit_total: int = 0
    node_policy_order: List[int] = field(default_factory=list)
    node_unlocked_set: set = field(default_factory=set)

    def puct_best_action(self, cpuct_value: float) -> int:
        """PUCTに基づき (Q+U) を最大化するアクションIDを返す（解放済み・合法のみ）。
        
        Args:
            cpuct_value: 探索バランス係数
        
        Returns:
            選択された action_id
        """
        best_score = -1e18
        best_action = None
        sqrt_parent = math.sqrt(max(1, self.node_visit_total))
        for a_id in self.node_unlocked_set:
            edge_obj = self.node_children[a_id]
            if edge_obj.edge_is_illegal:
                continue
            p = max(1e-12, edge_obj.edge_prior_p)
            n = edge_obj.edge_visit_count
            q = edge_obj.edge_q_value  # 訪問0なら0.0のまま
            u = cpuct_value * p * sqrt_parent / (1.0 + n)
            score = q + u
            if score > best_score:
                best_score = score
                best_action = a_id
        if best_action is None:
            # すべてが違法または未解放の異常系（上位側で扱う）
            raise RuntimeError("選択可能なアクションがありません（全て違法または未解放）。")
        return best_action

    def unlock_by_policy(self, target_count: int) -> None:
        """policy降順に基づき、未解放のアクションを target_count 個まで解放する。
        
        Args:
            target_count: 解放後の合計数（未満なら追加解放する）
        """
        while len(self.node_unlocked_set) < target_count:
            idx = len(self.node_unlocked_set)
            if idx >= len(self.node_policy_order):
                break
            a_id = self.node_policy_order[idx]
            self.node_unlocked_set.add(a_id)


# =========================================================
#  MCTS（PUCT + Progressive Widening）
# =========================================================

class PUCTSearch:
    """ValueHeadなしのPUCT探索（1エンド内で終端スコアまで展開する実装）。
    
    - Progressive Widening: M(N_parent) = max(c, floor(k * N_parent^a))
    - 非合法手はノード単位で恒久除外
    - 報酬は[-8,+8]を[-1,+1]に正規化し、バックアップで手番交代ごとに符号反転
    """

    def __init__(
        self,
        mcts_cpuct: float,
        widen_c_min: int,
        widen_k_coef: float,
        widen_a_exp: float,
        backup_use_alt_sign: bool = True,
        re_normalize_after_mask: bool = True,
    ) -> None:
        """PUCTSearchの初期化。
        
        Args:
            mcts_cpuct: U項の係数
            widen_c_min: Progressive Wideningの下限c
            widen_k_coef: k
            widen_a_exp: a
            backup_use_alt_sign: バックアップ時に(-1)^dの符号反転を行うか
            re_normalize_after_mask: 非合法除外後にPを再正規化するか
        """
        self.mcts_cpuct = mcts_cpuct
        self.widen_c_min = widen_c_min
        self.widen_k_coef = widen_k_coef
        self.widen_a_exp = widen_a_exp
        self.backup_use_alt_sign = backup_use_alt_sign
        self.re_normalize_after_mask = re_normalize_after_mask

    # ---------- 不足情報：外部依存インターフェイス ----------
    # PolicyNetwork.evaluate(input_state: "GameState") -> List[float]  # 2048次元の確率
    # Simulator.step(prev_state: "GameState", shot_info: "Shot") -> Tuple["GameState", bool]
    # Rules.is_end(next_state: "GameState") -> Tuple[bool, int]

    def run_search(
        self,
        search_root_state: GameState,
        search_budget_sims: int,
        external_policy: "PolicyNetwork",
        external_simulator: "Simulator",
        external_rules: "Rules",
    ) -> Tuple[int, Dict[int, SearchEdge]]:
        """探索本体。指定回数シミュレーションし、ルートの子エッジ情報を返す。
        
        Args:
            search_root_state: 探索開始局面
            search_budget_sims: シミュレーション回数
            external_policy: ポリシーネットワーク
            external_simulator: シミュレータ
            external_rules: ルール（終端判定とスコア）
        
        Returns:
            (best_action_id, root_children_dict)
        """
        root_node = self._create_root_node(search_root_state, external_policy)

        for _ in range(search_budget_sims):
            self._simulate_once(
                simulate_node=root_node,
                external_policy=external_policy,
                external_simulator=external_simulator,
                external_rules=external_rules,
            )

        # ルートの最終手の決め方（argmax N で固定：必要なら argmax Q に変更可能）
        best_id = self._select_root_action_by_visit(root_node)
        return best_id, root_node.node_children

    def _create_root_node(self, init_state: GameState, external_policy: "PolicyNetwork") -> SearchNode:
        """ルートノードを作成し、ポリシー取得・並べ替え・初期解放を行う。
        
        Args:
            init_state: ルート局面
            external_policy: NNポリシー
        
        Returns:
            SearchNode
        """
        policy_raw = self._evaluate_policy_dist(init_state, external_policy)
        node = SearchNode(node_state=init_state)
        node.node_policy = policy_raw[:]  # 違法除外は各展開時に行う
        node.node_policy_order = sorted(range(ActionSpace.ACTION_SIZE), key=lambda i: node.node_policy[i], reverse=True)

        # 初期解放
        target_m = self._progressive_m_value(parent_visits=node.node_visit_total)
        node.unlock_by_policy(target_m)
        # 子エッジの器を用意
        for a_id in node.node_unlocked_set:
            if a_id not in node.node_children:
                vx, vy, sp, _, _ = ActionSpace.index_to_params(a_id)
                node.node_children[a_id] = SearchEdge(
                    edge_action_id=a_id,
                    edge_shot=Shot(shot_vx=vx, shot_vy=vy, shot_spin=sp),
                    edge_prior_p=node.node_policy[a_id],
                )
        return node

    def _simulate_once(
        self,
        simulate_node: SearchNode,
        external_policy: "PolicyNetwork",
        external_simulator: "Simulator",
        external_rules: "Rules",
    ) -> None:
        """1回のシミュレーション（Select→Expand/Evaluate→Backup）を実行する。
        
        Args:
            simulate_node: ルートノード
            external_policy: NNポリシー
            external_simulator: シミュレータ
            external_rules: ルール
        """
        path_stack: List[Tuple[SearchNode, int, SearchEdge]] = []
        current_node = simulate_node
        ply_depth = 0

        # --- SELECT + EXPAND ループ ---
        while True:
            current_node.node_visit_total += 1  # ノード訪問カウント

            # Progressive Widening 解放
            target_m = self._progressive_m_value(parent_visits=current_node.node_visit_total)
            current_node.unlock_by_policy(target_m)
            for a_id in current_node.node_unlocked_set:
                if a_id not in current_node.node_children:
                    vx, vy, sp, _, _ = ActionSpace.index_to_params(a_id)
                    current_node.node_children[a_id] = SearchEdge(
                        edge_action_id=a_id,
                        edge_shot=Shot(shot_vx=vx, shot_vy=vy, shot_spin=sp),
                        edge_prior_p=current_node.node_policy[a_id],
                    )

            # 候補が無ければ打ち切り（異常系）
            candidate_ids = [aid for aid in current_node.node_unlocked_set if not current_node.node_children[aid].edge_is_illegal]
            if not candidate_ids:
                # すべてが違法に閉ざされた場合の救済：探索を終了し、中立値0でバックアップ
                self._backup_path(path_stack, leaf_value=0.0)
                return

            # PUCTで選択
            try:
                chosen_id = current_node.puct_best_action(self.mcts_cpuct)
            except RuntimeError:
                # 候補がない（すべて違法）—救済
                self._backup_path(path_stack, leaf_value=0.0)
                return

            chosen_edge = current_node.node_children[chosen_id]

            # 展開されていなければ遷移を試みる
            if chosen_edge.edge_child_ref is None:
                new_state, is_legal = external_simulator.step(current_node.node_state, chosen_edge.edge_shot)
                if not is_legal:
                    # 非合法なら除外して選び直し（この1手はバックアップしない）
                    chosen_edge.edge_is_illegal = True
                    # 次のループで再選択
                    continue

                # 子ノード生成
                child_node = SearchNode(node_state=new_state)
                chosen_edge.edge_child_ref = child_node
                path_stack.append((current_node, chosen_id, chosen_edge))

                # 終端判定
                is_terminal, end_score = external_rules.is_end(new_state)
                if is_terminal:
                    leaf_val = self._normalize_reward(end_score)  # [-1,+1]
                    # to-play視点化は backup 内で (-1)^d を適用
                    self._backup_path(path_stack, leaf_value=leaf_val)
                    return

                # 非終端ならポリシー評価
                child_policy = self._evaluate_policy_dist(new_state, external_policy)
                child_node.node_policy = child_policy[:]
                child_node.node_policy_order = sorted(range(ActionSpace.ACTION_SIZE), key=lambda i: child_node.node_policy[i], reverse=True)

                # 即座に展開ループに戻る（次手番）
                current_node = child_node
                ply_depth += 1
                continue

            else:
                # 既展開ならそのまま一手進める
                path_stack.append((current_node, chosen_id, chosen_edge))
                current_node = chosen_edge.edge_child_ref
                ply_depth += 1
                # 次のループでさらに選択

    def _backup_path(self, path_edges: List[Tuple[SearchNode, int, SearchEdge]], leaf_value: float) -> None:
        """葉からルートに向けてパス上のエッジをバックアップする。
        
        Args:
            path_edges: (親ノード, action_id, エッジ) のスタック
            leaf_value: 葉局面のto-play視点の価値（[-1,+1]）
        """
        # 下から上へ：深さ d 手前のノードから見た価値は (-1)^d * leaf_value
        for depth_index, (_, _, edge_obj) in enumerate(reversed(path_edges)):
            if self.backup_use_alt_sign:
                signed_val = leaf_value * ((-1.0) ** depth_index)
            else:
                signed_val = leaf_value
            edge_obj.update_backup(signed_val)

    def _progressive_m_value(self, parent_visits: int) -> int:
        """Progressive WideningのM値を返す。
        
        Args:
            parent_visits: 親ノードの訪問回数
        
        Returns:
            解放すべきアクション数M
        """
        m_val = max(self.widen_c_min, int(self.widen_k_coef * (parent_visits ** self.widen_a_exp)))
        return m_val

    def _evaluate_policy_dist(self, eval_state: GameState, external_policy: "PolicyNetwork") -> List[float]:
        """ポリシー分布の取得と（必要なら）再正規化を行う。
        
        Args:
            eval_state: 評価対象の局面
            external_policy: ポリシーネットワーク
        
        Returns:
            2048次元の確率分布
        """
        raw_p = external_policy.evaluate(eval_state)  # 不足情報（外部）
        if self.re_normalize_after_mask:
            s = sum(max(0.0, p) for p in raw_p)
            if s <= 0.0:
                # 全ゼロ等の異常時は一様分布にフォールバック
                return [1.0 / ActionSpace.ACTION_SIZE] * ActionSpace.ACTION_SIZE
            return [max(0.0, p) / s for p in raw_p]
        return raw_p

    def _select_root_action_by_visit(self, root_node: SearchNode) -> int:
        """ルートの最終手を訪問回数最大（argmax N）で決める。
        
        Args:
            root_node: ルートノード
        
        Returns:
            最終選択の action_id
        """
        if not root_node.node_children:
            raise RuntimeError("ルートに子が存在しません。探索が行われていない可能性があります。")
        # 解放済み・合法のみを対象
        candidates = [
            (aid, edge_obj.edge_visit_count)
            for aid, edge_obj in root_node.node_children.items()
            if (aid in root_node.node_unlocked_set) and (not edge_obj.edge_is_illegal)
        ]
        if not candidates:
            # 万一全て違法なら、最大Pの手でフォールバック
            return root_node.node_policy_order[0]
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    @staticmethod
    def _normalize_reward(raw_score: int) -> float:
        """[-8,+8] の整数スコアを [-1,+1] に線形正規化する。
        
        Args:
            raw_score: エンドの最終スコア（to-play視点・整数）
        
        Returns:
            正規化スコア
        """
        return max(-1.0, min(1.0, raw_score / 8.0))


# =========================================================
#  外部インターフェイスのプロトタイプ（ダミー）
#  ※ 本番ではユーザ側で実装する想定。ここではNotImplementedErrorを投げる。
# =========================================================

class PolicyNetwork:
    """ポリシーネットワークの抽象インターフェイス。"""

    def evaluate(self, policy_input_state: GameState) -> List[float]:
        """与えられた局面に対して2048次元の確率分布を返す。
        
        Args:
            policy_input_state: 入力局面
        
        Returns:
            2048次元の確率分布（softmax済み）
        """
        # PolicyNetwork.evaluate(input_state: "GameState") -> List[float]  # 2048
        raise NotImplementedError("ポリシーネットワークの実装が必要です。")


class Simulator:
    """1手だけシミュレーションを進める抽象インターフェイス。"""

    def step(self, sim_prev_state: GameState, sim_shot_info: Shot) -> Tuple[GameState, bool]:
        """1手進めて、次状態と合法フラグを返す。
        
        Args:
            sim_prev_state: 現局面
            sim_shot_info: 実行ショット
        
        Returns:
            (次局面, 合法フラグ)
        """
        # Simulator.step(prev_state: "GameState", shot_info: "Shot") -> Tuple["GameState", bool]
        raise NotImplementedError("シミュレータの実装が必要です。")


class Rules:
    """終端判定とスコア算出の抽象インターフェイス。"""

    def is_end(self, rule_state: GameState) -> Tuple[bool, int]:
        """終端かどうか、終端ならスコア（[-8,+8]）を返す。
        
        Args:
            rule_state: 判定対象局面
        
        Returns:
            (終端フラグ, スコア)
        """
        # Rules.is_end(next_state: "GameState") -> Tuple[bool, int]
        raise NotImplementedError("終端判定の実装が必要です。")


# =========================================================
#  使用例（ダミーの最小モックで配線をテスト）
# =========================================================

class DummyPolicy(PolicyNetwork):
    """デモ用のダミーポリシー。均一分布を返す。"""

    def evaluate(self, policy_input_state: GameState) -> List[float]:
        """疑似的に一様分布を返す。"""
        return [1.0 / ActionSpace.ACTION_SIZE] * ActionSpace.ACTION_SIZE


class DummySimulator(Simulator):
    """デモ用のダミーシミュレータ。投番だけ進める。"""

    def step(self, sim_prev_state: GameState, sim_shot_info: Shot) -> Tuple[GameState, bool]:
        """疑似的に合法として投番+1しただけの状態を返す。"""
        next_throw = sim_prev_state.state_throw_index + 1
        new_state = GameState(
            state_sheet=sim_prev_state.state_sheet[:],
            state_hammer=sim_prev_state.state_hammer,
            state_throw_index=next_throw,
            state_end_index=sim_prev_state.state_end_index,
            state_score_diff=sim_prev_state.state_score_diff,
        )
        return new_state, True


class DummyRules(Rules):
    """デモ用のダミールール。投番==15で終端、スコアは0固定。"""

    def is_end(self, rule_state: GameState) -> Tuple[bool, int]:
        """投番が15なら終端としてスコア0を返す。"""
        return (rule_state.state_throw_index >= 15, 0)


def demo_run() -> None:
    """ダミー実装でPUCTを1回走らせ、ルートのベスト手IDを表示するデモ。"""
    initial = GameState(
        state_sheet=[],
        state_hammer=True,
        state_throw_index=0,
        state_end_index=0,
        state_score_diff=0,
    )
    searcher = PUCTSearch(
        mcts_cpuct=1.5,
        widen_c_min=4,
        widen_k_coef=2.0,
        widen_a_exp=0.5,
        backup_use_alt_sign=True,
        re_normalize_after_mask=True,
    )
    pol = DummyPolicy()
    sim = DummySimulator()
    rul = DummyRules()

    best_id, root_children = searcher.run_search(
        search_root_state=initial,
        search_budget_sims=50,
        external_policy=pol,
        external_simulator=sim,
        external_rules=rul,
    )
    print("best action id:", best_id)
    # 任意で可視化：root_children[best_id].edge_visit_count など


if __name__ == "__main__":
    # デモ（本番では削除可）
    demo_run()
