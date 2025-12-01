from typing import List, Tuple, Optional
import math
import random

from nn.mcts.network.dual_net import DualNet

from PUCT.action import ActionSpace
from PUCT.edge import Edge
from PUCT.node import Node
from PUCT.PUCTConfig import PUCTConfig
from PUCT.silulator import Simulator
from PUCT.shot import Shot
from PUCT.state import State

class PUCTSearch:
    """
    Valueヘッド無しのポリシー単独PUCTを用いた1エンド限定の探索器。

    使い方:
        search = PUCTSearch(nn, sim, rules, cfg)
        best_action, root = search.search(root_state, n_simulations=1000)
    """

    def __init__(self, nn: DualNet, sim: Simulator, config: PUCTConfig):
        """
        構築子。

        Args:
            nn (PolicyNetwork): ポリシーネットワーク。
            sim (Simulator): 高速シミュレータ。
            config (PUCTConfig): パラメータ。
        """
        self.nn = nn
        self.sim = sim
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
        terminal, score = node.state.is_end_and_score()
        if terminal:
            node.P = [0.0] * ActionSpace.ACTION_SIZE
            node.policy_order = []
            node.unlocked_actions = []
            return

        policy_list = self.nn.evaluate(node.state)  # 長さ2048
        # ここでは非法手の情報が無いので、いったんそのまま保持（展開時に除外＆再正規化）
        # 初期の policy_order は NNの降順（暫定）
        order = list(range(ActionSpace.ACTION_SIZE))
        order.sort(key=lambda a: policy_list[a], reverse=True)

        node.P = list(policy_list)
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
            terminal, score = self.state.is_end_and_score(node.state)
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