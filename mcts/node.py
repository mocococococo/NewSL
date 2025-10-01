import math
import random
from mcts.kde import KDE
from mcts.constant import VAR_X, VAR_Y, STD_X, STD_Y
from dc3client.models import StoneRotation

class Node:
    def __init__(self, x, y, spin, prob): #初期値
        self.m_v = 0.0
        self.m_visits = 0.0

        self.m_move = [x, y, spin]
        self.m_prob = prob

        self.m_parent = None
        self.m_children = []

        self.m_kde_0 = KDE()
        self.m_kde_1 = KDE()
        
        self.m_init_infos = []

    def add_node(self, x, y, spin, prob): #新しい子ノードをノードに追加
        added_node = Node(x, y, spin, prob)
        self.m_children.append(added_node)
        added_node.m_parent = self
        return added_node

    def get_eval(self, is_white): #ノードの評価値を取得
        score = self.m_v / self.m_visits
        if is_white:
            score = -1 * score
        return score

    def ucb_select(self, is_white, ucb_const): #UCBに基づいてノードを選択
        best = None
        best_value = -float('inf')

        num_total_child_visits = sum(child.m_visits for child in self.m_children)
        numerator = math.log(num_total_child_visits)

        for child in self.m_children:
            #rint(is_white)
            E_v = child.get_eval(not is_white) #もともとはnotがついていなかった。要検討
            denom = child.m_visits
            value = E_v + ucb_const * math.sqrt(numerator / denom)

            if value > best_value:
                best_value = value
                best = child

        assert best is not None
        return best

    def ucb_select2(self, is_white, ucb_const): #UCBに基づいてノードを選択
        best = None
        best_value = -float('inf')

        num_total_child_visits = sum(child.m_visits for child in self.m_children)
        numerator = math.log(num_total_child_visits)

        for child in self.m_children:
            #rint(child.m_visits, child.m_v)
            E_v = child.get_eval(not is_white)
            denom = child.m_visits
            value = E_v + ucb_const * math.sqrt(numerator / denom)

            if value > best_value:
                best_value = value
                best = child

        assert best is not None
        return best

    def sample_move(self, num_sample, l): #ランダムな実行を行い、その中で評価が高いものを出力
        best_move = [0.0, 0.0, 0]
        selected_x, selected_y, selected_spin = self.m_move

        best_value = float('inf')

        for i in range(num_sample):
            err_x = random.uniform(-l * 2 * STD_X, l * 2 * STD_X)
            err_y = random.uniform(-l * 2 * STD_Y, l * 2 * STD_Y)

            sample_x = selected_x + err_x
            sample_y = selected_y + err_y

            if selected_spin == 0:
                value = self.m_kde_0.eval(sample_x, sample_y)
            else:
                value = self.m_kde_1.eval(sample_x, sample_y)

            if value < best_value:
                best_value = value
                best_move = [sample_x, sample_y, selected_spin]

        return best_move

    def kr_update(self, v): #ノードの行価値の更新及び、カーネル密度推定、昔に作ったから要確認
        first_update = (self.m_visits == 0)
        if self.m_parent is None: #Nodeが根ノードかどうか
            self.m_visits += 1
            self.m_v += v
        else:
            self.m_visits += 1
            self.m_v += v

            this_move = self.m_move

            if first_update:
                if this_move[2] == StoneRotation.clockwise: #回転方向
                    self.m_kde_0.add_ob(this_move[0], this_move[1])
                else:
                    self.m_kde_1.add_ob(this_move[0], this_move[1])
'''
            for sibling_node in self.m_parent.m_children:
                sibling_move = sibling_node.m_move
                if (self == sibling_node) or (this_move[2] != sibling_move[2]): #同じ回転のみ考える
                    continue

                dx = sibling_move[0] - this_move[0]
                dy = sibling_move[1] - this_move[1]
                k = KDE.kernel(dx, dy)

                if first_update:
                    self.m_visits += k
                    self.m_v += k * sibling_node.get_eval(False)

                    if this_move[2] == StoneRotation.clockwise:
                        sibling_node.m_kde_0.add_ob(this_move[0], this_move[1])
                    else:
                        sibling_node.m_kde_1.add_ob(this_move[0], this_move[1])

                    if sibling_move[2] == StoneRotation.clockwise:
                        self.m_kde_0.add_ob(sibling_move[0], sibling_move[1])
                    else:
                        self.m_kde_1.add_ob(sibling_move[0], sibling_move[1])

                sibling_node.m_visits += k
                sibling_node.m_v += k * v
'''

