#カーネル密度推定（Kernel Density Estimation）を行う
import math
from mcts.constant import VAR_X, VAR_Y

class KDE:
    def __init__(self):
        self.m_obs = []

    def add_ob(self, x, y):
        self.m_obs.append((x, y))

    @staticmethod
    def kernel(dx, dy):
        return math.exp(-0.5 * ((dx ** 2.0) / VAR_X + (dy ** 2.0) / VAR_Y))

    def eval(self, x, y):
        result = 0

        for ob in self.m_obs:
            dx = ob[0] - x
            dy = ob[1] - y

            result += self.kernel(dx, dy)

        return result
