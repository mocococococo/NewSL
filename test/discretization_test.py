import numpy as np

from nn.feature import discretization_velocity
from policy_shot import index_to_shot
from board.constant import VX_SIZE, VY_SIZE

vx = 0.26
vy = 3.5
rotation = 1  # 0: cw, 1: ccw
policy_plane = np.zeros(shape=(2, VX_SIZE * VY_SIZE))

vindex = discretization_velocity(vx, vy)
policy_plane[rotation][vindex] = 1

selected_index = np.argmax(policy_plane.reshape((VX_SIZE * VY_SIZE) * 2).astype(np.int64))
vx_out, vy_out, rotation_out = index_to_shot(selected_index)
print(f"Input vx: {vx}, vy: {vy}, rotation: {rotation}")
print(f"Output vx: {vx_out}, vy: {vy_out}, rotation: {rotation_out}")