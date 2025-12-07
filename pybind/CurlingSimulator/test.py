import fast_simulator as fs

pp = (0, 38.405)
v = 3.5
spin = 1

shot_a = fs.passpoint2shot(pp, v, spin)
shot_b = fs.passpointgo2shot(pp, v, spin)

print("A:", shot_a)
print("B:", shot_b)

# 障害物なしの到達点比較
da = fs.shot2dest(shot_a)
db = fs.shot2dest(shot_b)
print("dest A:", da)
print("dest B:", db)