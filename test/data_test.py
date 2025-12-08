import numpy as np

path = r"C:\Users\kirby\Programs\NewSL\data\sl_data_0.npz"
data = np.load(path)
print("keys:", data.files)   # ← ここに 'value' があるか確認
