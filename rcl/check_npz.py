import numpy as np
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "data" / "sl_data_0.npz"
data = np.load(path)

print("file:", path)
print("keys:", data.files)

x = data["input"]
policy = data["policy"]
value = data["value"]

print("input :", x.shape, x.dtype, x.min(), x.max())
print("policy:", policy.shape, policy.dtype)
print("value :", value.shape, value.dtype)
print("log_count:", data["log_count"])

print("policy sums:", policy.sum(axis=1)[:10])
print("value sums :", value.sum(axis=1)[:10])

bad_policy = np.where(policy.sum(axis=1) <= 0)[0]
bad_value = np.where(value.sum(axis=1) <= 0)[0]

print("bad policy rows:", bad_policy[:20], "count=", len(bad_policy))
print("bad value rows :", bad_value[:20], "count=", len(bad_value))

for i in range(min(3, len(policy))):
    top = np.argsort(policy[i])[-10:][::-1]
    print(f"\nsample {i}")
    print("policy top actions:", top)
    print("policy top counts :", policy[i][top])
    print("value counts      :", value[i])
    print("value classes     :", np.nonzero(value[i])[0] - 8)
