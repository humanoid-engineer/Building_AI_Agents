import numpy as np

v1 = np.array([1, 2, 3])
v2 = np.array([2, 1, 3])

cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
print(cos)