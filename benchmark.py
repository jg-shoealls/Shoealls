import time
import numpy as np

def with_import(data):
    from scipy.signal import resample
    return resample(data, 128, axis=0)

from scipy.signal import resample
def without_import(data):
    return resample(data, 128, axis=0)

data = np.random.randn(200, 6)

start = time.time()
for _ in range(1000):
    with_import(data)
print(f"With import: {time.time() - start:.4f}s")

start = time.time()
for _ in range(1000):
    without_import(data)
print(f"Without import: {time.time() - start:.4f}s")
