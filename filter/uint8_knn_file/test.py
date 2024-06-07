import uint8_knn
import faiss
import numpy as np
from time import time

nqs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
ts1 = []
ts2 = []

n, d = 10000, 1536
topk = 10

for nq in nqs:
    X = np.random.randint(0, 255, size=(n, d)).astype('uint8')
    Q = np.random.randint(0, 255, size=(nq, d)).astype('uint8')

    # Xf = np.random.randint(0, 255, size=(n, d)).astype('float')
    # Qf = np.random.randint(0, 255, size=(nq, d)).astype('float')

    res1 = uint8_knn.knn(Q, X, topk).reshape(nq, topk)
    _, res2 = faiss.knn(Q, X, topk)
    diff = (res1 != res2).sum()
    if diff > 0:
        print(f"diff = {diff}")
    st = time()
    iters = 10
    for _ in range(iters):
        uint8_knn.knn(Q, X, topk, nthreads=8).reshape(nq, topk)
        # faiss.knn(Q, X, topk)
    ed = time()
    ts1.append((ed - st) * 1000 / iters)

    st = time()
    for _ in range(iters):
        # uint8_knn.knn(Q, X, topk).reshape(nq, topk)
        faiss.knn(Q, X, topk)
    ed = time()
    ts2.append((ed - st) * 1000 / iters)
    print(f'nq = {nq}, uint8_knn time {ts1[-1]:.2f}ms, faiss time {ts2[-1]:.2f}ms')