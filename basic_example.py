from numba import jit, cuda, njit, prange
import numpy as np
# to measure exec time
from timeit import default_timer as timer
import torch


# normal function to run on cpu
@njit(parallel=True, fastmath=True, target_backend='cuda')
def func(a, b):
    c = [[5, 6], [7, 8]]
    for i in range(len(a)):
        for j in range(len(b[0])):
            for k in range(len(b)):
                c[i][j] += a[i][k] * b[k][j]

    # function optimized to run on gpu


@njit(parallel=True, fastmath=True, target_backend='cuda')
def func2(a, b):
    c = [[5, 6], [7, 8]]
    for i in range(len(a)):
        for j in range(len(b[0])):
            for k in range(len(b)):
                c[i][j] += a[i][k] * b[k][j]


@njit(parallel=True, fastmath=True, target_backend='cuda')
def mat_mul_and_sum2(img1, img2, alpha):
    NI, NJ, NK = img1.shape
    out = np.empty((NI, NJ, NK))

    for i in prange(NI):
        for j in prange(NJ):
            for k in prange(NK):
                out[i, j, k] = img1[i, j, k] * (1.0 - alpha[i, j, k]) + img2[i, j, k] * alpha[i, j, k]

    return out


@njit(parallel=False, fastmath=False, target_backend='cuda')
def mat_mul_and_sum(img1, img2, alpha):
    NI, NJ, NK = img1.shape
    out = np.empty((NI, NJ, NK))

    for i in range(NI):
        for j in range(NJ):
            for k in range(NK):
                out[i, j, k] = img1[i, j, k] * (1.0 - alpha[i, j, k]) + img2[i, j, k] * alpha[i, j, k]

    return out


if __name__ == "__main__":
    # n = 10000000
    # # a = np.ones(n, dtype=np.float64)
    # a = [[1, 2], [3, 4]]
    # b = [[5, 6], [7, 8]]
    #
    # # Инициализация результирующего тензора
    # c = [[5, 6], [7, 8]]
    # start = timer()
    # func(a, b)
    # print("with GPU and parallel:", timer() - start)
    #
    # start = timer()
    # func2(a, b)
    # print("with GPU without parallel:", timer() - start)
    N = 1000
    img1 = np.random.normal(size=(N, N, N))
    img2 = np.random.normal(size=(N, N, N))
    alpha = np.random.normal(size=(N, N, N))
    start = timer()
    A = mat_mul_and_sum(img1, img2, alpha)
    print("with GPU without parallel:", timer() - start)
    start = timer()
    B = mat_mul_and_sum2(img1, img2, alpha)
    print("with GPU with parallel:", timer() - start)
