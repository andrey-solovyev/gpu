import numpy as np
from numba import prange, njit
from numba.cuda import jit
from timeit import default_timer as timer


@jit
def function_mattiass(A):
    for i in prange(A.shape[0]):
        for j in prange(A.shape[1]):
            k = i + j
            t = j
            s = min(k + j - 1, i - 1)
            A[i, j] = (k + 1) / 2 ** (s + t)


def function_mattiass2(A):
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            k = i + j
            t = j
            s = min(k + j - 1, i - 1)
            A[i, j] = (k + 1) / 2 ** (s + t)


arg = np.zeros((1000, 1000))
threadsperblock = 1
blockspergrid = 1
start = timer()
function_mattiass[blockspergrid, threadsperblock](arg)
time1 = timer() - start
print("with GPU with parallel:", time1)
# with GPU with parallel: 0.6733782000083011
# Вычисляем функцию Маттиаса
threadsperblock = (32, 32)
blockspergrid = 1132
start = timer()
function_mattiass[blockspergrid, threadsperblock](arg)
time2 = timer() - start
print("with GPU without parallel:", time2)

print(time2 / time1 / 128)
