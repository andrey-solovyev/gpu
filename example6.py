from __future__ import division
from timeit import default_timer as timer

from numba import cuda, prange, jit, njit
import numpy
import math


@cuda.jit
def matmul(A, B, C):
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.
        for k in prange(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp


@njit(nopython=True, parallel=False)
def matmul_without(A, B, C):
    row, col = (0, 0)
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp


A = numpy.full((5000, 5000), 3)  # matrix containing all 3's
B = numpy.full((5000, 5000), 4)  # matrix containing all 4's

A_global_mem = cuda.to_device(A)
B_global_mem = cuda.to_device(B)

C_global_mem = cuda.device_array((5000, 5000))

threadsperblock = (32, 32)
blockspergrid_x = int(math.ceil(A.shape[0] / threadsperblock[0]))
blockspergrid_y = int(math.ceil(B.shape[1] / threadsperblock[1]))
blockspergridWithParallel = (blockspergrid_x, blockspergrid_y)
print(blockspergridWithParallel)
start = timer()
matmul[blockspergridWithParallel, threadsperblock](A_global_mem, B_global_mem, C_global_mem)
timeWithParallel = timer() - start
print("with GPU with parallel:", timeWithParallel)

threadsperblock = (1, 1)
blockspergrid_x = int(math.ceil(A.shape[0] / threadsperblock[0]))
blockspergrid_y = int(math.ceil(B.shape[1] / threadsperblock[1]))
blockspergrid = (blockspergrid_x, blockspergrid_y)
print(blockspergrid)
start = timer()
C_global_mem = cuda.device_array((5000, 5000))
matmul_without(A, B, A)
timeWithoutParallel = timer() - start
print("with GPU without parallel:", timeWithoutParallel)
print("Ep(n):", timeWithoutParallel / timeWithParallel / blockspergridWithParallel[0])
