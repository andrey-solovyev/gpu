import numpy as np
from pylab import imshow, show
from timeit import default_timer as timer

from numba import jit, prange, cuda, njit
from matplotlib.pylab import imshow, ion


@cuda.jit(device=True)
def mandelbrot(x, y, max_iters):
    c = complex(x, y)
    z = 0j
    for i in prange(max_iters):
        z = z * z + c
        if z.real * z.real + z.imag * z.imag >= 4:
            return 255 * i // max_iters
    return 255


@cuda.jit
def create_fractal(min_x, max_x, min_y, max_y, image, iters):
    height = image.shape[0]
    width = image.shape[1]
    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height
    x, y = cuda.grid(2)  # x = blockIdx.x * blockDim.x + threadIdx.xif x < width and y < height:
    if x < width and y < height:
        real = min_x + x * pixel_size_x
        imag = min_y + y * pixel_size_y
        color = mandelbrot(real, imag, iters)
        image[y, x] = color


image = np.zeros((500 * 4, 750 * 4), dtype=np.uint8)

start = timer()
nthreads = 32
nblocksy = (200 // nthreads) + 1
nblocksx = (200 // nthreads) + 1
config = (nblocksx, nblocksy), (nthreads, nthreads)
print(config)
create_fractal[config](-2.0, 1.0, -1.0, 1.0, image, 10000)
dt = timer() - start

print("create_fractal with parallel %f s" % dt)


@jit(nopython=True, parallel=False, target_backend='cuda')
def mandel(x, y, max_iters):
    i = 0
    c = complex(x, y)
    z = 0.0j
    for i in range(max_iters):
        z = z * z + c
        if (z.real * z.real + z.imag * z.imag) >= 4:
            return i

    return 255


@jit(nopython=True, parallel=False, target_backend='cuda')
def create_fractal(min_x, max_x, min_y, max_y, image, iters):
    height = image.shape[0]
    width = image.shape[1]

    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height
    for x in range(width):
        real = min_x + x * pixel_size_x
        for y in range(height):
            imag = min_y + y * pixel_size_y
            color = mandel(real, imag, iters)
            image[y, x] = color

    return image


image = np.zeros((500 * 4, 750 * 4), dtype=np.uint8)
start = timer()
nthreads = 1
nblocksy = 1
nblocksx = 1
config = (nblocksx, nblocksy), (nthreads, nthreads)
create_fractal(-2.0, 1.0, -1.0, 1.0, image, 10000)
dt = timer() - start

print("create_fractal without parallel %f s" % dt)


# :\Users\andru\AppData\Local\Programs\Python\Python310\lib\site-packages\numba\cuda\dispatcher.py:536: NumbaPerformanceWarning: Grid size 49 will likely result in GPU under-utilization due to low occupancy.
#   warn(NumbaPerformanceWarning(msg))
# ((7, 7), (32, 32))
# C:\Users\andru\AppData\Local\Programs\Python\Python310\lib\site-packages\numba\cuda\cudadrv\devicearray.py:886: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
#   warn(NumbaPerformanceWarning(msg))
# create_fractal with parallel 0.328841 s
# create_fractal without parallel 40.372357 s