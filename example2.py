import torch
import time


# x=torch.randn(dim,dim)
# y=torch.randn(dim,dim)
# start_time = time.time()
# z=torch.matmul(x,y)
# elapsed_time = time.time() - start_time
# print('CPU_time = ',elapsed_time)
# mod = SourceModule("""
# __global__ void kernel_function() {
#     torch::Tensor x = torch::randn({dim, dim}, torch::device(torch::kCUDA));
#     torch::Tensor y = torch::randn({dim, dim}, torch::device(torch::kCUDA));
#     auto start_time = std::chrono::high_resolution_clock::now();
#     torch::Tensor z = torch::matmul(x, y);
#     auto elapsed_time = std::chrono::high_resolution_clock::now() - start_time;
#     std::cout << "GPU_time = " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() << " ms" << std::endl;
# }
# """)
#
# # Получите функцию CUDA kernel
# kernel = mod.get_function("kernel_function")
#
# # Установите размер сетки вычислительных блоков
# grid_size = 256
#
# # Задайте количество рабочих блоков на видеокарте NVIDIA
# kernel(grid=(grid_size, 1, 1), block=(256, 1, 1))
from numba_expr import njit, prange
device=torch.device("cuda")


def work():
    dim = 30000
    x = torch.randn(dim, dim, device=device)
    y = torch.randn(dim, dim, device=device)
    start_time = time.time()
    z = torch.matmul(x, y)
    elapsed_time = time.time() - start_time
    print('GPU_time = ', elapsed_time)
    q = 542.46981243540077
    print(q/elapsed_time/1132)

work()

#
# # Пример установки максимальной частоты GPU (для GPU 0)
# target_gpu = 0
# max_performance = 60  # устанавливаем 100% производительности
# # subprocess.call(["nvidia-xconfig -enable-all-gpus -cool-bits=28 <b>--allow-empty-initial-configuration</b>"])
# subprocess.call(["nvidia-smi", "-i", str(target_gpu), "-pl", str(max_performance)])
# cuda.Device(0).set_attribute(cuda.device_attribute.max_threads_per_block, 32)
# cuda.Device(0).set_attribute(cuda.device_attribute.max_grid_size, (1000000, 1))
# import nvidia_smi
# import pynvml
# # Инициализация библиотеки
# nvidia_smi.nvmlInit()
#
# # Получаем управление для первой доступной видеокарты
# handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
#
# # Устанавливаем ограничение на максимальную частоту ядра GPU в MHz
# target_performance = 1000  # например, устанавливаем 1000 МГц
# pynvml.nvmlDeviceSetApplicationsClocks(handle, pynvml.NVML_VOLATILE_ECC, target_performance)
#
# # Завершение работы с библиотекой
# nvidia_smi.nvmlShutdown()
# 17
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
