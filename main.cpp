#include <iostream>

#include <torch/torch.h>
#include <iostream>
#include <chrono>
using namespace std;



void work(int dim) {
    torch::Tensor x = torch::randn({dim, dim}, torch::device(torch::kCUDA));
    torch::Tensor y = torch::randn({dim, dim}, torch::device(torch::kCUDA));
    auto start_time = std::chrono::high_resolution_clock::now();
    torch::Tensor z = torch::matmul(x, y);
    auto elapsed_time = std::chrono::high_resolution_clock::now() - start_time;
    std::cout << "GPU_time = " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() << " ms" << std::endl;
}

int main()
{
    work(256);

    return 0;
}