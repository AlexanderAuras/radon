#include <torch/extension.h>
#include <omp.h>

#ifdef RADON_CUDA_AVAILABLE
torch::Tensor cudaForward(torch::Tensor image, torch::Tensor thetas, torch::Tensor positions);
torch::Tensor cudaBackward(torch::Tensor sinogram, torch::Tensor thetas, torch::Tensor distances, size_t image_size);
#endif
torch::Tensor cpuForward(torch::Tensor image, torch::Tensor thetas, torch::Tensor positions);
torch::Tensor cpuBackward(torch::Tensor sinogram, torch::Tensor thetas, torch::Tensor distances, size_t image_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    #ifdef RADON_CUDA_AVAILABLE
    module.def("cuda_forward", &cudaForward);
    module.def("cuda_backward", &cudaBackward);
    #endif
    module.def("cpu_forward", &cpuForward);
    module.def("cpu_backward", &cpuBackward);
}