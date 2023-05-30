#include <torch/extension.h>
#include <omp.h>

#ifdef RADON_CUDA_AVAILABLE
torch::Tensor cudaForward(torch::Tensor image, torch::Tensor thetas, torch::Tensor positions);
torch::Tensor cudaBackward(torch::Tensor sinogram, torch::Tensor thetas, torch::Tensor distances, size_t image_size);
torch::Tensor cudaMatrix(torch::Tensor image, torch::Tensor thetas, torch::Tensor positions);
#endif
torch::Tensor cpuForward(torch::Tensor image, torch::Tensor thetas, torch::Tensor positions);
torch::Tensor cpuBackward(torch::Tensor sinogram, torch::Tensor thetas, torch::Tensor distances, size_t image_size);
torch::Tensor cpuMatrix(torch::Tensor image, torch::Tensor thetas, torch::Tensor positions);
//torch::Tensor matrixBackward(torch::Tensor sino, torch::Tensor thetas, torch::Tensor positions, size_t image_size);
//torch::Tensor matrixForward(torch::Tensor image, torch::Tensor thetas, torch::Tensor positions);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    #ifdef RADON_CUDA_AVAILABLE
    module.def("cuda_forward", &cudaForward);
    module.def("cuda_backward", &cudaBackward);
    module.def("cuda_matrix", &cudaMatrix);
    #endif
    module.def("cpu_forward", &cpuForward);
    module.def("cpu_backward", &cpuBackward);
    module.def("cpu_matrix", &cpuMatrix);
    //module.def("matrix_backward", &matrixBackward);
    //module.def("matrix_forward", &matrixForward);
}