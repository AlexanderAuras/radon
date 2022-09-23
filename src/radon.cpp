#include <torch/extension.h>
#include <omp.h>

#include "filters.hpp"

torch::Tensor cudaForward(torch::Tensor image, torch::Tensor angles, torch::Tensor positions);
torch::Tensor cudaBackward(torch::Tensor sinogram, torch::Tensor angles, torch::Tensor distances, size_t imageSize, const uint8_t filterId);
torch::Tensor cpuForward(torch::Tensor image, torch::Tensor angles, torch::Tensor positions);
torch::Tensor cpuBackward(torch::Tensor sinogram, torch::Tensor angles, torch::Tensor distances, size_t imageSize, const uint8_t filterId);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("cuda_forward", &cudaForward);
    module.def("cuda_backward", &cudaBackward);
    module.def("cpu_forward", &cpuForward);
    module.def("cpu_backward", &cpuBackward);
    module.attr("_RAM_LAK_FILTER_ID") = py::int_(RADON_RAM_LAK_FILTER_ID);
    module.attr("_HANN_FILTER_ID") = py::int_(RADON_HANN_FILTER_ID);
}