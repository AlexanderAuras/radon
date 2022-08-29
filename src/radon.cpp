#include <torch/extension.h>

torch::Tensor cudaForward(torch::Tensor image, torch::Tensor angles, torch::Tensor positions);
torch::Tensor cudaBackward(torch::Tensor sinogram, torch::Tensor angles, torch::Tensor distances, size_t imageSize, const uint64_t filterFuncPtr);
torch::Tensor cudaInverse(torch::Tensor sinogram, torch::Tensor angles, torch::Tensor distances, size_t imageSize);

void initFilterPointers();
extern void* ramLakPtr;
extern void* hannPtr;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("forward", &cudaForward);
    module.def("backward", &cudaBackward);
    module.def("inverse", &cudaInverse);
    initFilterPointers();
    module.attr("_ram_lak_filter_ptr") = py::int_(reinterpret_cast<uint64_t>(ramLakPtr));
    module.attr("_hann_filter_ptr") = py::int_(reinterpret_cast<uint64_t>(hannPtr));
}