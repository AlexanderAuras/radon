#include <torch/extension.h>

torch::Tensor cuda_forward(torch::Tensor image, torch::Tensor angles, torch::Tensor positions);
torch::Tensor cuda_backward(torch::Tensor sinogram, torch::Tensor angles, torch::Tensor distances, size_t image_size, const uint8_t filter_id);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("forward", &cuda_forward);
    module.def("backward", &cuda_backward);
}