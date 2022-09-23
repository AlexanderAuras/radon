#include <torch/extension.h>

#include "filters.hpp"

float cpuRamLakFilter(float x, float max) {
    return abs(x);
}

float cpuHannFilter(float x, float max) {
    return abs(x)*0.5*(1.0+cos(2.0*M_PI*x/max));
}

torch::Tensor cpuBackward(const torch::Tensor sinogram, const torch::Tensor angles, const torch::Tensor positions, const size_t imageSize, const uint8_t filterId) {
    torch::Tensor image = torch::zeros({1, 1, static_cast<signed long>(imageSize), static_cast<signed long>(imageSize)});
    #pragma omp parallel
    {
        /*switch(filterId) {
            case RADON_RAM_LAK_FILTER_ID: cudaRamLakFilter(0.01, 1.0); break;
            case RADON_HANN_FILTER_ID:    cudaHannFilter(0.01, 1.0);   break;
        }*/
    }
    return image;
}