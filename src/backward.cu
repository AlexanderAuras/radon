#include <torch/extension.h>

#include "filters.hpp"

__device__ float cudaRamLakFilter(float x, float max) {
    return abs(x);
}

__device__ float cudaHannFilter(float x, float max) {
    return abs(x)*0.5*(1.0+cos(2.0*M_PI*x/max));
}

template <typename T> __global__ void cudaBackwardKernel(
        const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> sinogram,
        const torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> angles,
        const torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> positions,
        torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> image,
        const size_t batchCount,
        const size_t imageSize,
        const size_t angleCount,
        const size_t positionCount,
        const uint8_t filterId) {
    const size_t batchIdx = static_cast<size_t>(blockIdx.z*blockDim.z+threadIdx.z);
    const size_t angleIdx = static_cast<size_t>(blockIdx.y*blockDim.y+threadIdx.y);
    const size_t posIdx   = static_cast<size_t>(blockIdx.x*blockDim.x+threadIdx.x);
    if(batchIdx >= batchCount || angleIdx >= angleCount || posIdx >= positionCount) {
        return;
    }

    /*switch(filterId) {
        case RADON_RAM_LAK_FILTER_ID: cudaRamLakFilter(0.01, 1.0); break;
        case RADON_HANN_FILTER_ID:    cudaHannFilter(0.01, 1.0);   break;
    }*/
    
    if(angleIdx >= imageSize || posIdx >= imageSize) {
        return;
    }
    image[batchIdx][0][angleIdx][posIdx] = posIdx+angleIdx*imageSize; //TODO
}

torch::Tensor cudaBackward(const torch::Tensor sinogram, const torch::Tensor angles, const torch::Tensor positions, const size_t imageSize, const uint8_t filterId) {
    const dim3 threads(8, 8, 2);
    const dim3 blocks(
        ceil(positions.sizes()[0]/static_cast<float>(threads.x)), 
        ceil(angles.sizes()[0]/static_cast<float>(threads.y)), 
        ceil(sinogram.sizes()[0]/static_cast<float>(threads.z))
    );
    torch::Tensor image = torch::zeros({1, 1, static_cast<signed long>(imageSize), static_cast<signed long>(imageSize)}, c10::TensorOptions(torch::kCUDA));
    AT_DISPATCH_FLOATING_TYPES(sinogram.scalar_type(), "radon_cudaBackward", ([&] {
            cudaBackwardKernel<scalar_t><<<blocks, threads>>>(
                sinogram.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                angles.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
                positions.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
                image.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                sinogram.sizes()[0],
                imageSize,
                sinogram.sizes()[3],
                sinogram.sizes()[2],
                filterId
            );
        })
    );
    return image;
}