#include <torch/extension.h>

#include <cufftXt.h>

template <typename T> __global__ void cuda_backward_kernel(
        const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> sinogram,
        const torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> angles,
        const torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> positions,
        torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> image,
        const size_t batch_count,
        const size_t image_size,
        const size_t angle_count,
        const size_t position_count,
        const uint8_t filter_id) {
    const size_t batch_idx = static_cast<size_t>(blockIdx.z*blockDim.z+threadIdx.z);
    const size_t angle_idx = static_cast<size_t>(blockIdx.y*blockDim.y+threadIdx.y);
    const size_t pos_idx   = static_cast<size_t>(blockIdx.x*blockDim.x+threadIdx.x);
    if(batch_idx >= batch_count || angle_idx >= angle_count || pos_idx >= position_count) {
        return;
    }

    if(angle_idx >= image_size || pos_idx >= image_size) {
        return;
    }
    image[batch_idx][0][angle_idx][pos_idx] = pos_idx+angle_idx*image_size; //TODO
}

torch::Tensor cuda_backward(const torch::Tensor sinogram, const torch::Tensor angles, const torch::Tensor positions, const size_t image_size, const uint8_t filter_id) {
    const dim3 threads(8, 8, 2);
    const dim3 blocks(
        ceil(positions.sizes()[0]/static_cast<float>(threads.x)), 
        ceil(angles.sizes()[0]/static_cast<float>(threads.y)), 
        ceil(sinogram.sizes()[0]/static_cast<float>(threads.z))
    );
    torch::Tensor image = torch::zeros({1, 1, static_cast<signed long>(image_size), static_cast<signed long>(image_size)}, c10::TensorOptions(torch::kCUDA));
    AT_DISPATCH_FLOATING_TYPES(sinogram.scalar_type(), "radon_cuda_backward", ([&] {
            cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
                sinogram.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                angles.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
                positions.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
                image.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                sinogram.sizes()[0],
                image_size,
                sinogram.sizes()[3],
                sinogram.sizes()[2],
                filter_id
            );
        })
    );
    return image;
}