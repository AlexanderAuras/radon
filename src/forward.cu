#include <torch/extension.h>

template <typename T> __global__ void cuda_forward_kernel(
        const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> image,
        const torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> angles,
        const torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> positions,
        torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> sinogram,
        const size_t batch_count,
        const size_t image_size,
        const size_t angle_count,
        const size_t position_count) {
    const size_t batch_idx = static_cast<size_t>(blockIdx.z*blockDim.z+threadIdx.z);
    const size_t angle_idx = static_cast<size_t>(blockIdx.y*blockDim.y+threadIdx.y);
    const size_t pos_idx   = static_cast<size_t>(blockIdx.x*blockDim.x+threadIdx.x);
    if(batch_idx >= batch_count || angle_idx >= angle_count || pos_idx >= position_count) {
        return;
    }

    sinogram[batch_idx][0][angle_idx][pos_idx] = pos_idx+angle_idx*position_count; //TODO
}

torch::Tensor cuda_forward(const torch::Tensor image, const torch::Tensor angles, const torch::Tensor positions) {
    const dim3 threads(8, 8, 2);
    const dim3 blocks(
        ceil(positions.sizes()[0]/static_cast<float>(threads.x)), 
        ceil(angles.sizes()[0]/static_cast<float>(threads.y)), 
        ceil(image.sizes()[0]/static_cast<float>(threads.z))
    );
    torch::Tensor sinogram = torch::zeros({1, 1, angles.sizes()[0], positions.sizes()[0]}, c10::TensorOptions(torch::kCUDA));
    AT_DISPATCH_FLOATING_TYPES(image.scalar_type(), "radon_cuda_forward", ([&] {
            cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
                image.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                angles.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
                positions.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
                sinogram.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                image.sizes()[0],
                image.sizes()[2],
                angles.sizes()[0],
                positions.sizes()[0]
            );
        })
    );
    return sinogram;
}