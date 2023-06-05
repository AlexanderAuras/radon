#include <cstdlib>
#include <cstdio>

#include <torch/extension.h>

#include "dyn_type_math.hpp"

#define PI 3.14159265359
#define CLAMP(x, mi, ma) x<mi?mi:(x<ma?x:ma)
#define FLOAT_CMP_THRESHOLD 0.001

template <typename T> __global__ void cudaBackwardKernel(
        const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> sinogram,
        const torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> thetas,
        const torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> positions,
        torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> image,
        const size_t batch_count,
        const size_t image_size,
        const size_t theta_count,
        const size_t position_count,
        const double min_pos,
        const double max_pos) {
    const size_t batch_idx = static_cast<size_t>(blockIdx.z*blockDim.z+threadIdx.z);
    const size_t y_idx     = static_cast<size_t>(blockIdx.y*blockDim.y+threadIdx.y);
    const size_t x_idx     = static_cast<size_t>(blockIdx.x*blockDim.x+threadIdx.x);
    if(batch_idx >= batch_count || y_idx >= image_size || x_idx >= image_size) {
        return;
    }

    const T M_half = image_size/2.0;

    for(int32_t theta_idx = 0; theta_idx < theta_count; theta_idx++) {
        const T theta0 = thetas[theta_idx];
        const T theta  = fmodf(theta0,PI);

        const T exact_position = DynTypeMath<T>::cos(theta)*(x_idx+0.5-M_half)+DynTypeMath<T>::sin(theta)*(y_idx+0.5-M_half);
        int32_t position_min_idx = 0;
        int32_t position_max_idx = 0;
        T position_min = -std::numeric_limits<T>::infinity();
        T position_max =  std::numeric_limits<T>::infinity();
        for(int32_t i = 0; i < position_count; i++) {
            if((positions[i] < exact_position || DynTypeMath<T>::abs(positions[i]-exact_position) <= FLOAT_CMP_THRESHOLD) && positions[i] > position_min) {
                position_min = positions[i];
                position_min_idx = i;
            }
            if((positions[i] > exact_position || DynTypeMath<T>::abs(positions[i]-exact_position) <= FLOAT_CMP_THRESHOLD) && positions[i] < position_max) {
                position_max = positions[i];
                position_max_idx = i;
            }
        }
        position_min_idx = CLAMP(position_min_idx, 0, position_count-1);
        position_max_idx = CLAMP(position_max_idx, 0, position_count-1);
        position_min = CLAMP(position_min, min_pos, max_pos);
        position_max = CLAMP(position_max, min_pos, max_pos);

        if(position_min_idx == position_max_idx) {
            //printf("Min=Max    %f: %f (%u),    %f (%u)\n", exact_position, position_min, position_min_idx, position_max, position_max_idx);
            atomicAdd(&image[batch_idx][0][y_idx][x_idx], sinogram[batch_idx][0][theta_idx][position_min_idx]);
            //atomicAdd(&image[batch_idx][0][y_idx][x_idx], 1);
        } else {
            T fraction = (exact_position-position_min)/(position_max-position_min);
            //printf("Min!=Max (%f)    %f: %f (%u),    %f (%u)\n", fraction, exact_position, position_min, position_min_idx, position_max, position_max_idx);
            const T value = fraction*sinogram[batch_idx][0][theta_idx][position_min_idx]+(1.0-fraction)*sinogram[batch_idx][0][theta_idx][position_max_idx];
            atomicAdd(&image[batch_idx][0][y_idx][x_idx], value);
            //atomicAdd(&image[batch_idx][0][y_idx][x_idx], fraction);
        }
    }
}

torch::Tensor cudaBackward(const torch::Tensor sinogram, const torch::Tensor thetas, const torch::Tensor positions, const size_t image_size) {
    const dim3 threads(8, 8, 2);
    const dim3 blocks(
        ceil(image_size/static_cast<float>(threads.x)), 
        ceil(image_size/static_cast<float>(threads.y)), 
        ceil(sinogram.sizes()[0]/static_cast<float>(threads.z))
    );
    torch::Tensor image = torch::zeros({sinogram.sizes()[0], 1, static_cast<signed long>(image_size), static_cast<signed long>(image_size)}, torch::TensorOptions(torch::kCUDA));
    AT_DISPATCH_FLOATING_TYPES(sinogram.scalar_type(), "radon_cudaBackward", ([&] {
            cudaBackwardKernel<scalar_t><<<blocks, threads>>>(
                sinogram.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                thetas.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
                positions.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
                image.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                sinogram.sizes()[0],
                image_size,
                thetas.sizes()[0],
                positions.sizes()[0],
                positions.min().item().toDouble(),
                positions.max().item().toDouble()
            );
        })
    );
    return image;
}