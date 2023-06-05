#include <torch/extension.h>

#include "dyn_type_math.hpp"

#define LINE_OF_X(T,s,a,x) ((s)-(x)*DynTypeMath<T>::cos(a))/DynTypeMath<T>::sin(a)
#define LINE_OF_Y(T,s,a,y) ((s)-(y)*DynTypeMath<T>::sin(a))/DynTypeMath<T>::cos(a)
#define PI 3.14159265359
#define PI_HALF 1.57079632679
#define FLOAT_CMP_THRESHOLD 0.001
#define CLAMP(x, mi, ma) x<mi?mi:(x<ma?x:ma)

template<typename T> struct Vec2f {
    T x;
    T y;
};

struct Vec2i {
    int32_t x;
    int32_t y;
};

enum class Case {
    TOP_PLUS, TOP_MINUS,
    LEFT_PLUS, LEFT_MINUS,
    BOTTOM_PLUS, BOTTOM_MINUS
};

template <typename T> __global__ void cudaForwardKernel(
        const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> image,
        const torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> thetas,
        const torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> positions,
        torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> sinogram,
        const size_t batch_count,
        const size_t image_size,
        const size_t theta_count,
        const size_t position_count) {
    const size_t batch_idx    = static_cast<size_t>(blockIdx.z*blockDim.z+threadIdx.z);
    const size_t theta_idx    = static_cast<size_t>(blockIdx.y*blockDim.y+threadIdx.y);
    const size_t position_idx = static_cast<size_t>(blockIdx.x*blockDim.x+threadIdx.x);
    if(batch_idx >= batch_count || theta_idx >= theta_count || position_idx >= position_count) {
        return;
    }
    
    const T M_half      = image_size/2.0;
    const T grid_offset = DynTypeMath<T>::mod(M_half, 1.0);
    const T theta0      = thetas[theta_idx];
    const T theta       = DynTypeMath<T>::mod(theta0,PI);
    const T delta_t_x   = DynTypeMath<T>::abs(1.0/DynTypeMath<T>::sin(theta));
    const T delta_t_y   = DynTypeMath<T>::abs(1.0/DynTypeMath<T>::cos(theta));
            
    const T pos    = positions[position_idx];
    Vec2f<T> left   = {-M_half, LINE_OF_X(T, pos, theta0, -M_half)};
    Vec2f<T> right  = { M_half, LINE_OF_X(T, pos, theta0,  M_half)};
    Vec2f<T> bottom = {LINE_OF_Y(T, pos, theta0, -M_half), -M_half};
    Vec2f<T> top    = {LINE_OF_Y(T, pos, theta0,  M_half),  M_half};
    T t            = 0.0;
    T last_t_x     = 0.0;
    T last_t_y     = 0.0;
    Vec2i img_idx      = {};

    if(DynTypeMath<T>::abs(theta0) < FLOAT_CMP_THRESHOLD || DynTypeMath<T>::abs(theta0 - PI) < FLOAT_CMP_THRESHOLD) {
        left.y  = std::numeric_limits<T>::infinity();
        right.y = std::numeric_limits<T>::infinity();
    } else if(DynTypeMath<T>::abs(theta0 - PI_HALF) < FLOAT_CMP_THRESHOLD || DynTypeMath<T>::abs(theta0 - 3.0*PI_HALF) < FLOAT_CMP_THRESHOLD) {
        bottom.x = std::numeric_limits<T>::infinity();
        top.x    = std::numeric_limits<T>::infinity();
    }

    //Calculate case
    Case curr_case = Case::TOP_PLUS;
    if(-M_half-FLOAT_CMP_THRESHOLD <= top.x && top.x < M_half+FLOAT_CMP_THRESHOLD) {
        curr_case = theta<PI_HALF?Case::TOP_PLUS:Case::TOP_MINUS;
        top.x = CLAMP(top.x, -M_half, M_half);
    } else if(-M_half-FLOAT_CMP_THRESHOLD <= left.y && left.y < M_half+FLOAT_CMP_THRESHOLD) {
        curr_case = theta>PI_HALF?Case::LEFT_PLUS:Case::LEFT_MINUS;
        left.y = CLAMP(left.y, -M_half, M_half);
    } else if(-M_half-FLOAT_CMP_THRESHOLD <= bottom.x && bottom.x < M_half+FLOAT_CMP_THRESHOLD) {
        curr_case = theta>PI_HALF?Case::BOTTOM_PLUS:Case::BOTTOM_MINUS;
        bottom.x = CLAMP(bottom.x, -M_half, M_half);
    } else {
        return;
    }

    //Init last_t_x, last_t_y
    switch(curr_case) {
        case Case::TOP_MINUS:    last_t_y = 0.0; last_t_x = -(DynTypeMath<T>::ceil(top.x+grid_offset)-grid_offset-top.x)/DynTypeMath<T>::sin(theta); break;
        case Case::TOP_PLUS:     last_t_y = 0.0; last_t_x =  (DynTypeMath<T>::floor(top.x+grid_offset)-grid_offset-top.x)/DynTypeMath<T>::sin(theta); break;
        case Case::LEFT_MINUS:   last_t_x = 0.0; last_t_y = -(DynTypeMath<T>::ceil(left.y+grid_offset)-grid_offset-left.y)/DynTypeMath<T>::cos(theta); break;
        case Case::LEFT_PLUS:    last_t_x = 0.0; last_t_y = -(DynTypeMath<T>::floor(left.y+grid_offset)-grid_offset-left.y)/DynTypeMath<T>::cos(theta); break;
        case Case::BOTTOM_MINUS: last_t_y = 0.0; last_t_x = -(DynTypeMath<T>::ceil(bottom.x+grid_offset)-grid_offset-bottom.x)/DynTypeMath<T>::sin(theta); break;
        case Case::BOTTOM_PLUS:  last_t_y = 0.0; last_t_x =  (DynTypeMath<T>::floor(bottom.x+grid_offset)-grid_offset-bottom.x)/DynTypeMath<T>::sin(theta); break;
    }
    if(DynTypeMath<T>::abs(theta0) < FLOAT_CMP_THRESHOLD || DynTypeMath<T>::abs(theta0 - PI) < FLOAT_CMP_THRESHOLD) {
        last_t_x = std::numeric_limits<T>::infinity();
    } else if(DynTypeMath<T>::abs(theta0 - PI_HALF) < FLOAT_CMP_THRESHOLD || DynTypeMath<T>::abs(theta0 - 3.0*PI_HALF) < FLOAT_CMP_THRESHOLD) {
        last_t_y = std::numeric_limits<T>::infinity();
    }

    //Init img_idx
    switch(curr_case) {
        case Case::TOP_MINUS:    img_idx = {static_cast<int32_t>(DynTypeMath<T>::floor(top.x+M_half)), static_cast<int32_t>(image_size-1)}; break;
        case Case::TOP_PLUS:     img_idx = {static_cast<int32_t>(DynTypeMath<T>::floor(top.x+M_half)), static_cast<int32_t>(image_size-1)}; break;
        case Case::LEFT_MINUS:   img_idx = {0, static_cast<int32_t>(DynTypeMath<T>::floor(left.y+M_half))}; break;
        case Case::LEFT_PLUS:    img_idx = {0, static_cast<int32_t>(DynTypeMath<T>::floor(left.y+M_half))}; break;
        case Case::BOTTOM_MINUS: img_idx = {static_cast<int32_t>(DynTypeMath<T>::floor(bottom.x+M_half)), 0}; break;
        case Case::BOTTOM_PLUS:  img_idx = {static_cast<int32_t>(DynTypeMath<T>::floor(bottom.x+M_half)), 0}; break;
    }
    img_idx.x = CLAMP(img_idx.x, 0, image_size-1);
    img_idx.y = CLAMP(img_idx.y, 0, image_size-1);

    //March ray
    while(img_idx.x >= 0 && img_idx.x < image_size && img_idx.y >= 0 && img_idx.y < image_size) {
        //Diagonal crossing
        if(DynTypeMath<T>::abs(last_t_x+delta_t_x-last_t_y-delta_t_y) < FLOAT_CMP_THRESHOLD) {
            last_t_x += delta_t_x;
            last_t_y += delta_t_y;
            sinogram[batch_idx][0][theta_idx][position_idx] += (last_t_x-t)*image[batch_idx][0][img_idx.y][img_idx.x];
            //Modify img_idx
            switch(curr_case) {
                case Case::TOP_MINUS:    img_idx.x--; img_idx.y--; break;
                case Case::TOP_PLUS:     img_idx.x++; img_idx.y--; break;
                case Case::LEFT_MINUS:   img_idx.x++; img_idx.y--; break;
                case Case::LEFT_PLUS:    img_idx.x++; img_idx.y++; break;
                case Case::BOTTOM_MINUS: img_idx.x--; img_idx.y++; break;
                case Case::BOTTOM_PLUS:  img_idx.x++; img_idx.y++; break;
            }
            t = last_t_x;
        } else if(last_t_x+delta_t_x < last_t_y+delta_t_y) { //Horizontal crossing
            last_t_x += delta_t_x;
            sinogram[batch_idx][0][theta_idx][position_idx] += (last_t_x-t)*image[batch_idx][0][img_idx.y][img_idx.x];
            //Modify img_idx
            switch(curr_case) {
                case Case::TOP_MINUS:    img_idx.x--; break;
                case Case::TOP_PLUS:     img_idx.x++; break;
                case Case::LEFT_MINUS:   img_idx.x++; break;
                case Case::LEFT_PLUS:    img_idx.x++; break;
                case Case::BOTTOM_MINUS: img_idx.x--; break;
                case Case::BOTTOM_PLUS:  img_idx.x++; break;
            }
            t = last_t_x;
        } else { //Vertical crossing
            last_t_y += delta_t_y;
            sinogram[batch_idx][0][theta_idx][position_idx] += (last_t_y-t)*image[batch_idx][0][img_idx.y][img_idx.x];
            //Modify img_idx
            switch(curr_case) {
                case Case::TOP_MINUS:    img_idx.y--; break;
                case Case::TOP_PLUS:     img_idx.y--; break;
                case Case::LEFT_MINUS:   img_idx.y--; break;
                case Case::LEFT_PLUS:    img_idx.y++; break;
                case Case::BOTTOM_MINUS: img_idx.y++; break;
                case Case::BOTTOM_PLUS:  img_idx.y++; break;
            }
            t = last_t_y;
        }
    }
}

torch::Tensor cudaForward(const torch::Tensor image, const torch::Tensor thetas, const torch::Tensor positions) {
    const dim3 threads(8, 8, 2);
    const dim3 blocks(
        ceil(positions.sizes()[0]/static_cast<float>(threads.x)), 
        ceil(thetas.sizes()[0]/static_cast<float>(threads.y)), 
        ceil(image.sizes()[0]/static_cast<float>(threads.z))
    );
    torch::Tensor sinogram = torch::zeros({image.sizes()[0], 1, thetas.sizes()[0], positions.sizes()[0]}, c10::TensorOptions(torch::kCUDA));
    AT_DISPATCH_FLOATING_TYPES(image.scalar_type(), "radon_cudaForward", ([&] {
            cudaForwardKernel<float><<<blocks, threads>>>(
                image.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
                thetas.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
                positions.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
                sinogram.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
                image.sizes()[0],
                image.sizes()[3],
                thetas.sizes()[0],
                positions.sizes()[0]
            );
        })
    );
    return sinogram/(image.sizes()[3]*1.41421356237);
    //return sinogram / static_cast<float>(positions.sizes()[0]);
}