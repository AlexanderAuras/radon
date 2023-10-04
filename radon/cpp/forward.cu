#include <torch/extension.h>

#define LINE_OF_X(s,a,x) ((s)-(x)*cosf(a))/sinf(a)
#define LINE_OF_Y(s,a,y) ((s)-(y)*sinf(a))/cosf(a)
#define PI 3.14159265359f
#define PI_HALF 1.57079632679f
#define FLOAT_CMP_THRESHOLD 0.001f
#define CLAMP(x, mi, ma) x<mi?mi:(x<ma?x:ma)

struct Vec2f {
    float x;
    float y;
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
    
    const float M_half      = image_size/2.0f;
    const float grid_offset = fmodf(M_half, 1.0f);
    const float theta0      = thetas[theta_idx];
    const float theta       = fmodf(theta0,PI);
    const float delta_t_x   = fabsf(1.0f/sinf(theta));
    const float delta_t_y   = fabsf(1.0f/cosf(theta));
            
    const float pos    = positions[position_idx];
    Vec2f left   = {-M_half, LINE_OF_X(pos, theta0, -M_half)};
    Vec2f right  = { M_half, LINE_OF_X(pos, theta0,  M_half)};
    Vec2f bottom = {LINE_OF_Y(pos, theta0, -M_half), -M_half};
    Vec2f top    = {LINE_OF_Y(pos, theta0,  M_half),  M_half};
    float t            = 0.0f;
    float last_t_x     = 0.0f;
    float last_t_y     = 0.0f;
    Vec2i img_idx      = {};

    if(fabsf(theta0) < FLOAT_CMP_THRESHOLD || fabsf(theta0 - PI) < FLOAT_CMP_THRESHOLD) {
        left.y  = std::numeric_limits<float>::infinity();
        right.y = std::numeric_limits<float>::infinity();
    } else if(fabsf(theta0 - PI_HALF) < FLOAT_CMP_THRESHOLD || fabsf(theta0 - 3.0f*PI_HALF) < FLOAT_CMP_THRESHOLD) {
        bottom.x = std::numeric_limits<float>::infinity();
        top.x    = std::numeric_limits<float>::infinity();
    }

    //Edge-cases for ϑ=0 and ϑ=π/2
    /*if(fabsf(theta0) < FLOAT_CMP_THRESHOLD) {
        if(-M_half <= pos && pos < M_half) {
            for(uint32_t i = 0; i < image_size; i++) {
                if(-M_half+0.5f < pos) {
                    sinogram[batch_idx][0][theta_idx][position_idx] += 0.5f*image[batch_idx][0][i][static_cast<size_t>(floorf(pos+M_half-0.5f))];
                }
                sinogram[batch_idx][0][theta_idx][position_idx] += 0.5f*image[batch_idx][0][i][static_cast<size_t>(floorf(pos+M_half))];
            }
        }
        return;
    } else if(fabsf(theta0 - PI_HALF) < FLOAT_CMP_THRESHOLD) {
        if(-M_half <= pos && pos < M_half) {
            for(uint32_t i = 0; i < image_size; i++) {
                if(-M_half+0.5f < pos) {
                    sinogram[batch_idx][0][theta_idx][position_idx] += 0.5f*image[batch_idx][0][static_cast<size_t>(floorf(pos+M_half-0.5f))][i];
                }
                sinogram[batch_idx][0][theta_idx][position_idx] += 0.5f*image[batch_idx][0][static_cast<size_t>(floorf(pos+M_half))][i];
            }
        }
        return;
    } else if(fabsf(theta0 - PI) < FLOAT_CMP_THRESHOLD) {
        if(-M_half <= -pos && -pos < M_half) {
            for(uint32_t i = 0; i < image_size; i++) {
                if(-M_half+0.5f < -pos) {
                    sinogram[batch_idx][0][theta_idx][position_idx] += 0.5f*image[batch_idx][0][i][static_cast<size_t>(floorf(-pos+M_half-0.5f))];
                }
                sinogram[batch_idx][0][theta_idx][position_idx] += 0.5f*image[batch_idx][0][i][static_cast<size_t>(floorf(-pos+M_half))];
            }
        }
        return;
    } else if(fabsf(theta0 - 3.0f*PI_HALF) < FLOAT_CMP_THRESHOLD) {
        if(-M_half <= -pos && -pos < M_half) {
            for(uint32_t i = 0; i < image_size; i++) {
                if(-M_half+0.5f < -pos) {
                    sinogram[batch_idx][0][theta_idx][position_idx] += 0.5f*image[batch_idx][0][static_cast<size_t>(floorf(-pos+M_half-0.5f))][i];
                }
                sinogram[batch_idx][0][theta_idx][position_idx] += 0.5f*image[batch_idx][0][static_cast<size_t>(floorf(-pos+M_half))][i];
            }
        }
        return;
    }*/

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
        case Case::TOP_MINUS:    last_t_y = 0.0f; last_t_x = -(ceilf(top.x+grid_offset)-grid_offset-top.x)/sinf(theta); break;
        case Case::TOP_PLUS:     last_t_y = 0.0f; last_t_x =  (floorf(top.x+grid_offset)-grid_offset-top.x)/sinf(theta); break;
        case Case::LEFT_MINUS:   last_t_x = 0.0f; last_t_y = -(ceilf(left.y+grid_offset)-grid_offset-left.y)/cosf(theta); break;
        case Case::LEFT_PLUS:    last_t_x = 0.0f; last_t_y = -(floorf(left.y+grid_offset)-grid_offset-left.y)/cosf(theta); break;
        case Case::BOTTOM_MINUS: last_t_y = 0.0f; last_t_x = -(ceilf(bottom.x+grid_offset)-grid_offset-bottom.x)/sinf(theta); break;
        case Case::BOTTOM_PLUS:  last_t_y = 0.0f; last_t_x =  (floorf(bottom.x+grid_offset)-grid_offset-bottom.x)/sinf(theta); break;
    }
    if(fabsf(theta0) < FLOAT_CMP_THRESHOLD || fabsf(theta0 - PI) < FLOAT_CMP_THRESHOLD) {
        last_t_x = std::numeric_limits<float>::infinity();
    } else if(fabsf(theta0 - PI_HALF) < FLOAT_CMP_THRESHOLD || fabsf(theta0 - 3.0f*PI_HALF) < FLOAT_CMP_THRESHOLD) {
        last_t_y = std::numeric_limits<float>::infinity();
    }

    //Init img_idx
    switch(curr_case) {
        case Case::TOP_MINUS:    img_idx = {static_cast<int32_t>(floorf(top.x+M_half)), static_cast<int32_t>(image_size-1)}; break;
        case Case::TOP_PLUS:     img_idx = {static_cast<int32_t>(floorf(top.x+M_half)), static_cast<int32_t>(image_size-1)}; break;
        case Case::LEFT_MINUS:   img_idx = {0, static_cast<int32_t>(floorf(left.y+M_half))}; break;
        case Case::LEFT_PLUS:    img_idx = {0, static_cast<int32_t>(floorf(left.y+M_half))}; break;
        case Case::BOTTOM_MINUS: img_idx = {static_cast<int32_t>(floorf(bottom.x+M_half)), 0}; break;
        case Case::BOTTOM_PLUS:  img_idx = {static_cast<int32_t>(floorf(bottom.x+M_half)), 0}; break;
    }
    img_idx.x = CLAMP(img_idx.x, 0, image_size-1);
    img_idx.y = CLAMP(img_idx.y, 0, image_size-1);

    //March ray
    while(img_idx.x >= 0 && img_idx.x < image_size && img_idx.y >= 0 && img_idx.y < image_size) {
        //Diagonal crossing
        if(fabsf(last_t_x+delta_t_x-last_t_y-delta_t_y) < FLOAT_CMP_THRESHOLD) {
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
                image.sizes()[2],
                thetas.sizes()[0],
                positions.sizes()[0]
            );
        })
    );
    return sinogram/(image.sizes()[3]*1.41421356237f);
}