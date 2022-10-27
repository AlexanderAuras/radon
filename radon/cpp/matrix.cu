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

template <typename T> __global__ void cudaMatrixKernel(
        const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> image,
        const torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> thetas,
        const torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> positions,
        torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> matrix,
        const size_t image_size,
        const size_t theta_count,
        const size_t position_count) {
    const size_t theta_idx    = static_cast<size_t>(blockIdx.y*blockDim.y+threadIdx.y);
    const size_t position_idx = static_cast<size_t>(blockIdx.x*blockDim.x+threadIdx.x);
    if(theta_idx >= theta_count || position_idx >= position_count) {
        return;
    }
    
    const float M_half      = image_size/2.0f;
    const float grid_offset = fmodf(M_half, 1.0f);
    const float theta0      = thetas[theta_idx];
    const float theta       = fmodf(theta0,PI);
    const float delta_t_x   = fabsf(1.0f/sinf(theta));
    const float delta_t_y   = fabsf(1.0f/cosf(theta));
            
    const float pos    = positions[position_idx];
    const Vec2f left   = {-M_half, LINE_OF_X(pos, theta0, -M_half)};
    const Vec2f right  = { M_half, LINE_OF_X(pos, theta0,  M_half)};
    const Vec2f bottom = {LINE_OF_Y(pos, theta0, -M_half), -M_half};
    const Vec2f top    = {LINE_OF_Y(pos, theta0,  M_half),  M_half};
    float t            = 0.0f;
    float last_t_x     = 0.0f;
    float last_t_y     = 0.0f;
    Vec2i img_idx      = {};

    //Edge-cases for ϑ=0 and ϑ=π/2
    if(fabsf(theta0) < FLOAT_CMP_THRESHOLD) {
        if(-M_half <= pos && pos < M_half) {
            for(uint32_t i = 0; i < image_size; i++) {
                if(-M_half < pos) {
                    matrix[theta_idx][position_idx][i][static_cast<size_t>(floorf(pos+M_half-0.5f))] += 0.5f;
                }
                matrix[theta_idx][position_idx][i][static_cast<size_t>(floorf(pos+M_half))] += 0.5f;
            }
        }
        return;
    } else if(fabsf(theta0 - PI_HALF) < FLOAT_CMP_THRESHOLD) {
        if(-M_half <= pos && pos < M_half) {
            for(uint32_t i = 0; i < image_size; i++) {
                if(-M_half < pos) {
                    matrix[theta_idx][position_idx][static_cast<size_t>(floorf(pos+M_half-0.5f))][i] += 0.5f;
                }
                matrix[theta_idx][position_idx][static_cast<size_t>(floorf(pos+M_half))][i] += 0.5f;
            }
        }
        return;
    } else if(fabsf(theta0 - PI) < FLOAT_CMP_THRESHOLD) {
        if(-M_half <= -pos && -pos < M_half) {
            for(uint32_t i = 0; i < image_size; i++) {
                if(-M_half < -pos) {
                    matrix[theta_idx][position_idx][i][static_cast<size_t>(floorf(-pos+M_half-0.5f))] += 0.5f;
                }
                matrix[theta_idx][position_idx][i][static_cast<size_t>(floorf(-pos+M_half))] += 0.5f;
            }
        }
        return;
    } else if(fabsf(theta0 - 3.0f*PI_HALF) < FLOAT_CMP_THRESHOLD) {
        if(-M_half <= -pos && -pos < M_half) {
            for(uint32_t i = 0; i < image_size; i++) {
                if(-M_half < -pos) {
                    matrix[theta_idx][position_idx][static_cast<size_t>(floorf(-pos+M_half-0.5f))][i] += 0.5f;
                }
                matrix[theta_idx][position_idx][static_cast<size_t>(floorf(-pos+M_half))][i] += 0.5f;
            }
        }
        return;
    }

    //Calculate case
    Case curr_case = Case::TOP_PLUS;
    if(-M_half <= top.x && top.x < M_half) {
        curr_case = theta<PI_HALF?Case::TOP_PLUS:Case::TOP_MINUS;
    } else if(-M_half <= left.y && left.y < M_half) {
        curr_case = theta>PI_HALF?Case::LEFT_PLUS:Case::LEFT_MINUS;
    } else if(-M_half <= bottom.x && bottom.x < M_half) {
        curr_case = theta>PI_HALF?Case::BOTTOM_PLUS:Case::BOTTOM_MINUS;
    } else {
        return;
    }

    //Init last_t_x, last_t_y
    // Check again
    switch(curr_case) {
        case Case::TOP_MINUS:    last_t_y = 0.0f; last_t_x = -(ceilf(top.x+grid_offset)-grid_offset-top.x)/sinf(theta); break;
        case Case::TOP_PLUS:     last_t_y = 0.0f; last_t_x =  (floorf(top.x+grid_offset)-grid_offset-top.x)/sinf(theta); break;
        case Case::LEFT_MINUS:   last_t_x = 0.0f; last_t_y = -(ceilf(left.y+grid_offset)-grid_offset-left.y)/cosf(theta); break;
        case Case::LEFT_PLUS:    last_t_x = 0.0f; last_t_y = -(floorf(left.y+grid_offset)-grid_offset-left.y)/cosf(theta); break;
        case Case::BOTTOM_MINUS: last_t_y = 0.0f; last_t_x = -(ceilf(bottom.x+grid_offset)-grid_offset-bottom.x)/sinf(theta); break;
        case Case::BOTTOM_PLUS:  last_t_y = 0.0f; last_t_x =  (floorf(bottom.x+grid_offset)-grid_offset-bottom.x)/sinf(theta); break;
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
            matrix[theta_idx][position_idx][img_idx.y][img_idx.x] += last_t_x-t;
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
            matrix[theta_idx][position_idx][img_idx.y][img_idx.x] += last_t_x-t;
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
            matrix[theta_idx][position_idx][img_idx.y][img_idx.x] += last_t_y-t;
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

torch::Tensor cudaMatrix(const torch::Tensor image, const torch::Tensor thetas, const torch::Tensor positions) {
    const dim3 threads(16, 16, 1);
    const dim3 blocks(
        ceil(positions.sizes()[0]/static_cast<float>(threads.x)), 
        ceil(thetas.sizes()[0]/static_cast<float>(threads.y)), 
        1
    );
    torch::Tensor matrix = torch::zeros({thetas.sizes()[0], positions.sizes()[0], image.sizes()[0], image.sizes()[0]}, c10::TensorOptions(torch::kCUDA));
    AT_DISPATCH_FLOATING_TYPES(image.scalar_type(), "radon_cudaMatrix", ([&] {
            cudaMatrixKernel<float><<<blocks, threads>>>(
                image.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                thetas.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
                positions.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
                matrix.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
                image.sizes()[0],
                thetas.sizes()[0],
                positions.sizes()[0]
            );
        })
    );
    return (matrix/image.sizes()[0]).reshape({-1, image.sizes()[0]*image.sizes()[0]});
}