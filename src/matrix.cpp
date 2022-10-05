#include <torch/extension.h>
#include <omp.h>

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

torch::Tensor cpuMatrix(const torch::Tensor image_tensor, const torch::Tensor thetas_tensor, const torch::Tensor positions_tensor) {
    const torch::PackedTensorAccessor32<float,2> image = image_tensor.packed_accessor32<float,2>();
    const torch::PackedTensorAccessor32<float,1> thetas = thetas_tensor.packed_accessor32<float,1>();
    const torch::PackedTensorAccessor32<float,1> positions = positions_tensor.packed_accessor32<float,1>();
    const torch::Tensor matrix = torch::zeros({thetas_tensor.sizes()[0], positions_tensor.sizes()[0], image_tensor.sizes()[0], image_tensor.sizes()[0]});
    torch::PackedTensorAccessor32<float,4> sinogram = matrix.packed_accessor32<float,4>();

    const float M_half      = image_tensor.sizes()[0]/2.0f;
    const float grid_offset = fmodf(M_half, 1.0f);
    #pragma omp parallel for collapse(2)
    for(uint32_t theta_idx = 0; theta_idx < thetas_tensor.sizes()[0]; theta_idx++) {
        for(uint32_t position_idx = 0; position_idx < positions_tensor.sizes()[0]; position_idx++) {
            const float theta0    = thetas[theta_idx];
            const float theta     = fmodf(theta0,PI);
            const float delta_t_x = fabsf(1.0f/sinf(theta));
            const float delta_t_y = fabsf(1.0f/cosf(theta));
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
                    for(uint32_t i = 0; i < image_tensor.sizes()[0]; i++) {
                        if(-M_half < pos) {
                            matrix[theta_idx][position_idx][i][static_cast<size_t>(floorf(pos+M_half-0.5f))] += 0.5f;
                        }
                        matrix[theta_idx][position_idx][i][static_cast<size_t>(floorf(pos+M_half))] += 0.5f;
                    }
                }
                continue;
            } else if(fabsf(theta0 - PI_HALF) < FLOAT_CMP_THRESHOLD) {
                if(-M_half <= pos && pos < M_half) {
                    for(uint32_t i = 0; i < image_tensor.sizes()[0]; i++) {
                        if(-M_half < pos) {
                            matrix[theta_idx][position_idx][static_cast<size_t>(floorf(pos+M_half-0.5f))][i] += 0.5f;
                        }
                        matrix[theta_idx][position_idx][static_cast<size_t>(floorf(pos+M_half))][i] += 0.5f;
                    }
                }
                continue;
            } else if(fabsf(theta0 - PI) < FLOAT_CMP_THRESHOLD) {
                if(-M_half <= -pos && -pos < M_half) {
                    for(uint32_t i = 0; i < image_tensor.sizes()[0]; i++) {
                        if(-M_half < -pos) {
                            matrix[theta_idx][position_idx][i][static_cast<size_t>(floorf(-pos+M_half-0.5f))] += 0.5f;
                        }
                        matrix[theta_idx][position_idx][i][static_cast<size_t>(floorf(-pos+M_half))] += 0.5f;
                    }
                }
                continue;
            } else if(fabsf(theta0 - 3.0f*PI_HALF) < FLOAT_CMP_THRESHOLD) {
                if(-M_half <= -pos && -pos < M_half) {
                    for(uint32_t i = 0; i < image_tensor.sizes()[0]; i++) {
                        if(-M_half < -pos) {
                            matrix[theta_idx][position_idx][static_cast<size_t>(floorf(-pos+M_half-0.5f))][i] += 0.5f;
                        }
                        matrix[theta_idx][position_idx][static_cast<size_t>(floorf(-pos+M_half))][i] += 0.5f;
                    }
                }
                continue;
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
                continue;
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
                case Case::TOP_MINUS:    img_idx = {static_cast<int32_t>(floorf(top.x+M_half)), static_cast<int32_t>(image_tensor.sizes()[0]-1)}; break;
                case Case::TOP_PLUS:     img_idx = {static_cast<int32_t>(floorf(top.x+M_half)), static_cast<int32_t>(image_tensor.sizes()[0]-1)}; break;
                case Case::LEFT_MINUS:   img_idx = {0, static_cast<int32_t>(floorf(left.y+M_half))}; break;
                case Case::LEFT_PLUS:    img_idx = {0, static_cast<int32_t>(floorf(left.y+M_half))}; break;
                case Case::BOTTOM_MINUS: img_idx = {static_cast<int32_t>(floorf(bottom.x+M_half)), 0}; break;
                case Case::BOTTOM_PLUS:  img_idx = {static_cast<int32_t>(floorf(bottom.x+M_half)), 0}; break;
            }
            img_idx.x = CLAMP(img_idx.x, 0, image_tensor.sizes()[0]-1);
            img_idx.y = CLAMP(img_idx.y, 0, image_tensor.sizes()[0]-1);

            //March ray
            while(img_idx.x >= 0 && img_idx.x < image_tensor.sizes()[1] && img_idx.y >= 0 && img_idx.y < image_tensor.sizes()[0]) {
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
    } 
    return (matrix/image_tensor.sizes()[0]).reshape({-1, image_tensor.sizes()[0]*image_tensor.sizes()[0]});
}