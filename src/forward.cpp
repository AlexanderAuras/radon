#include <torch/extension.h>
#include <omp.h>
#include <iostream>

#define LINE_OF_X(s,a,x) ((s)-(x)*cosf(a))/sinf(a)
#define LINE_OF_Y(s,a,y) ((s)-(y)*sinf(a))/cosf(a)
#define PI 3.14159265359f
#define PI_HALF 1.57079632679f
#define FLOAT_CMP_THRESHOLD 0.001f

struct Vec2f {
    float x;
    float y;
};

struct Vec2i {
    uint32_t x;
    uint32_t y;
};

enum class Case {
    TOP_PLUS, TOP_MINUS,
    LEFT_PLUS, LEFT_MINUS,
    BOTTOM_PLUS, BOTTOM_MINUS
};

/******************
 * MIGHT BE TRASH *
 ******************/
torch::Tensor cpuForward(const torch::Tensor _image, const torch::Tensor _thetas, const torch::Tensor _positions) {
    torch::PackedTensorAccessor32<float,4> image = _image.packed_accessor32<float,4>();
    torch::PackedTensorAccessor32<float,1> thetas = _thetas.packed_accessor32<float,1>();
    torch::PackedTensorAccessor32<float,1> positions = _positions.packed_accessor32<float,1>();
    torch::Tensor _sinogram = torch::zeros({_image.sizes()[0], 1, _thetas.sizes()[0], _positions.sizes()[0]});
    torch::PackedTensorAccessor32<float,4> sinogram = _sinogram.packed_accessor32<float,4>();

    float M_half = _image.sizes()[3]/2.0f;
    float tmp = 0.0f;
    float grid_offset = modf(M_half, &tmp);
    for(uint32_t batch_idx = 0; batch_idx < _image.sizes()[0]; batch_idx++) {
        for(uint32_t theta_idx = 0; theta_idx < _thetas.sizes()[0]; theta_idx++) {
            float delta_t_x = abs(1.0f/sinf(thetas[theta_idx]));
            float delta_t_y = abs(1.0f/cosf(thetas[theta_idx]));
            float theta = thetas[theta_idx];
            for(uint32_t position_idx = 0; position_idx < _positions.sizes()[0]; position_idx++) {
                float pos = positions[position_idx];
                Vec2f left   = {-M_half, LINE_OF_X(pos, theta, -M_half)};
                Vec2f right  = { M_half, LINE_OF_X(pos, theta,  M_half)};
                Vec2f bottom = {LINE_OF_Y(pos, theta, -M_half), -M_half};
                Vec2f top    = {LINE_OF_Y(pos, theta,  M_half),  M_half};
                float t = 0.0f;
                float last_t_x = 0.0f;
                float last_t_y = 0.0f;
                Vec2i img_idx = {};

                //Edge-cases for ϑ=0 and ϑ=π/2
                if(abs(theta) < FLOAT_CMP_THRESHOLD && -M_half <= pos && pos < M_half) {
                    for(uint32_t i = 0; i < _image.sizes()[3]; i++) {
                        sinogram[batch_idx][0][theta_idx][position_idx] += image[batch_idx][0][i][static_cast<size_t>(floorf(pos+M_half))];
                    }
                } else if(abs(theta-PI_HALF) < FLOAT_CMP_THRESHOLD && -M_half <= pos && pos < M_half) {
                    for(uint32_t i = 0; i < _image.sizes()[3]; i++) {
                        sinogram[batch_idx][0][theta_idx][position_idx] += image[batch_idx][0][static_cast<size_t>(floorf(pos+M_half))][i];
                    }
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
                switch(curr_case) {
                    case Case::TOP_MINUS:    last_t_y = 0.0f; last_t_x = -(ceilf(top.x+grid_offset)-grid_offset-top.x)/sinf(theta); break;
                    case Case::TOP_PLUS:     last_t_y = 0.0f; last_t_x =  (floorf(top.x+grid_offset)-grid_offset-top.x)/sinf(theta); break;
                    case Case::LEFT_MINUS:   last_t_x = 0.0f; last_t_y = -(ceilf(left.y+grid_offset)-grid_offset-left.y)/sinf(theta); break;
                    case Case::LEFT_PLUS:    last_t_x = 0.0f; last_t_y =  (floorf(left.y+grid_offset)-grid_offset-left.y)/sinf(theta); break;
                    case Case::BOTTOM_MINUS: last_t_y = 0.0f; last_t_x = -(ceilf(bottom.x+grid_offset)-grid_offset-bottom.x)/sinf(theta); break;
                    case Case::BOTTOM_PLUS:  last_t_y = 0.0f; last_t_x =  (floorf(bottom.x+grid_offset)-grid_offset-bottom.x)/sinf(theta); break;
                }

                //Init img_idx
                switch(curr_case) {
                    case Case::TOP_MINUS:    img_idx = {static_cast<uint32_t>(floorf(top.x+M_half)), static_cast<uint32_t>(_image.sizes()[3]-1)}; break;
                    case Case::TOP_PLUS:     img_idx = {static_cast<uint32_t>(floorf(top.x+M_half)), static_cast<uint32_t>(_image.sizes()[3]-1)}; break;
                    case Case::LEFT_MINUS:   img_idx = {0, static_cast<uint32_t>(floorf(left.y+M_half))}; break;
                    case Case::LEFT_PLUS:    img_idx = {0, static_cast<uint32_t>(floorf(left.y+M_half))}; break;
                    case Case::BOTTOM_MINUS: img_idx = {static_cast<uint32_t>(floorf(bottom.x+M_half)), 0}; break;
                    case Case::BOTTOM_PLUS:  img_idx = {static_cast<uint32_t>(floorf(bottom.x+M_half)), 0}; break;
                }

                //March ray
                while(img_idx.x >= 0 && img_idx.x < _image.sizes()[3] && img_idx.y >= 0 && img_idx.y < _image.sizes()[2]) {
                    //Diagonal crossing
                    if(abs(last_t_x+delta_t_x-last_t_y-delta_t_y) < FLOAT_CMP_THRESHOLD) {
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
        } 
    }
    return _sinogram;
}