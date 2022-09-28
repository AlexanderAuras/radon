#include <torch/extension.h>
#include <omp.h>
#include <iostream>

#define LINE_OF_X(s,a,x) ((s)-(x)*cosf(a))/sinf(a)
#define LINE_OF_Y(s,a,y) ((s)-(y)*sinf(a))/cosf(a)
#define PI 3.14159265359f
#define PI_HALF 1.57079632679f
#define FLOAT_CMP_THRESHOLD 0.001f

#undef DEBUG_OUTPUT

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
torch::Tensor cpuForward(const torch::Tensor image_tensor, const torch::Tensor thetas_tensor, const torch::Tensor positions_tensor) {
    const torch::PackedTensorAccessor32<float,4> image = image_tensor.packed_accessor32<float,4>();
    const torch::PackedTensorAccessor32<float,1> thetas = thetas_tensor.packed_accessor32<float,1>();
    const torch::PackedTensorAccessor32<float,1> positions = positions_tensor.packed_accessor32<float,1>();
    const torch::Tensor sinogram_tensor = torch::zeros({image_tensor.sizes()[0], 1, thetas_tensor.sizes()[0], positions_tensor.sizes()[0]});
    torch::PackedTensorAccessor32<float,4> sinogram = sinogram_tensor.packed_accessor32<float,4>();

    const float M_half = image_tensor.sizes()[3]/2.0f;
    float tmp = 0.0f;
    const float grid_offset = modf(M_half, &tmp);
    #ifdef DEBUG_OUTPUT
    std::cout << "M_half: " << M_half << std::endl;
    std::cout << "grid_offset: " << grid_offset << std::endl;
    #endif
    //#pragma omp parallel for
    for(uint32_t batch_idx = 0; batch_idx < image_tensor.sizes()[0]; batch_idx++) {
        for(uint32_t theta_idx = 0; theta_idx < thetas_tensor.sizes()[0]; theta_idx++) {
            float theta0 = thetas[theta_idx];
            float theta = fmod(theta0,PI);
            float delta_t_x = abs(1.0f/sinf(theta));
            float delta_t_y = abs(1.0f/cosf(theta));
            std::cout << "theta: " << theta << std::endl;
            #ifdef DEBUG_OUTPUT
            std::cout << "delta_t_x: " << delta_t_x << std::endl;
            std::cout << "delta_t_y: " << delta_t_y << std::endl;
            #endif
            for(uint32_t position_idx = 0; position_idx < positions_tensor.sizes()[0]; position_idx++) {
                float pos = positions[position_idx];
                Vec2f left   = {-M_half, LINE_OF_X(pos, theta0, -M_half)};
                Vec2f right  = { M_half, LINE_OF_X(pos, theta0,  M_half)};
                Vec2f bottom = {LINE_OF_Y(pos, theta0, -M_half), -M_half};
                Vec2f top    = {LINE_OF_Y(pos, theta0,  M_half),  M_half};
                float t = 0.0f;
                float last_t_x = 0.0f;
                float last_t_y = 0.0f;
                Vec2i img_idx = {};

                #ifdef DEBUG_OUTPUT
                std::cout << "\tpos: " << pos << std::endl;
                std::cout << "\tleft: (" << left.x << ", " << left.y << ")" << std::endl;
                std::cout << "\tright: (" << right.x << ", " << right.y << ")" << std::endl;
                std::cout << "\tbottom: (" << bottom.x << ", " << bottom.y << ")" << std::endl;
                std::cout << "\ttop: (" << top.x << ", " << top.y << ")" << std::endl;
                #endif

                //Edge-cases for ϑ=0 and ϑ=π/2
                if(abs(theta) < FLOAT_CMP_THRESHOLD && -M_half <= pos && pos < M_half) {
                    #ifdef DEBUG_OUTPUT
                    std::cout << "\t\tEdge-Case ϑ=0" << std::endl;
                    #endif
                    for(uint32_t i = 0; i < image_tensor.sizes()[3]; i++) {
                        if(-M_half < pos) {
                            sinogram[batch_idx][0][theta_idx][position_idx] += 0.5f*image[batch_idx][0][i][static_cast<size_t>(floorf(pos+M_half-0.5f))];
                        }
                        sinogram[batch_idx][0][theta_idx][position_idx] += 0.5f*image[batch_idx][0][i][static_cast<size_t>(floorf(pos+M_half))];
                    }
                    continue;
                } else if(abs(theta-PI_HALF) < FLOAT_CMP_THRESHOLD && -M_half <= pos && pos < M_half) {
                    #ifdef DEBUG_OUTPUT
                    std::cout << "\t\tEdge-Case ϑ=π/2" << std::endl;
                    #endif
                    for(uint32_t i = 0; i < image_tensor.sizes()[3]; i++) {
                        if(-M_half < pos) {
                            sinogram[batch_idx][0][theta_idx][position_idx] += 0.5f*image[batch_idx][0][static_cast<size_t>(floorf(pos+M_half-0.5f))][i];
                        }
                        sinogram[batch_idx][0][theta_idx][position_idx] += 0.5f*image[batch_idx][0][static_cast<size_t>(floorf(pos+M_half))][i];
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
                    #ifdef DEBUG_OUTPUT
                    std::cout << "\t\tCase OUTSIDE" << std::endl;
                    #endif
                    continue;
                }

                #ifdef DEBUG_OUTPUT
                switch(curr_case) {
                    case Case::TOP_MINUS:    std::cout << "\t\tCase TOP_MINUS" << std::endl; break;
                    case Case::TOP_PLUS:     std::cout << "\t\tCase TOP_PLUS" << std::endl; break;
                    case Case::LEFT_MINUS:   std::cout << "\t\tCase LEFT_MINUS" << std::endl; break;
                    case Case::LEFT_PLUS:    std::cout << "\t\tCase LEFT_PLUS" << std::endl; break;
                    case Case::BOTTOM_MINUS: std::cout << "\t\tCase BOTTOM_MINUS" << std::endl; break;
                    case Case::BOTTOM_PLUS:  std::cout << "\t\tCase BOTTOM_PLUS" << std::endl; break;
                }
                #endif

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
                #ifdef DEBUG_OUTPUT
                std::cout << "\t\tlast_t_x: " << last_t_x << ", last_t_y: " << last_t_y << std::endl;
                #endif

                //Init img_idx
                switch(curr_case) {
                    case Case::TOP_MINUS:    img_idx = {static_cast<uint32_t>(floorf(top.x+M_half)), static_cast<uint32_t>(image_tensor.sizes()[3]-1)}; break;
                    case Case::TOP_PLUS:     img_idx = {static_cast<uint32_t>(floorf(top.x+M_half)), static_cast<uint32_t>(image_tensor.sizes()[3]-1)}; break;
                    case Case::LEFT_MINUS:   img_idx = {0, static_cast<uint32_t>(floorf(left.y+M_half))}; break;
                    case Case::LEFT_PLUS:    img_idx = {0, static_cast<uint32_t>(floorf(left.y+M_half))}; break;
                    case Case::BOTTOM_MINUS: img_idx = {static_cast<uint32_t>(floorf(bottom.x+M_half)), 0}; break;
                    case Case::BOTTOM_PLUS:  img_idx = {static_cast<uint32_t>(floorf(bottom.x+M_half)), 0}; break;
                }
                #ifdef DEBUG_OUTPUT
                std::cout << "\t\timg_idx: (" << img_idx.x << ", " << img_idx.y << ")" << std::endl;
                #endif

                //March ray
                while(img_idx.x >= 0 && img_idx.x < image_tensor.sizes()[3] && img_idx.y >= 0 && img_idx.y < image_tensor.sizes()[2]) {
                    //Diagonal crossing
                    if(abs(last_t_x+delta_t_x-last_t_y-delta_t_y) < FLOAT_CMP_THRESHOLD) {
                        last_t_x += delta_t_x;
                        last_t_y += delta_t_y;
                        sinogram[batch_idx][0][theta_idx][position_idx] += (last_t_x-t)*image[batch_idx][0][img_idx.y][img_idx.x];
                        #ifdef DEBUG_OUTPUT
                        std::cout << "\t\t\tD-STEP img_idx: (" << img_idx.x << ", " << img_idx.y << "), dt: " << (last_t_x-t) << ", t: " << last_t_x << ", last_t_x: " << last_t_x << ", last_t_y: " << last_t_y << std::endl;
                        #endif
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
                        #ifdef DEBUG_OUTPUT
                        std::cout << "\t\t\tH-STEP img_idx: (" << img_idx.x << ", " << img_idx.y << "), dt: " << (last_t_x-t) << ", t: " << last_t_x << ", last_t_x: " << last_t_x << ", last_t_y: " << last_t_y << std::endl;
                        #endif
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
                        #ifdef DEBUG_OUTPUT
                        std::cout << "\t\t\tV-STEP img_idx: (" << img_idx.x << ", " << img_idx.y << "), dt: " << (last_t_y-t) << ", t: " << last_t_y << ", last_t_x: " << last_t_x << ", last_t_y: " << last_t_y << std::endl;
                        #endif
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
    return sinogram_tensor;
}