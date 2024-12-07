#pragma once

template <typename T>
__global__ void conv_kernel(
    T* result, 
    const T* input, 
    const T* filter, 
    int Cin, 
    int H, 
    int W, 
    int Cout, 
    int K
);

template <typename T>
void launch_conv2d(
    T* h_result, 
    T* h_x, 
    T* h_y, 
    int Cin, 
    int H, 
    int W, 
    int Cout, 
    int K
);