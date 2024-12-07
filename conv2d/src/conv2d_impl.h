#pragma once

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