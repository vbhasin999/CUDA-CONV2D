#pragma once

template <typename T>
void launch_conv2d_basic(
    T* h_result, 
    const T* h_x, 
    const T* h_y, 
    int Cin, 
    int H, 
    int W, 
    int Cout, 
    int K
);

template <typename T>
void launch_conv2d_cout(
    T* h_result, 
    const T* h_x, 
    const T* h_y, 
    int Cin, 
    int H, 
    int W, 
    int Cout, 
    int K
);

template <typename T>
void launch_conv2d_opt(
    T* h_result, 
    const T* h_x, 
    const T* h_y, 
    int Cin, 
    int H, 
    int W, 
    int Cout, 
    int K
);

template <typename T>
void ref_conv(
    T* h_result, 
    const T* h_x, 
    const T* h_y, 
    int Cin, 
    int H, 
    int W, 
    int Cout, 
    int K
);