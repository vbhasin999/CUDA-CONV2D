#include "conv2d_impl.h"

// Add batch, stride and padding later
// we could have 1 block <-> 1 output channel
// and 1 thread <-> 1 output pixel
template <typename T>
__global__ void conv_kernel_basic(T *result, const T *input, const T *filter, int Cin, int H, int W, int Cout, int K) {
    int x = threadIdx.x;
    int y = threadIdx.y;
    int z = blockIdx.x;

    int H_out = H - K + 1;
    int W_out = W - K + 1;

    T sum = 0;
    if (x < H_out && y < W_out){
        // loop over input channels
        for (int i = 0; i < Cin; i++) {
            // loop over filter dimensions
            for (int j = 0; j < K; j++) {
                for (int k = 0; k < K; k++) {
                    // input of shape (Cin, H, W) with strides (H*W, W, 1)
                    // filter of shape (Cout, Cin, K, K) with strides (Cin*K*K, K*K, K, 1)
                    sum += input[i * H * W + (x + j) * W + (y + k)] * filter[z * Cin * K * K + i * K * K + j * K + k];
                }
            }
        }
        // result of shape (Cout, H_out, W_out) with strides (H_out * W_out, W_out, 1)
        result[z * H_out * W_out + x * W_out + y] = sum;
    }
}

template <typename T>
void launch_conv2d_basic(T *h_result, T *h_x, T *h_y, 
                   int Cin, int H, int W, int Cout, int K) {
    // Output dimensions
    int H_out = H - K + 1;
    int W_out = W - K + 1;

    // Device memory allocation
    T *d_x, *d_y, *d_result;
    size_t input_size = Cin * H * W * sizeof(T);
    size_t filter_size = Cout * Cin * K * K * sizeof(T);
    size_t output_size = Cout * H_out * W_out * sizeof(T);

    cudaMalloc(&d_x, input_size);
    cudaMalloc(&d_y, filter_size);
    cudaMalloc(&d_result, output_size);

    // Copy inputs to device
    cudaMemcpy(d_x, h_x, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, filter_size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 threadsPerBlock(W_out, H_out); // one thread per output pixel
    dim3 numBlocks(Cout);          // One block per output channel and input channel

    // Launch kernel
    conv_kernel_basic<<<numBlocks, threadsPerBlock>>>(d_result, d_x, d_y, Cin, H, W, Cout, K);

    // Copy results back to host
    cudaMemcpy(h_result, d_result, output_size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_result);
}

// Supports batching
// Z dim for batch
// X dim for channel
// Y dim for height, width
template <typename T>
__global__ void conv_kernel_batched(T *result, const T *input, const T *filter, int N, int Cin, int H, int W, int Cout, int Kh, int Kw) {
    int x = blockIdx.x; // output channel
    int y = threadIdx.y; // output pixel
    int z = threadIdx.z; // batch

    int H_out = H - Kh + 1;
    int W_out = W - Kw + 1;

    int t_r = y / W_out;
    int t_c = y % W_out;

    T sum = 0;

    if (x < Cout && y < H_out * W_out && z < N){
        for (int i = 0; i < Cin; ++i ){
            for (int j = 0; j < Kh; ++j){
                for (int k = 0; k < Kw; ++k){
                    
                    sum += input[(z * Cin * H * W) + (i * H * W) + ((t_r + j) * W) + (k + t_c) ] * filter[(x * Cin * Kh * Kw) + (i * Kh * Kw) + (j * Kw) + k ];
    
                }
            }
        }

        // result of shape N, Cout, Hout, Wout
        result[(z * Cout * H_out * W_out) + (x * H_out * W_out) + (t_r * W_out) + t_c] = sum;

    }
    
    
}

template <typename T>
void launch_conv2d_batched(T *h_result, T *h_x, T *h_y, int N, int Cin, int H, int W, int Cout, int Kh, int Kw) {
    // Output dimensions
    int H_out = H - Kh + 1;
    int W_out = W - Kw + 1;

    // Device memory allocation
    T *d_x, *d_y, *d_result;
    size_t input_size = N * Cin * H * W * sizeof(T);
    size_t filter_size = Cout * Cin * Kh * Kw * sizeof(T);
    size_t output_size = N * Cout * H_out * W_out * sizeof(T);

    cudaMalloc(&d_x, input_size);
    cudaMalloc(&d_y, filter_size);
    cudaMalloc(&d_result, output_size);

    // Copy inputs to device
    cudaMemcpy(d_x, h_x, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, filter_size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 threadsPerBlock(1, W_out * H_out, N); // one thread per output pixel
    dim3 numBlocks(Cout);          // One block per output channel and input channel

    // Launch kernel
    conv_kernel_batched<<<numBlocks, threadsPerBlock>>>(d_result, d_x, d_y, N, Cin, H, W, Cout, Kh, Kw);

    // Copy results back to host
    cudaMemcpy(h_result, d_result, output_size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_result);
}

template <typename T>
void ref_conv(T *result, const T *input, const T *filter, int N, int Cin, int H, int W, int Cout, int Kh, int Kw){
    int H_out = H - Kh + 1;
    int W_out = W - Kw + 1;

    for (int n = 0; n < N; ++n) {
        for (int cout = 0; cout < Cout; ++cout) {
            for (int h_out = 0; h_out < H_out; ++h_out) {
                for (int w_out = 0; w_out < W_out; ++w_out) {
                    T sum = 0;
                    for (int cin = 0; cin < Cin; ++cin) {
                        for (int kh = 0; kh < Kh; ++kh) {
                            for (int kw = 0; kw < Kw; ++kw) {
                                int input_index = (n * Cin * H * W) 
                                                  + (cin * H * W) 
                                                  + ((h_out + kh) * W) 
                                                  + (w_out + kw);
    
                                int filter_index = (cout * Cin * Kh * Kw) 
                                                   + (cin * Kh * Kw) 
                                                   + (kh * Kw) 
                                                   + kw;
    
                                sum += input[input_index] * filter[filter_index];
                            }
                        }
                    }
                    result[(n * Cout * H_out * W_out) + (cout * H_out * W_out) + (h_out * W_out) + w_out] = sum;
                }
            }
        }
    }

}

template void launch_conv2d_basic<float>(float *h_result, float *h_x, float *h_y, int Cin, int H, int W, int Cout, int K);
template void launch_conv2d_batched<float>(float *h_result, float *h_x, float *h_y, int N, int Cin, int H, int W, int Cout, int Kh, int Kw);
template void ref_conv<float>(float *h_result, const float *h_x, const float *h_y, int N, int Cin, int H, int W, int Cout, int Kh, int Kw);