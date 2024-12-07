#include "conv2d_impl.h"

// Add batch, stride and padding later
// we could have 1 block <-> 1 output channel
// and 1 thread <-> 1 output pixel
template <typename T>
__global__ void conv_kernel(T *result, const T *input, const T *filter, int Cin, int H, int W, int Cout, int K) {
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
void launch_conv2d(T *h_result, T *h_x, T *h_y, 
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
    conv_kernel<<<numBlocks, threadsPerBlock>>>(d_result, d_x, d_y, Cin, H, W, Cout, K);

    // Copy results back to host
    cudaMemcpy(h_result, d_result, output_size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_result);
}