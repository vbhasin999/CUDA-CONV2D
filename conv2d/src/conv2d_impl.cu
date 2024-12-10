#include "conv2d_impl.h"
#include <stdio.h>

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
void launch_conv2d_basic(T *h_result, const T *h_x, const T *h_y, 
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
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    }

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
// template <typename T>
// __global__ void conv_kernel_batched(T *result, const T *input, const T *filter, int N, int Cin, int H, int W, int Cout, int Kh, int Kw) {
//     int x = threadIdx.x; // output channel
//     int y = threadIdx.y; // output pixel
//     int z = threadIdx.z; // batch

//     int H_out = H - Kh + 1;
//     int W_out = W - Kw + 1;

//     int t_r = y / W_out;
//     int t_c = y % W_out;

//     T sum = 0;

//     if (x < Cout && y < H_out * W_out && z < N){
//         for (int i = 0; i < Cin; ++i ){
//             for (int j = 0; j < Kh; ++j){
//                 for (int k = 0; k < Kw; ++k){
//                     printf("z: %d, x: %d, y: %d, i: %d, j: %d, k: %d\n sum += %f * %f", z, x, y, i, j, k, input[(z * Cin * H * W) + (i * H * W) + ((t_r + j) * W) + (k + t_c) ], filter[(x * Cin * Kh * Kw) + (i * Kh * Kw) + (j * Kw) + k ]);
//                     sum += input[(z * Cin * H * W) + (i * H * W) + ((t_r + j) * W) + (k + t_c) ] * filter[(x * Cin * Kh * Kw) + (i * Kh * Kw) + (j * Kw) + k ];
    
//                 }
//             }
//         }

//         // result of shape N, Cout, Hout, Wout
//         result[(z * Cout * H_out * W_out) + (x * H_out * W_out) + (t_r * W_out) + t_c] = sum;

//     }
    
    
// }


// 1 thread should handle the contribution of 1 input channel to 1 output channel
template <typename T>
__global__ void conv_kernel_opt(
    T* __restrict__ result, 
    const T* __restrict__ input, 
    const T* __restrict__ filter, 
    int Cin, int H, int W, int Cout, int K) {

    int out_x = threadIdx.x + blockIdx.x * blockDim.x;  // Output width index
    int out_y = threadIdx.y + blockIdx.y * blockDim.y;  // Output height index
    int out_c = blockIdx.z;                            // Output channel index
    int in_c = threadIdx.z;                            // Input channel index within block

    int H_out = H - K + 1;
    int W_out = W - K + 1;

    // Shared memory for reduction
    __shared__ T partial_sums[16][16];  
    // Initialize shared memory only once per block
    if (threadIdx.z == 0) partial_sums[threadIdx.x][threadIdx.y]= 0.0f;

    __syncthreads();

    T local_sum = 0;
    if (out_x < H_out && out_y < W_out){
        // loop over filter dimensions
        for (int kx = 0; kx < K; ++kx) { // loop over kernel rows
            for (int ky = 0; ky < K; ++ky) { // loop over kernel columns
                // input of shape (Cin, H, W) with strides (H*W, W, 1)
                // filter of shape (Cout, Cin, K, K) with strides (Cin*K*K, K*K, K, 1)
                if ((out_x + kx < W) && (out_y + ky < H)) {
                    local_sum += input[in_c * H * W + (out_x + kx) * W + (out_y + ky)] *
                                filter[out_c * Cin * K * K + in_c * K * K + kx * K + ky];
                }
            }
        }


        atomicAdd(&partial_sums[threadIdx.x][threadIdx.y], local_sum);

        // printf("\n\ntx %d \tty %d added %f to smem, smem is now %f", threadIdx.x, threadIdx.y, local_sum, partial_sums[threadIdx.x][threadIdx.y]);

        __syncthreads();

        if (threadIdx.z == 0) {
            // result of shape (Cout, H_out, W_out) with strides (H_out * W_out, W_out, 1)
            result[out_c * H_out * W_out + out_x * W_out + out_y] = partial_sums[threadIdx.x][threadIdx.y];
        }
    }
}

template <typename T>
void launch_conv2d_opt(T *h_result, const T *h_x, const T *h_y, int Cin, int H, int W, int Cout, int K) {
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
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "1. CUDA error: %s\n", cudaGetErrorString(err));
    }

    // Copy inputs to device
    cudaMemcpy(d_x, h_x, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, filter_size, cudaMemcpyHostToDevice);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "2. CUDA error: %s\n", cudaGetErrorString(err));
    }

    // Define grid and block dimensions
    dim3 blockDim(16, 16, 3);  // Threads per block: 16x16 spatial, 4 for input channels
    dim3 gridDim((W_out + blockDim.x - 1) / blockDim.x,
                (H_out + blockDim.y - 1) / blockDim.y,
                Cout);   

    // Launch kernel
    conv_kernel_opt<<<gridDim, blockDim>>>(d_result, d_x, d_y, Cin, H, W, Cout, K);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "3. CUDA error: %s\n", cudaGetErrorString(err));
    }

    // Copy results back to host
    cudaMemcpy(h_result, d_result, output_size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_result);
}

template <typename T>
void ref_conv(T *result, const T *input, const T *filter, int Cin, int H, int W, int Cout, int K){
    int H_out = H - K + 1;
    int W_out = W - K + 1;


    for (int cout = 0; cout < Cout; ++cout) {
        for (int h_out = 0; h_out < H_out; ++h_out) {
            for (int w_out = 0; w_out < W_out; ++w_out) {
                T sum = 0;
                for (int cin = 0; cin < Cin; ++cin) {
                    for (int kh = 0; kh < K; ++kh) {
                        for (int kw = 0; kw < K; ++kw) {
                            int input_index = (cin * H * W) 
                                                + ((h_out + kh) * W) 
                                                + (w_out + kw);

                            int filter_index = (cout * Cin * K * K) 
                                                + (cin * K * K) 
                                                + (kh * K) 
                                                + kw;

                            sum += input[input_index] * filter[filter_index];
                        }
                    }
                }
                result[(cout * H_out * W_out) + (h_out * W_out) + w_out] = sum;
            }
        }
    }
    

}

template void launch_conv2d_basic<float>(float *h_result, const float *h_x, const float *h_y, int Cin, int H, int W, int Cout, int K);
template void launch_conv2d_opt<float>(float *h_result, const float *h_x, const float *h_y, int Cin, int H, int W, int Cout, int K);
template void ref_conv<float>(float *h_result, const float *h_x, const float *h_y, int Cin, int H, int W, int Cout, int K);