#include "conv2d_impl.h"
#include <stdio.h>

#define BLOCK_DIM_X 2
#define BLOCK_DIM_Y 2
#define BLOCK_DIM_Z 1
#define PIXELS_PER_THREAD 16

// Add batch, stride and padding later
// we could have 1 block <-> 1 output channel
// and 1 thread <-> 1 output pixel

template <typename T>
__global__ void conv_kernel_basic(
    T* __restrict__ result, 
    const T* __restrict__ input, 
    const T* __restrict__ filter, 
    int Cin, int H, int W, int Cout, int K) {

    int out_x = threadIdx.x + blockIdx.x * blockDim.x;  // Output width index
    int out_y = threadIdx.y + blockIdx.y * blockDim.y;  // Output height index
    int out_c = blockIdx.z;                            // Output channel index                 

    int H_out = H - K + 1;
    int W_out = W - K + 1;

    // Shared memory for reduction
    __shared__ T partial_sums[BLOCK_DIM_X][BLOCK_DIM_Y];  
    // Initialize shared memory only once per block
    if (threadIdx.z == 0) partial_sums[threadIdx.x][threadIdx.y]= 0.0f;

    __syncthreads();

    T local_sum = 0;
    if (out_x < H_out && out_y < W_out) {
        // Loop over input channels in chunks
        for (int in_c = threadIdx.z; in_c < Cin; in_c += BLOCK_DIM_Z) {
            for (int kx = 0; kx < K; ++kx) { // Kernel rows
                for (int ky = 0; ky < K; ++ky) { // Kernel columns
                    int in_x = out_x + kx;
                    int in_y = out_y + ky;

                    if (in_x < W && in_y < H) {  // Bounds check
                        local_sum += input[in_c * H * W + in_y * W + in_x] *
                                     filter[out_c * Cin * K * K + in_c * K * K + ky * K + kx];
                    }
                }
            }
        }

        // Atomic addition to shared memory
        atomicAdd(&partial_sums[threadIdx.x][threadIdx.y], local_sum);
    }

    __syncthreads();

    // Write final result to global memory
    if (threadIdx.z == 0) {
        if (out_x < H_out && out_y < W_out) {
            result[out_c * H_out * W_out + out_y * W_out + out_x] = partial_sums[threadIdx.x][threadIdx.y];
        }
    }
}


template <typename T>
void launch_conv2d_basic(T *h_result, const T *h_x, const T *h_y, int Cin, int H, int W, int Cout, int K) {
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
    dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z);  
    dim3 gridDim((W_out + blockDim.x - 1) / blockDim.x,
                (H_out + blockDim.y - 1) / blockDim.y,
                Cout);   

    // Launch kernel
    conv_kernel_basic<<<gridDim, blockDim>>>(d_result, d_x, d_y, Cin, H, W, Cout, K);
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
__global__ void conv_kernel_opt(
    T* __restrict__ result, 
    const T* __restrict__ input, 
    const T* __restrict__ filter, 
    int Cin, int H, int W, int Cout, int K) {

    int out_x = threadIdx.x + blockIdx.x * blockDim.x;  // Output width index
    int out_y = threadIdx.y + blockIdx.y * blockDim.y;  // Output height index
    int in_c = blockIdx.z;                            // Input channel index                 

    int H_out = H - K + 1;
    int W_out = W - K + 1;

    // should be of size blockDim.x + K - 1 * blockDim.y + K - 1 i.e. 
    // the receptive field of the block
    extern __shared__ T sInput[];
    if ((threadIdx.z == 0) && (threadIdx.x % K == 0) && (threadIdx.y % K == 0)) {
        for (int i = 0; i < K; ++i) {
            for (int j = 0; j < K; ++j) {
                int in_x = out_x + i;
                int in_y = out_y + j;
                
                // printf("Block: (%d, %d, %d), Thread: (%d, %d, %d), loading: %f into smem\n",
                //             blockIdx.x, blockIdx.y, blockIdx.z,
                //             threadIdx.x, threadIdx.y, threadIdx.z,
                //             input[in_c * H * W + in_y * W + in_x]);
                sInput[(threadIdx.y + i) * (BLOCK_DIM_X + K - 1) + threadIdx.x + j] = (in_x < W && in_y < H) ? input[in_c * H * W + in_y * W + in_x] : 0.0f;
            }
        }
    }

    __syncthreads();

    // if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0){
    //     printf("smem loaded: %f %f %f %f", sInput[0], sInput[1], sInput[2],sInput[3]);
    // }

    
    if (out_x < H_out && out_y < W_out && in_c < Cin) {
        // Loop over output channels in chunks
        for (int out_c = threadIdx.z; out_c < Cout; out_c += BLOCK_DIM_Z) {
            // printf("Block: (%d, %d, %d), Thread: (%d, %d, %d), out_c: %d\n",
            //         blockIdx.x, blockIdx.y, blockIdx.z,
            //         threadIdx.x, threadIdx.y, threadIdx.z,
            //         out_c);
            T local_sum = 0;
            for (int kx = 0; kx < K; ++kx) { // Kernel rows
                for (int ky = 0; ky < K; ++ky) { // Kernel columns
                    int in_x = out_x + kx;
                    int in_y = out_y + ky;

                    if (in_x < W && in_y < H) {  // Bounds check
                        local_sum += sInput[(threadIdx.y + kx) * (BLOCK_DIM_X + K - 1) + threadIdx.x + ky] *
                                        filter[(out_c * Cin * K * K )+ (in_c * K * K) + (kx * K) + ky];
                    }
                }
            }
            // printf("Block: (%d, %d, %d), Thread: (%d, %d, %d), local sum: %f\n",
            //                 blockIdx.x, blockIdx.y, blockIdx.z,
            //                 threadIdx.x, threadIdx.y, threadIdx.z,
            //                 local_sum);            
            atomicAdd(&result[out_c * H_out * W_out + out_y * W_out + out_x], local_sum);
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
    dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z);  
    dim3 gridDim((W_out + blockDim.x - 1) / blockDim.x,
                (H_out + blockDim.y - 1) / blockDim.y,
                Cin);   

    size_t shared_mem_size = (BLOCK_DIM_X + K - 1) * (BLOCK_DIM_Y + K - 1) * sizeof(T);

    // Launch kernel
    conv_kernel_opt<<<gridDim, blockDim, shared_mem_size>>>(d_result, d_x, d_y, Cin, H, W, Cout, K);
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