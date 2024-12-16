import numpy as np
import conv2d  # The compiled module
import time
import argparse
import conv2d

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run Conv2D and compare with reference implementation.")
parser.add_argument("--Cin", type=int, default=3, help="Number of input channels")
parser.add_argument("--H", type=int, default=128, help="Height of input tensor")
parser.add_argument("--W", type=int, default=128, help="Width of input tensor")
parser.add_argument("--Cout", type=int, default=8, help="Number of output channels")
parser.add_argument("--K", type=int, default=8, help="Kernel size")

args = parser.parse_args()

# Set parameters from command-line arguments
Cin, H, W, Cout, K = args.Cin, args.H, args.W, args.Cout, args.K

# Generate random input and filter tensors
input_tensor = np.random.rand(Cin, H, W).astype(np.float32)  # Input: (Cin, H, W)
filter_tensor = np.random.rand(Cout, Cin, K, K).astype(np.float32)  # Filter: (Cout, Cin, K, K)
# filter_tensor = np.arange(K*K).astype(np.float32).reshape(Cout, Cin, K, K)

# print(f"input\n{input_tensor} \nfilter\n{filter_tensor}")

# Reference implementation of convolution and time it
H_out = H - K + 1
W_out = W - K + 1
start_time_ref = time.time()
reference_output = conv2d.conv2d_ref(input_tensor, filter_tensor, Cin, H, W, Cout, K)
end_time_ref = time.time()
ref_time = end_time_ref - start_time_ref
print(f"Reference Conv2D Execution Time: {ref_time*1000:.6f} miliseconds")


start_time_cuda_basic = time.time()
output_tensor = conv2d.conv2d_basic(input_tensor, filter_tensor, Cin, H, W, Cout, K)
end_time_cuda_basic = time.time()
cuda_time_basic = end_time_cuda_basic - start_time_cuda_basic
print(f"CUDA Conv2D BASIC Execution Time: {cuda_time_basic*1000:.6f} miliseconds")
print(f"Speedup: {ref_time / cuda_time_basic :.2f}x")
print("Are outputs close:", np.allclose(output_tensor, reference_output, atol=1e-5))
# print(f'out: \n{output_tensor} \nref: \n{reference_output}')


start_time_cuda_cout = time.time()
output_tensor = conv2d.conv2d_cout(input_tensor, filter_tensor, Cin, H, W, Cout, K)
end_time_cuda_cout = time.time()
cuda_time_cout = end_time_cuda_cout - start_time_cuda_cout
print(f"CUDA Conv2D COUT Execution Time: {cuda_time_cout*1000:.6f} miliseconds")
print(f"Speedup: {ref_time / cuda_time_cout :.2f}x")
print("Are outputs close:", np.allclose(output_tensor, reference_output, atol=1e-5))



# Call the conv2d function and time it
start_time_cuda = time.time()
output_tensor = conv2d.conv2d_opt(input_tensor, filter_tensor, Cin, H, W, Cout, K)
end_time_cuda = time.time()
cuda_time = end_time_cuda - start_time_cuda
print(f"CUDA Conv2D OPT Execution Time: {cuda_time*1000:.6f} miliseconds")
print(f"Speedup: {ref_time / cuda_time :.2f}x")
print("Are outputs close:", np.allclose(output_tensor, reference_output, atol=1e-5))






# Compare results
# print(f"\nCorrectness:")
# print("Output tensor shape:", output_tensor.shape)
# print("Reference tensor shape:", reference_output.shape)

# Print timing information

