import numpy as np
import conv2d  # The compiled module
import time
import argparse

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

# Call the conv2d function and time it
start_time_cuda = time.time()
output_tensor = conv2d.conv2d(input_tensor, filter_tensor, Cin, H, W, Cout, K)
end_time_cuda = time.time()

print(f"CUDA Conv2D Execution Time: {end_time_cuda - start_time_cuda:.6f} seconds")

# Reference implementation of convolution and time it
H_out = H - K + 1
W_out = W - K + 1


start_time_ref = time.time()
reference_output = conv2d.conv2d_ref(input_tensor, filter_tensor, Cin, H, W, Cout, K, H_out, W_out)
end_time_ref = time.time()

# Compare results
print("Output tensor shape:", output_tensor.shape)
print("Reference tensor shape:", reference_output.shape)
print("Are outputs close:", np.allclose(output_tensor, reference_output, atol=1e-5))

# Print timing information

print(f"Reference Conv2D Execution Time: {end_time_ref - start_time_ref:.6f} seconds")