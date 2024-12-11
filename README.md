# 15618 F24 MLIR-IREE Project

This project demonstrates the use of **MLIR** (Multi-Level Intermediate Representation) with **IREE** (Intermediate Representation Execution Environment). The workflow includes compiling MLIR files and executing the compiled artifacts using IREE tools. This project builds on [IREE MLIR Linalg Tutorial](https://iree.dev/community/blog/2024-01-29-iree-mlir-linalg-tutorial/). It demonstrates the use of MLIR with IREE for compiling and executing tensor computations.

## Project Roadmap
- [x] Implement Naive CUDA Conv2d
- [x] Implement optimized Conv2d
- [ ] Further optimize, cooperative fetching for smem, vectorized ops, thread tiling, minimizing atomic adds
- [ ] Add tensor core optimizations to conv2d opt
- [ ] Implement CUTLASS Conv2d
80% of the project goal achieved by this point
- [ ] Implement MLIR conv2d (ideally in mlir, backup plan: maybe in pytorch -> MLIR, or Trition or openACC or some other higher level acceleration library)
- [ ] Conduct benchmarking and performance experiments
100% of project goal achieved by this point, everything that follows are potential extensions
- [ ] Depthwise separable convolutions over vanilla convolutions
- [ ] fused MHSA kernel 
- [ ] Architecture specific kenels (Turing, Ampere)

## Requirements

Ensure you have the following installed:
- Python 3.10 or later
- IREE tools: `iree-base-compiler`, `iree-base-runtime`

## Installation

Install the required Python packages using `pip`:

```bash
python -m pip install \
  iree-base-compiler \
  iree-base-runtime
```

## Steps to Compile and Run an MLIR File

1. **Prepare an MLIR File**

   Write an MLIR file (e.g., `prog.mlir`) with the required computation. Below is an example structure with matrix multiplication:

   ```mlir
   func.func @foo(%lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>, %acc: tensor<?x?xf32>) -> tensor<?x?xf32> {
     %result = linalg.matmul
       ins(%lhs, %rhs: tensor<?x?xf32>, tensor<?x?xf32>)
       outs(%acc: tensor<?x?xf32>)
     -> tensor<?x?xf32>
     return %result: tensor<?x?xf32>
   }
   ```

2. **Compile the MLIR File**

   Use `iree-compile` to compile the MLIR file into a VM flatbuffer (`.vmfb`):

   ```bash
   iree-compile --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-target-cpu=host prog.mlir -o ./prog.vmfb
   ```
   #### Supported Backends and Devices

   The table below lists the available IREE backends and corresponding devices:

   | Backend          | Description                   | Example Device           |
   |-------------------|-------------------------------|--------------------------|
   | `llvm-cpu`       | CPU backend                   | `--device=local-task`    |
   | `cuda`           | NVIDIA GPU backend (CUDA)     | `--device=cuda`          |
   | `rocm`           | AMD GPU backend (ROCm)        | `--device=rocm`          |
   | `vulkan-spirv`   | GPU backend (Vulkan/SPIR-V)   | `--device=vulkan`        |

   For a full list of supported configurations, refer to the [IREE Deployment Configurations Guide](https://iree.dev/guides/deployment-configurations/).

   ---

   #### Example Commands

    In this example we compile for the `llvm-cpu` backend:\, passing in `host` as the target CPU to optimize
    for the host machine.

   ```bash
   iree-compile \
     --iree-hal-target-backends=llvm-cpu \
     --iree-llvmcpu-target-cpu=host \
     input.mlir -o output.vmfb
    ```

3. **Run the Compiled Module**

   Execute the compiled `.vmfb` file using `iree-run-module`:

   ```bash
   iree-run-module --module=./prog.vmfb \
     --input=10xf32=[0,1,2,3,4,5,6,7,8,9] \
     --input=10xf32=[90,80,70,60,50,40,30,20,10,0]
   ```

   Example output:

   ```
   EXEC @foo
   result[0]: hal.buffer_view
   10xf32=90 81 72 63 54 45 36 27 18 9
   ```

## Directory Structure

```
project-root/
├── mlir-examples/           # Contains MLIR files
│   └── prog.mlir            # Example MLIR program
├── compiled/                # Compiled VMFB files
│   └── prog.vmfb            # Output from iree-compile
├── scripts/                 # Helper scripts
│   └── run_example.sh       # Automates the steps above
└── README.md                # Project instructions
```

