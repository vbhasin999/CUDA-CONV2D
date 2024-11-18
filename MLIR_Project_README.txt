# MLIR-IREE Project

This project demonstrates the use of **MLIR** (Multi-Level Intermediate Representation) with **IREE** (Intermediate Representation Execution Environment). The workflow includes compiling MLIR files and executing the compiled artifacts using IREE tools.

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

   Write an MLIR file (e.g., `prog.mlir`) with the required computation. Below is an example structure:

   ```mlir
   func.func @foo(
         %lhs : tensor<10xf32>,
         %rhs : tensor<10xf32>
       ) -> tensor<10xf32> {
     %result = linalg.generic {
       indexing_maps = [#map_1d_identity, #map_1d_identity, #map_1d_identity],
       iterator_types = ["parallel"]
     } ins(%lhs, %rhs : tensor<10xf32>, tensor<10xf32>)
       outs(tensor.empty() : tensor<10xf32>)
     {
       ^bb0(%lhs_entry : f32, %rhs_entry : f32, %result_entry : f32):
         %add = arith.addf %lhs_entry, %rhs_entry : f32
         linalg.yield %add : f32
     } -> tensor<10xf32>
     return %result : tensor<10xf32>
   }
   ```

2. **Compile the MLIR File**

   Use `iree-compile` to compile the MLIR file into a VM flatbuffer (`.vmfb`):

   ```bash
   iree-compile --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-target-cpu=host prog.mlir -o ./prog.vmfb
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

Here’s a suggested directory layout for the project:

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

## Notes

- **Backends**: This example uses the `llvm-cpu` backend. Other backends like `vulkan-spirv` can also be used by modifying the `--iree-hal-target-backends` flag.
- **Inputs**: Adjust the inputs (`--input`) to match the tensor shapes and types expected by your MLIR function.
