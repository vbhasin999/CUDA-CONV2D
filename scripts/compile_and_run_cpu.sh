#!/bin/bash

# Exit on any error
set -e

# Check if an input file was provided
if [ -z "$1" ]; then
  echo "Usage: $0 <mlir_file_name> [function_type]"
  echo "Example: $0 vecadd-generic.mlir vecadd"
  exit 1
fi

# Get the absolute path of the root directory
ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)

# Define directory paths
MLIR_DIR="$ROOT_DIR/mlir"
COMPILED_DIR="$ROOT_DIR/compiled"

# Input MLIR file
MLIR_FILE="$MLIR_DIR/$1"

# Ensure the input file exists
if [ ! -f "$MLIR_FILE" ]; then
  echo "Error: MLIR file '$MLIR_FILE' does not exist in $MLIR_DIR."
  exit 1
fi

# Extract base name (remove extension)
BASE_NAME=$(basename "$MLIR_FILE" .mlir)

# Output VMFB file
VMFB_FILE="$COMPILED_DIR/${BASE_NAME}.vmfb"

# Ensure the compiled directory exists
mkdir -p "$COMPILED_DIR"

# Compile the MLIR file
echo "Compiling $MLIR_FILE to $VMFB_FILE..."
iree-compile --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-target-cpu=host "$MLIR_FILE" -o "$VMFB_FILE"
echo "Compilation successful!"

# Determine the function type
FUNCTION_TYPE=${2:-vecadd} # Default to "vecadd" if no type is specified

# Debugging output for casing
echo "Function type: $FUNCTION_TYPE"
echo "Base name: $BASE_NAME"

# Run the compiled VMFB file
echo "Running $VMFB_FILE..."
case "$FUNCTION_TYPE" in
  vecadd)
    echo "Executing vecadd..."
    iree-run-module --module="$VMFB_FILE" \
      --input=10xf32=[0,1,2,3,4,5,6,7,8,9] \
      --input=10xf32=[90,80,70,60,50,40,30,20,10,0]
    ;;
  matmul)
    echo "Executing matmul..."
    iree-run-module --module="$VMFB_FILE" \
      --input=2x2xf32=[[1,2],[3,4]] \
      --input=2x2xf32=[[1,4],[3,2]] \
      --input=2x2xf32=[[0,0],[0,0]]
    ;;
  *)
    echo "Error: Unknown function type '$FUNCTION_TYPE'. Supported types are: vecadd, matmul."
    exit 1
    ;;
esac