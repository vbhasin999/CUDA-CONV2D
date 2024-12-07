#!/bin/bash

MODULE_NAME="conv2d"

PYBIND11_INCLUDES=$(python3 -m pybind11 --includes)

# If the system is MacOS, add an extra flag to the final compilation command.
if [[ "$(uname)" == "Darwin" ]]; then
    undefined_dynamic_lookup="-undefined dynamic_lookup"
else
    undefined_dynamic_lookup=""
fi

# Compile the CUDA source file to an object file
nvcc -c src/conv2d_impl.cu \
  -std=c++17 \
  -O3 \
  -lineinfo \
  -I/usr/local/cuda/include \
  -Xcompiler -fPIC \
  -o conv2d_impl.o

# Compile the C++ source file to an object file
g++ -c -std=c++17 -O3 -fPIC ${PYBIND11_INCLUDES} \
  src/${MODULE_NAME}.cc -o ${MODULE_NAME}.o

# Link the object files into a shared library
g++ ${MODULE_NAME}.o ${MODULE_NAME}_impl.o \
  -o ${MODULE_NAME}.so \
  -shared \
  -std=c++17 \
  -O3 \
  -L/usr/local/cuda/lib64 \
  -lcuda \
  -lcudart \
  ${undefined_dynamic_lookup} \
  ${PYBIND11_INCLUDES}

# remove intermediate .o files
rm -rf ${MODULE_NAME}.o ${MODULE_NAME}_impl.o
