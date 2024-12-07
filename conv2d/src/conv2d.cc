#include "pybind11/pybind11.h"

#include "conv2d_impl.h"

PYBIND11_MODULE(conv2d, m) {
    m.doc() = "pybind11 conv2d plugin";  // module docstring

    m.def("add", &add<int>, "An example function to add two numbers.");
}
