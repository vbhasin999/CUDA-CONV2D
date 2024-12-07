#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "conv2d_impl.h"

namespace py = pybind11;

PYBIND11_MODULE(conv2d, m) {
    m.doc() = "pybind11 conv2d plugin";  // module docstring

    // Wrap launch_conv2d function
    m.def("conv2d", [](py::array_t<float> input, py::array_t<float> filter, int Cin, int H, int W, int Cout, int K) {
        // Ensure proper array dimensions
        py::buffer_info input_info = input.request();
        py::buffer_info filter_info = filter.request();

        if (input_info.ndim != 3 || filter_info.ndim != 4) {
            throw std::runtime_error("Invalid input dimensions. Input must be (Cin, H, W) and filter must be (Cout, Cin, K, K).");
        }

        // Extract dimensions
        auto result_shape = std::vector<ssize_t>{Cout, H - K + 1, W - K + 1};

        // Allocate output array
        py::array_t<float> result(result_shape);
        py::buffer_info result_info = result.request();

        // Call the CUDA kernel launcher
        launch_conv2d(
            static_cast<float *>(result_info.ptr),
            static_cast<float *>(input_info.ptr),
            static_cast<float *>(filter_info.ptr),
            Cin, H, W, Cout, K
        );

        return result;
    }, "Perform 2D convolution using CUDA",
       py::arg("input"), py::arg("filter"), py::arg("Cin"), py::arg("H"), py::arg("W"), py::arg("Cout"), py::arg("K"));
}