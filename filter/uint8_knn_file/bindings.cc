#include "uint8_knn.hpp"
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <chrono>

namespace py = pybind11;
using namespace pybind11::literals; // needed to bring in _a literal

inline void get_input_array_shapes(const py::buffer_info &buffer, size_t *rows,
                                   size_t *features) {
  if (buffer.ndim != 2 && buffer.ndim != 1) {
    char msg[256];
    snprintf(msg, sizeof(msg),
             "Input vector data wrong shape. Number of dimensions %d. Data "
             "must be a 1D or 2D array.",
             buffer.ndim);
  }
  if (buffer.ndim == 2) {
    *rows = buffer.shape[0];
    *features = buffer.shape[1];
  } else {
    *rows = 1;
    *features = buffer.shape[0];
  }
}

py::object knn(py::object Q, py::object X, int32_t topk) {
  py::array_t<uint8_t, py::array::c_style | py::array::forcecast> qq(Q);
  py::array_t<uint8_t, py::array::c_style | py::array::forcecast> xx(X);
  size_t nq, d, nx;
  get_input_array_shapes(qq.request(), &nq, &d);
  get_input_array_shapes(xx.request(), &nx, &d);
  int *ids = new int[nq * topk];
  uint8_knn(d, qq.data(0), nq, xx.data(0), nx, topk, ids);
  py::capsule free_when_done(ids, [](void *f) { delete[] f; });
  return py::array_t<int>({nq * topk}, {sizeof(int)}, ids, free_when_done);
}

PYBIND11_PLUGIN(uint8_knn) {
  py::module m("uint8_knn");

  m.def("knn", &knn, py::arg("Q"), py::arg("X"), py::arg("topk"));

  return m.ptr();
}