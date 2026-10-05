// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <Python.h>
#include <cuda_runtime_api.h>
#include <torch/version.h>

namespace {
PyObject* build_info(PyObject*, PyObject*) {
  // prefill_schedule_version 2: prefill_tile_gemm hands out stream-K CTA
  // indices in start order (MARLIN_MOE_ORDERED_STREAM_K, common_tiles.h).
  return Py_BuildValue("{s:i,s:s,s:i,s:i,s:i}", "abi_version", 1,
                       "torch_version", TORCH_VERSION,
                       "cuda_version", CUDART_VERSION,
                       "cxx11_abi", _GLIBCXX_USE_CXX11_ABI,
                       "prefill_schedule_version", 2);
}

PyMethodDef methods[] = {
    {"build_info", build_info, METH_NOARGS, "Return the compiled ABI metadata."},
    {nullptr, nullptr, 0, nullptr},
};
}  // namespace

PyMODINIT_FUNC PyInit__ampere_marlin_C() {
  static PyModuleDef module = {PyModuleDef_HEAD_INIT, "_ampere_marlin_C",
                              nullptr, 0, methods};
  return PyModule_Create(&module);
}
