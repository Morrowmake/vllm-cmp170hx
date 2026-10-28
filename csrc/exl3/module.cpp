#include "moe_prefill.h"
#include "moe_v2.h"
#include "ops.h"
#include "ple.h"
#include "dense_splitk.h"
#include "engram_fetch.h"
#include <Python.h>
#include <cuda_runtime_api.h>
#include <torch/library.h>
#include <torch/version.h>

TORCH_LIBRARY(_exl3_C, m) {
  m.def("linear_out(Tensor x, Tensor trellis, Tensor suh, Tensor svh, "
        "int bits_x2, int codebook, int logical_in, int logical_out, "
        "Tensor(a!) out, Tensor(b!) workspace) -> ()");
  m.def("linear_out_splitk(Tensor x, Tensor trellis, Tensor suh, Tensor svh, "
        "int bits_x2, int codebook, int logical_in, int logical_out, "
        "Tensor(a!) out, Tensor(b!) workspace) -> ()");
  m.def("moe_out(Tensor x, Tensor topk_ids, Tensor topk_weights, "
        "Tensor packed_ptrs, Tensor packed_meta, Tensor expert_map, "
        "int intermediate_size, Tensor(a!) out, Tensor(b!) workspace, "
        "float swiglu_limit=0.0) -> ()");
  m.def("moe_out_v2(Tensor x, Tensor topk_ids, Tensor topk_weights, "
        "Tensor packed_ptrs, Tensor packed_meta, Tensor expert_map, "
        "int intermediate_size, int bits_x2, Tensor(a!) out, "
        "Tensor(b!) workspace, float swiglu_limit=0.0) -> ()");
  m.def("moe_out_prefill(Tensor x, Tensor topk_ids, Tensor topk_weights, "
        "Tensor packed_ptrs, Tensor packed_meta, Tensor expert_map, "
        "int intermediate_size, Tensor(a!) out, Tensor(b!) workspace, "
        "float swiglu_limit=0.0) -> ()");
  m.def("ple_decode_out(Tensor packed_rows, Tensor global_ids, Tensor head_bias, "
        "Tensor head_offsets, Tensor head_vocab_sizes, int owner_lo, "
        "int owner_hi, int bits, Tensor(a!) out) -> ()");
  m.def("ple_read_rows(int fd, Tensor ids, Tensor(a!) dest, int count, "
        "int base, int row_bytes, int owner_lo, int owner_hi, "
        "Tensor head_offsets, Tensor head_vocab_sizes, int retries, "
        "int max_threads) -> Tensor");
  register_ple_host_reader(m);
  register_engram_host_fetch(m);
  // No tensor arguments: these are catch-all CPU arithmetic functions, not
  // CUDA-dispatched ops, and never inspect a device or initialize CUDA.
  m.def("linear_workspace_size(int max_rows, int padded_in, int padded_out) -> int",
        &exl3::linear_workspace_size);
  m.def("linear_splitk_splits(int rows, int padded_out, int sms, int bits_x2, "
        "int codebook) -> int", &exl3::linear_splitk_splits);
  m.def("moe_workspace_size(int max_rows, int hidden_size, int intermediate_size, "
        "int topk, int local_experts) -> int", &exl3::moe_workspace_size);
  m.def("moe_workspace_size_v2(int max_rows, int hidden_size, int intermediate_size, "
        "int topk, int local_experts) -> int", &exl3::moe_workspace_size_v2);
  m.def("moe_workspace_size_prefill(int max_rows, int hidden_size, "
        "int intermediate_size, int topk, int local_experts) -> int",
        &exl3::moe_workspace_size_prefill);
  // Test-only: today's scalar kernels as the bit-exact oracle, and a block
  // shape override so a test can run every shape at every row count.
  m.def("linear_out_reference(Tensor x, Tensor trellis, Tensor suh, Tensor svh, "
        "int bits_x2, int codebook, int logical_in, int logical_out, "
        "Tensor(a!) out, Tensor(b!) workspace) -> ()");
  m.def("moe_out_reference(Tensor x, Tensor topk_ids, Tensor topk_weights, "
        "Tensor packed_ptrs, Tensor packed_meta, Tensor expert_map, "
        "int intermediate_size, Tensor(a!) out, Tensor(b!) workspace, "
        "float swiglu_limit=0.0) -> ()");
  m.def("set_gemm_shape_override(int shape) -> ()", &exl3::set_gemm_shape_override);
}

TORCH_LIBRARY_IMPL(_exl3_C, CUDA, m) {
  m.impl("linear_out", &exl3::linear_out);
  m.impl("linear_out_splitk", &exl3::linear_out_splitk);
  m.impl("moe_out", &exl3::moe_out);
  m.impl("moe_out_v2", &exl3::moe_out_v2);
  m.impl("moe_out_prefill", &exl3::moe_out_prefill);
  m.impl("ple_decode_out", &ple_decode_out);
  m.impl("linear_out_reference", &exl3::linear_out_reference);
  m.impl("moe_out_reference", &exl3::moe_out_reference);
}

TORCH_LIBRARY_IMPL(_exl3_C, CPU, m) {
  m.impl("ple_read_rows", &ple_read_rows);
}

namespace {

// TORCH_LIBRARY is not the stable ABI: exl3.ops.ensure_extension compares
// these with the running PyTorch and refuses a library built for another.
// abi_version 3: moe_out_v2 takes bits_x2 (2: the MoE operators take
// swiglu_limit).
PyObject* build_info(PyObject*, PyObject*) {
  return Py_BuildValue("{s:i,s:s,s:i,s:i}", "abi_version", 3, "torch_version",
                       TORCH_VERSION, "cuda_version", CUDART_VERSION,
                       "cxx11_abi", _GLIBCXX_USE_CXX11_ABI);
}

PyMethodDef methods[] = {
    {"build_info", build_info, METH_NOARGS, "Return the compiled ABI metadata."},
    {nullptr, nullptr, 0, nullptr},
};

}  // namespace

PyMODINIT_FUNC PyInit__exl3_C() {
  static PyModuleDef module = {PyModuleDef_HEAD_INIT, "_exl3_C", nullptr, 0,
                               methods};
  return PyModule_Create(&module);
}
