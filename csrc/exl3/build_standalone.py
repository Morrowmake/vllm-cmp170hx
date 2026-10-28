# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the EXL3 extension vllm_sm80._exl3_C (SM 8.0, BF16) without rebuilding vLLM.

EXL3 checkpoints need it (vllm_sm80/kernels/exl3/ops.py).
Requires CUDA-enabled PyTorch, a matching CUDA toolkit, a C++20 compiler and
Ninja. No visible GPU is needed. The output directory is explicit; an installed
precompiled engine is retained. module.cpp registers the operators with the
non-stable TORCH_LIBRARY API, so the library works only with the PyTorch it
was built against: exl3.ops.ensure_extension compares its build_info() with
the running PyTorch and refuses another build.
"""

import argparse
import os
import shutil
import sys
import tempfile
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path

HERE = Path(__file__).resolve().parent
CSRC = HERE.parent
ROOT = CSRC.parent
NAME = "_exl3_C"
SOURCES = (
    "moe_prefill.cu",
    "dense.cu", "dense_splitk.cu", "gemm.cu", "moe.cu", "moe_v2.cu", "ple.cu",
    "dense_mma.cu",
    "ple_read.cpp", "engram_fetch.cpp", "reference.cu",
    "module.cpp",
    "dense_prefill_tc.cu",
    "moe_prefill_v2.cu",
    # VLLM_EXL3_MOE_V2_RING: own namespace and headers (moe_v2_try/); leaves every
    # other object's code as it is.
    "moe_v2_try.cu",
    "moe_prefill_try.cu",
)


def build(out: Path, build_dir: Path, verbose: bool) -> None:
    # Never infer architectures from visible devices or initialise CUDA.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get(
        "PATH", ""
    )
    import torch
    from torch.utils import cpp_extension

    if torch.version.cuda is None or cpp_extension.CUDA_HOME is None:
        raise RuntimeError("A CUDA-enabled PyTorch and matching CUDA toolkit are required")
    build_dir.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    # Keep compiler diagnostics and __FILE__ strings relocatable.
    maps = [
        f"-ffile-prefix-map={ROOT}=vllm",
        f"-ffile-prefix-map={build_dir}=build",
        f"-ffile-prefix-map={Path.home()}=source",
        f"-ffile-prefix-map={Path(torch.__file__).resolve().parent}=torch",
    ]
    # The flags the former CMake target compiled with (VLLM_GPU_FLAGS for
    # SM 8.0 plus its definitions), so both builds generate the same code.
    defines = ["-DNDEBUG", "-DUSE_CUDA", "-DPy_LIMITED_API=3"]
    cuda_flags = [
        "-O3", "-std=c++20", *defines, "-DENABLE_FP8", "-DENABLE_SCALED_MM_C2X=1",
        "--expt-relaxed-constexpr", "--expt-extended-lambda", "--compress-mode=size",
        # torch.utils.cpp_extension defines these; the CMake build does not.
        "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__",
        *[f"-Xcompiler={flag}" for flag in maps],
    ]
    cpp_extension.load(
        name=NAME,
        sources=[str(HERE / source) for source in SOURCES],
        extra_include_paths=[str(CSRC)],
        extra_cuda_cflags=cuda_flags,
        extra_cflags=["-O3", "-std=gnu++20", *defines, *maps],
        build_directory=str(build_dir),
        is_python_module=False,
        verbose=verbose,
    )
    source = build_dir / f"{NAME}.so"
    # Match the source-build install path; tagged files outrank plain .so imports.
    destination = out / f"{NAME}.abi3.so"
    with tempfile.NamedTemporaryFile(dir=out, prefix=NAME, suffix=".tmp", delete=False) as f:
        temporary = Path(f.name)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    for suffix in EXTENSION_SUFFIXES:
        obsolete = out / (NAME + suffix)
        if obsolete != destination:
            obsolete.unlink(missing_ok=True)
    print(f"{destination} ({destination.stat().st_size} bytes)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True,
                        help="output directory for _exl3_C.abi3.so (the tree's vllm/)")
    parser.add_argument("--build-dir", type=Path,
                        help="persistent build directory (default: temporary)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.build_dir is None:
        with tempfile.TemporaryDirectory(prefix="exl3_") as directory:
            build(args.out.resolve(), Path(directory), args.verbose)
    else:
        build(args.out.resolve(), args.build_dir.resolve(), args.verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
