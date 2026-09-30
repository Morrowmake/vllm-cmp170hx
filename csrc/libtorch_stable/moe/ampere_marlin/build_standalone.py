# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the optional sm_80 decode and prefill library without rebuilding vLLM.

Requires VLLM_BUILD_AMPERE_MARLIN=1, CUDA-enabled PyTorch, a matching CUDA
toolkit (12.8 or newer), a C++20 compiler and Ninja. No visible GPU is needed.
The output directory is explicit; an installed precompiled engine is retained.
"""

import argparse
from importlib.machinery import EXTENSION_SUFFIXES
import os
from pathlib import Path
import shutil
import sys
import tempfile

HERE = Path(__file__).resolve().parent
CSRC = HERE.parents[2]
ROOT = CSRC.parent
NAME = "_ampere_marlin_C"


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
    flags = ["-O3", "-std=c++20", "-DUSE_CUDA"]
    cuda_flags = [
        *flags, "--expt-relaxed-constexpr", "--threads=4",
        "-static-global-template-stub=false",
        "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__",
        "-DENABLE_FP8", *[f"-Xcompiler={flag}" for flag in maps],
    ]
    cpp_extension.load(
        name=NAME,
        sources=[str(HERE / source) for source in
                 ("decode.cu", "decode_orig.cu", "ops.cu", "kernels_sm80.cu",
                  "module.cpp")],
        extra_include_paths=[str(CSRC)],
        extra_cuda_cflags=cuda_flags,
        extra_cflags=[*flags, *maps],
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
                        help="output directory for _ampere_marlin_C.abi3.so")
    parser.add_argument("--build-dir", type=Path,
                        help="persistent build directory (default: temporary)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if os.environ.get("VLLM_BUILD_AMPERE_MARLIN", "0") != "1":
        parser.error("set VLLM_BUILD_AMPERE_MARLIN=1 to build this optional library")
    if args.build_dir is None:
        with tempfile.TemporaryDirectory(prefix="ampere_marlin_") as directory:
            build(args.out.resolve(), Path(directory), args.verbose)
    else:
        build(args.out.resolve(), args.build_dir.resolve(), args.verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
