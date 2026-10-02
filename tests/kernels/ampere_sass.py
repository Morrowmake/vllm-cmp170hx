# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compile a Triton kernel for sm_80 without a GPU and read its SASS.

Used by the sm_80 kernel tests to prove which instructions a kernel variant
contains (for example that no scalar ``F2F.BF16.F32`` narrowing is left).
``triton.compile`` with an explicit target needs no device; ``cuobjdump`` is
the one Triton ships. Arguments follow the JIT's specialisation: a string
``"*fp32"`` is a 16-byte aligned pointer, an int equal to 1 becomes the
constexpr 1, an int divisible by 16 gets the divisibility hint, a 1-tuple
``(v,)`` is a constexpr.
"""

import collections
import os
import re
import subprocess
import tempfile

_LINE = re.compile(r"/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P\w+\s+)?([A-Z0-9_.]+)")


def _cuobjdump():
    import triton

    path = os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia",
                        "bin", "cuobjdump")
    return path if os.path.exists(path) else None


def available() -> bool:
    return _cuobjdump() is not None


def compile_sm80(kernel, args: dict, num_warps: int = 4, num_stages: int = 1):
    """Return (sass text, ptx text, opcode Counter) of ``kernel`` for sm_80."""
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    fn = getattr(kernel, "fn", kernel) if not hasattr(kernel, "arg_names") else kernel
    sig, const, attrs = {}, {}, {}
    for i, name in enumerate(fn.arg_names):
        v = args[name]
        if isinstance(v, tuple):
            sig[name] = "constexpr"
            const[name] = v[0]
        elif isinstance(v, str):
            sig[name] = v
            attrs[(i,)] = [["tt.divisibility", 16]]
        elif isinstance(v, bool):
            sig[name] = "constexpr"
            const[name] = v
        elif isinstance(v, int):
            if v == 1:
                sig[name] = "constexpr"
                const[name] = 1
            else:
                sig[name] = "i64" if abs(v) >= 2**31 else "i32"
                if v % 16 == 0 and v != 0:
                    attrs[(i,)] = [["tt.divisibility", 16]]
        elif isinstance(v, float):
            sig[name] = "fp32"
        else:
            raise TypeError((name, v))
    src = ASTSource(fn=fn, signature=sig, constexprs=const, attrs=attrs)
    k = triton.compile(src, target=GPUTarget("cuda", 80, 32),
                       options={"num_warps": num_warps, "num_stages": num_stages})
    with tempfile.TemporaryDirectory() as d:
        cub = os.path.join(d, "k.cubin")
        with open(cub, "wb") as f:
            f.write(k.asm["cubin"])
        sass = subprocess.run([_cuobjdump(), "-sass", cub], capture_output=True,
                              text=True, check=True).stdout
    ops = collections.Counter(m.group(1) for m in map(_LINE.search, sass.splitlines()) if m)
    return sass, k.asm["ptx"], ops


def count(ops: collections.Counter, prefix: str) -> int:
    """Number of instructions whose opcode is ``prefix`` or starts with ``prefix.``."""
    return sum(n for op, n in ops.items() if op == prefix or op.startswith(prefix + "."))
