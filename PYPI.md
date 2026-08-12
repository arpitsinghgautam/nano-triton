# nano-triton

**A from-scratch nano-Triton (`newt`) and nano-Helion (`deuteron`):** the modern
GPU-kernel DSL stack, rebuilt in ~4,000 lines of readable Python. Measured on
three GPUs, it holds memory-bandwidth parity with real Triton and reaches 70-88%
of its tensor-core matmul throughput on discrete Blackwell parts. No
dependencies beyond torch.

![nano-triton](https://arpitsinghgautam.me/nano-triton/assets/og-card.png)

- **newt** is a nano-Triton: `@newt.jit` kernels compile Python AST to CUDA C++
  to NVRTC to a cubin, launched through the raw CUDA driver API via ctypes. No
  MLIR, no LLVM, no nvcc.
- **deuteron** is a nano-Helion: write PyTorch-like tile code with no kernel
  details; it generates a newt kernel and autotunes it against an eager-PyTorch
  correctness oracle.

## Install

```
pip install nano-triton
```

Requires `torch` and an NVIDIA GPU with the CUDA toolkit (newt uses NVRTC to
compile kernels at runtime). Installing the package gives you both `import newt`
and `import deuteron`.

## Quick start

```python
import torch
import newt
import newt.language as nl

@newt.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: nl.constexpr):
    pid = nl.program_id(0)
    offs = pid * BLOCK + nl.arange(0, BLOCK)
    mask = offs < n
    x = nl.load(x_ptr + offs, mask=mask)
    y = nl.load(y_ptr + offs, mask=mask)
    nl.store(out_ptr + offs, x + y, mask=mask)

x, y = torch.randn(2, 1_000_000, device="cuda")
out = torch.empty_like(x)
add_kernel[lambda m: (newt.cdiv(1_000_000, m["BLOCK"]),)](
    x, y, out, 1_000_000, BLOCK=1024)
```

## Benchmarks

Three GPUs (RTX PRO 5000 Blackwell laptop, RTX 5090, GB10 Grace Blackwell),
same kernel source and tuning sweep for newt and real Triton. fp16 tensor-core
matmul, newt as a percentage of Triton:

| newt as % of Triton | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|
| RTX PRO 5000 laptop | 88 | 76 | 76 | 79 |
| RTX 5090 | 76 | 73 | 71 | 79 |
| GB10 | 69 | 66 | 55 | 67 |

Peak measured throughput is 169 TFLOP/s on the RTX 5090. Memory-bound kernels
(softmax, layernorm, elementwise) sit within 2% of Triton on all three devices,
geomean 100.4%. tf32 matmul is the weak path at roughly 40-45%, since it still
uses WMMA rather than `mma.sync`.

## Docs and source

- Full writeup, from-zero explainer, benchmarks, glossary: https://arpitsinghgautam.me/nano-triton/
- Source and issues: https://github.com/arpitsinghgautam/nano-triton

MIT licensed.
