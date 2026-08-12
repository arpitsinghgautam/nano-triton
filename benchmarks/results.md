# newt benchmarks

Three GPUs, chosen to vary the axes that a single-machine comparison would
confound: compute capability, host architecture, and memory system.

| | RTX PRO 5000 Blackwell Laptop | RTX 5090 | GB10 (DGX Spark) |
|---|---|---|---|
| compute capability | sm_120 | sm_120 | sm_121 |
| SMs | - | 170 | 48 |
| memory | 24 GB GDDR7 | 31.3 GB GDDR7 | 121.6 GB LPDDR5X, unified |
| power | ~110 W, throttles | desktop, no observed throttling | unified SoC |
| host | Windows 11, x86_64 | Ubuntu 24.04, x86_64 | Ubuntu 24.04, aarch64 |
| driver CUDA | 13.2 | 13.2 | 13.0 |
| torch | 2.11.0+cu128 | 2.11.0+cu128 | 2.11.0+cu128 |
| triton | triton-windows | 3.6.0 | 3.6.0 |

Method: `python benchmarks/bench.py --cooldown 300`, identical on all three.
Medians over 50 reps, L2 flushed between reps, same kernel source and same
config sweep for newt and triton. Tests pass on all three devices (176/176 on
the laptop and the 5090; 175/176 on GB10, where the one failure is on the torch
reference side, see Portability below).

## Headline

- **Memory-bound kernels are at parity with Triton on every device**: geomean
  100.4% over the 33 bandwidth-bound cells, always within 2%, and marginally
  ahead on average. newt records the highest measured streaming bandwidth of the
  three frameworks on both the laptop and the 5090.
- **fp16 tensor-core matmul reaches 70-88% of Triton on the discrete Blackwell
  GPUs** (geomean 77%) and 55-69% on GB10. Across all three devices and four
  sizes the geomean is 72%.
- **tf32 matmul is the weak path at roughly 40-45% of Triton**, consistently on
  all three devices, because it still uses WMMA rather than the `mma.sync` path
  used for fp16.
- **The gap to Triton is architectural, not thermal.** See below.

## The gap is not thermal

The obvious hypothesis for the laptop numbers was thermal throttling. Removing
the power limit refutes it. Going from the 110 W laptop to an unconstrained RTX
5090 roughly doubled absolute throughput, from 87.6 to 169.2 TFLOP/s, and the
ratio to Triton did not improve:

| fp16 matmul, newt as % of Triton | 1024³ | 2048³ | 4096³ | 8192³ |
|---|---|---|---|---|
| RTX PRO 5000 laptop (110 W) | 87.7 | 76.3 | 75.8 | 78.6 |
| RTX 5090 (no power limit) | 76.4 | 72.6 | 70.9 | 78.9 |
| GB10 (48 SMs, unified memory) | 69.2 | 66.0 | 55.0 | 66.7 |

A cold-start run on the 5090, after 600 s of idle, reproduces the sustained run
within noise (170.2 vs 169.2 TFLOP/s at 8192³), so there is no separate cold
regime on a device that does not throttle. An earlier laptop-only measurement
reported roughly 92% of Triton on a cold start; that figure was an artifact of
how the two compilers' baselines decay on a thermally limited part and does not
reproduce off the laptop.

## Absolute results

### RTX 5090

```
vector add fp32 (GB/s)         torch |  newt | triton
1M                              1281 |  1177 |  1285
16M                             1530 |  1531 |  1530
64M                             1561 |  1565 |  1562
128M                            1568 |  1568 |  1567

fused softmax fp32 (GB/s)      torch |  newt | triton
4096x1024                       1638 |  1489 |  1489
4096x4096                       1481 |  1524 |  1516
4096x8192                       1507 |  1507 |  1524
4096x16384                      1509 |  1524 |  1515

layernorm fwd fp32 (GB/s)      torch |  newt | triton
4096x1024                       1239 |  1394 |  1371
4096x2048                       1365 |  1503 |  1489
4096x4096                       1359 |  1524 |  1524
4096x8192                       1097 |  1511 |  1502

matmul float16 (TFLOP/s)       torch |  newt | triton
1024^3                         105.5 |  80.2 |  105.0
2048^3                         145.2 | 114.9 |  158.3
4096^3                         207.1 | 152.2 |  214.7
8192^3                         216.9 | 169.2 |  214.5

matmul fp32/tf32 (TFLOP/s)     torch |  newt | triton
1024^3                          65.5 |  21.4 |   61.6
2048^3                          79.1 |  31.5 |   83.9
4096^3                         105.4 |  42.2 |  103.8
8192^3                         114.6 |  47.2 |  108.8

matmul float16, cold start (600 s pre-cooldown)
1024^3                         105.4 |  80.6 |  105.0
2048^3                         145.3 | 114.9 |  161.3
4096^3                         204.0 | 153.0 |  215.1
8192^3                         216.1 | 170.2 |  215.6
```

### RTX PRO 5000 Blackwell Laptop

```
vector add fp32 (GB/s)         torch |  newt | triton
1M                               599 |   623 |   703
16M                              636 |   639 |   635
64M                              645 |   787 |   780
128M                             762 |   787 |   766

fused softmax fp32 (GB/s)      torch |  newt | triton
4096x1024                         37 |   634 |   634   [a]
4096x4096                        619 |   763 |   767
4096x8192                        765 |   769 |   765
4096x16384                       758 |   767 |   766

layernorm fwd fp32 (GB/s)      torch |  newt | triton
4096x1024                        656 |   756 |   754
4096x2048                        719 |   765 |   769
4096x4096                        745 |   764 |   764
4096x8192                        626 |   769 |   763

matmul float16 (TFLOP/s)       torch |  newt | triton
1024^3                          72.9 |  66.5 |   75.8
2048^3                         100.7 |  83.7 |  109.7
4096^3                          67.1 |  87.6 |  115.5   [a]
8192^3                         102.3 |  81.3 |  103.4

matmul fp32/tf32 (TFLOP/s)     torch |  newt | triton
1024^3                          51.4 |  23.9 |   44.1
2048^3                          66.1 |  23.9 |   55.2
4096^3                          57.9 |  23.2 |   57.6
8192^3                          59.3 |  21.4 |   52.4
```

### GB10 (DGX Spark)

```
vector add fp32 (GB/s)         torch |  newt | triton
1M                               150 |   150 |   150
16M                              232 |   235 |   234
64M                              238 |   240 |   237
128M                             238 |   241 |   238

fused softmax fp32 (GB/s)      torch |  newt | triton
4096x1024                        211 |   216 |   218
4096x4096                        241 |   239 |   239
4096x8192                        244 |   242 |   240
4096x16384                       243 |   243 |   240

layernorm fwd fp32 (GB/s)      torch |  newt | triton
4096x1024                        210 |   216 |   217
4096x2048                        218 |   231 |   232
4096x4096                        224 |   238 |   238
4096x8192                        174 |   242 |   238

matmul float16 (TFLOP/s)       torch |  newt | triton
1024^3                           9.6 |  27.9 |   40.3   [b]
2048^3                          12.5 |  45.2 |   68.5   [b]
4096^3                          12.1 |  28.4 |   51.6   [b]
8192^3                          10.9 |  23.6 |   35.4   [b]

matmul fp32/tf32 (TFLOP/s)     torch |  newt | triton
1024^3                          18.9 |  10.8 |   23.3
2048^3                          33.4 |  15.4 |   34.3
4096^3                          38.9 |  14.3 |   29.6
8192^3                          41.5 |  11.7 |   26.6
```

## Ratios

### Memory-bound, newt as % of Triton

Over the 33 bandwidth-bound cells: range 98.9 to 102.7, geomean **100.4**.
Including the three 1M vector-add points, which are launch-latency-bound rather
than bandwidth-bound, the range widens to 88.6 to 102.7 (geomean 99.8).

### tf32 matmul, newt as % of Triton

| | 1024³ | 2048³ | 4096³ | 8192³ |
|---|---|---|---|---|
| laptop | 54.2 | 43.3 | 40.3 | 40.8 |
| RTX 5090 | 34.7 | 37.5 | 40.7 | 43.4 |
| GB10 | 46.4 | 44.9 | 48.3 | 44.0 |

Geomean 42.9. On the laptop newt's tf32 throughput is flat at 21.4 to 23.9
TFLOP/s across a 512x range in problem size while Triton scales from 44.1 to
57.6, which is the signature of the WMMA path rather than a tuning problem.

## Portability

newt compiled and ran correctly on GB10 (sm_121), a compute capability that
postdates the compiler, with no source changes. Two observations from that run:

1. `tests/test_ops.py::test_unary[exp2]` fails on GB10, and the failure is on
   the reference side rather than in newt. `torch.exp2` is jiterator-backed and
   compiles at runtime through torch's bundled NVRTC 12.8, which predates
   sm_121, giving "nvrtc: error: invalid value for --gpu-architecture". newt
   resolves NVRTC through the bare `libnvrtc.so` soname and so picks up the
   system CUDA 13.0, which supports the target.
2. torch's fp16 matmul on GB10 runs at 9.6 to 12.5 TFLOP/s, 2 to 6x below both
   newt and Triton at every size. Two independent code generators agreeing rules
   out a newt artifact. This is consistent with the cu128 wheel shipping cuBLAS
   12.8.4.1, which also predates sm_121, though the mechanism is a hypothesis
   rather than something measured directly.

## Footnotes and threats to validity

- **[a]** Two torch cells on the laptop are measurement anomalies and are
  excluded from any newt-versus-torch statistic: softmax 4096x1024 at 37 GB/s
  (17 to 21x below its own neighbours) and matmul 4096³ at 67.1 TFLOP/s (a
  non-monotonic collapse between 100.7 at 2048³ and 102.3 at 8192³). The
  newt-versus-Triton cells at those points are unaffected.
- **[b]** The entire torch fp16 matmul column on GB10 is excluded from
  newt-versus-torch comparison for the reason given under Portability.
- **No thermal telemetry on the RTX 5090.** NVML was broken on that host by a
  driver userspace and kernel-module mismatch requiring a reboot. CUDA was
  unaffected and all 176 tests passed. The evidence that it did not throttle is
  circumstantial: the cold run reproduces the sustained run within noise.
- **The GB10 Triton baseline required user-local Python headers.** GB10 had no
  `python3-dev` and no passwordless sudo, so Triton could not compile its
  launcher shim. The headers were extracted into a user prefix, at the same
  package version as the 5090's, with nothing modified system-wide. An earlier
  GB10 run without a Triton baseline reproduces the torch and newt columns to
  within about 1% on memory-bound kernels and 2 to 6% on fp16 matmul.
- `bench.py` previously cooled only between suites and not before the first, so
  the opening suite could start on a warm GPU. That is fixed; `--cooldown` now
  idles before every suite including the first.
