# cuda-census

Same CUDA source on vendor toolchains: compile, run, and numeric diff for HIP (AMD, Hygon), MUSA (musify), and MACA (cu-bridge). Official tools only.

This repository records what happens when one corpus is built with the toolchain that vendor ships. It is not a converter, and it does not call [cuda-retarget](https://github.com/Tes-bo/cuda-retarget). A green cell means that vendor's own tool compiled the case, ran it, and printed the same integers as the committed `nvcc` stdout.

cuda-census is not affiliated with NVIDIA, AMD, Hygon, Moore Threads, or MetaX. CUDA names the source language of the corpus. The vendor names above name the official tools under test.

## What is recorded

Each case prints `name=integer` lines on stdout and nothing else. The runner writes `results/<toolchain>/<case>.json` with one of these statuses:

| Status | Meaning |
| --- | --- |
| `pass` | The program ran and its integers match `gold/<case>.stdout`. |
| `numeric-mismatch` | The program ran and at least one integer differs from gold. |
| `compile-fail` | Translate or compile returned non-zero. |
| `run-fail` | The binary returned non-zero. |
| `unparsed-stdout` | Stdout was not a set of `name=integer` lines. |
| `gold-missing` | A vendor toolchain ran, but `gold/` has no `nvcc` stdout for that case. |
| `toolchain-missing` | The official tool is not on `PATH`. |

`gold/` is produced only by `nvcc`. Vendor toolchains are compared with that file. They are not compared with each other, and `--write-gold` is rejected for every toolchain except `nvcc`.

## Corpus

| Case | What it exercises | Gold |
| --- | --- | --- |
| `vector_add_int` | Runtime API, launch, device memcpy | `sum=8256` |
| `warp_shuffle` | `__shfl_sync` across one warp | `sum=160` |
| `warp_size` | the `warpSize` builtin | `width=32` |
| `match_any` | `__match_any_sync` / `__match_all_sync` across two warps | `lane0=1`, `lane32=1`, `all=0`, `pred=0` |
| `warp_vote` | `__ballot_sync`, `__any_sync`, `__all_sync`, `__uni_sync` across two warps | `lane0=4294967295`, `any0=1`, `all0=1`, `uni0=1`, `lane32=0`, `any32=0`, `all32=0`, `uni32=1` |
| `warp_reduce` | `__shfl_xor_sync` butterfly in each of two warps | `lane0=496`, `lane32=1520` |
| `syncwarp` | no-argument `__syncwarp()` then a shared-memory sum | `sum=496` |
| `shared_atomic` | `atomicAdd` on shared memory | `total=128` |
| `double_as_longlong` | `__double_as_longlong(1.0)` | `bits=4607182418800017408` |
| `double_hi_lo` | `__double2hiint` / `__double2loint` / `__hiloint2double` of 1.0 | `hi=1072693248`, `lo=0`, `round=1` |
| `float_as_int` | `__float_as_int(1.0f)` | `bits=1065353216` |
| `funnelshift` | `__funnelshift_l` / `_lc` / `_r` / `_rc` wrap versus clamp | `l1=2021161081`, `l32=1010580540`, `lc32=2779096485`, `r1=1389548242`, `rc32=1010580540` |
| `dp4a` | signed and unsigned `__dp4a` four-byte dot products | `ones=4`, `s=2`, `s80=-512`, `u=258`, `u80=512` |
| `packed_add` | `__vadd2` / `__vsub2` / `__vadd4` / `__vsub4`, wrap versus signed and unsigned saturation | `add2=2147483648`, `addss2=2147418112`, `addus2=2147549183`, `subss2=2147549183`, `add4=2147647744`, `addss4=2130870528`, `addus4=2147647999` |
| `packed_minmax` | `__vmins2` / `__vminu2` / `__vmaxs2` / `__vmaxu2` and the 4-byte forms | `mins2=2147483648`, `minu2=2147418112`, `maxs2=2147418113`, `maxu2=2147483649`, `mins4=2147581952`, `minu4=2130804480` |
| `graph_replay` | Stream capture, instantiate, replay | `sum=32` |
| `graph_async_alloc` | `cudaMallocAsync` / `cudaFreeAsync` inside a capture | `sum=32` |

The committed gold files were produced by `nvcc` from CUDA 12.0 on a Tesla V100-SXM2-32GB. `gold/provenance.json` records that run. HIPIFY, musify, and cu-bridge are wired in `toolchains.json` and were not executed in this tree, because those tools are not installed here. A machine that has them should run the same command and publish the JSON.

## Run

```bash
python3 -m census self-check
python3 -m census list
python3 -m census run --toolchain nvcc
python3 -m census run --toolchain hipify
python3 -m census run --toolchain musify
python3 -m census run --toolchain cu-bridge
```

`run` exits 0 when every selected case is `pass`. It exits 2 when the only failures are missing tools. Any compile, run, or numeric failure exits 1.

Regenerate gold only from a real `nvcc` run, and only when the printed integers still match `corpus/manifest.json`:

```bash
python3 -m census run --toolchain nvcc --write-gold
```

Host-side tests, with no GPU:

```bash
python3 -m unittest discover -s tests -v
```

## Toolchain command lines

`toolchains.json` is the command line that will be executed. No other translator is consulted.

- `nvcc` compiles the corpus file.
- `hipify` runs `hipify-perl` and compiles the translated file with `hipcc`. On a Hygon machine, `hipcc` is the DTK compiler on `PATH`.
- `musify` copies the file, runs `musify-text --inplace`, and compiles with `mcc`.
- `cu-bridge` does not rewrite source. `cucc` compiles the original `.cu`.

Override a binary by putting the official tool on `PATH`. A path whose name is `cuda-retarget` is refused.

## Adding a case

Add one directory under `corpus/<id>/` with a single `main.cu` that prints integer lines. Add the source path and the expected integers to `corpus/manifest.json`. Produce gold with `nvcc --write-gold` on NVIDIA hardware. Do not commit a gold file from any other compiler.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
