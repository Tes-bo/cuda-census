# cuda-census

The same CUDA source, compiled and run by each vendor's own tools, then diffed against `nvcc`. This repository is not a converter.

[![License](https://img.shields.io/github/license/Tes-bo/cuda-census?style=flat-square&color=blue)](LICENSE)
[![Gold](https://img.shields.io/badge/gold-nvcc%20stdout-1a7f37?style=flat-square)](gold/provenance.json)
[![Toolchains](https://img.shields.io/badge/toolchains-official%20only-0969da?style=flat-square)](toolchains.json)

[cuda-retarget](https://github.com/Tes-bo/cuda-retarget) is a separate rewriter. It is not a column in this table. A path whose name is `cuda-retarget` is refused.

<p align="center">
  <img src="docs/overview.svg" alt="One CUDA corpus is compiled by nvcc, hipify, musify, and cu-bridge. Vendor integers are diffed against nvcc gold." width="820">
</p>

**Quick start · [Statuses](#statuses) · [Corpus](#corpus) · [Run](#run)**

A green cell means that vendor's tool compiled the case, ran it, and printed the same integers as the committed `nvcc` stdout. Vendor results are not compared with each other.

cuda-census is not affiliated with NVIDIA, AMD, Hygon, Moore Threads, or MetaX. CUDA names the source language. The vendor names name the official tools under test.

## Quick start

Python 3.10 or newer.

```bash
python3 -m census self-check
python3 -m census list
python3 -m census run --toolchain nvcc
python3 -m census run --toolchain hipify
python3 -m census run --toolchain musify
python3 -m census run --toolchain cu-bridge
```

`run` exits 0 when every selected case is `pass`. It exits 2 when the only failures are missing tools. Any compile, run, or numeric failure exits 1.

Each case prints `name=integer` lines and nothing else. The runner writes `results/<toolchain>/<case>.json`. `results/` is not committed.

## Statuses

| Status | Meaning |
| --- | --- |
| `pass` | The program ran and its integers match `gold/<case>.stdout`. |
| `numeric-mismatch` | The program ran and at least one integer differs from gold. The diff is actual minus gold. |
| `compile-fail` | Translate or compile returned non-zero. |
| `run-fail` | The binary returned non-zero. |
| `unparsed-stdout` | Stdout was not a set of `name=integer` lines. |
| `gold-missing` | A vendor toolchain ran, but `gold/` has no `nvcc` stdout for that case. |
| `toolchain-missing` | The official tool is not on `PATH`. |

`gold/` is produced only by `nvcc`. `--write-gold` is rejected for every other toolchain.

The committed gold was produced by CUDA 12.0 (`Cuda compilation tools, release 12.0, V12.0.140`) on a Tesla V100-SXM2-32GB. `gold/provenance.json` records that run. HIPIFY, musify, and cu-bridge are wired in `toolchains.json` and were not executed in this tree, because those tools are not installed here. A machine that has them should run the same command and publish the JSON.

## Corpus

34 cases. The gold column is the `nvcc` stdout.

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
| `packed_abs` | per-lane abs, negation, and signed versus unsigned absolute difference | `abs2=2147483650`, `absss2=2147418114`, `neg2=2147549182`, `negss2=2147549182`, `negss4=2147451263`, `diffs2=2147483650`, `diffu2=2147549182` |
| `byte_perm` | `__byte_perm` byte select; high selector bits are ignored | `id=2712847316`, `fromy=523124044`, `high=523124044`, `nibble=3569595041`, `zero=3570717908` |
| `average` | `__hadd` / `__rhadd` / `__uhadd`, rounding and overflow | `h12=1`, `r12=2`, `hneg=-3`, `rneg=-2`, `hmin=-2147483648`, `uhwrap=2147483648` |
| `mulhi` | high half of a signed or unsigned product, 32-bit and 64-bit | `min1=-1`, `minneg=0`, `minsq=1073741824`, `umax=4294967294`, `neg64=-1`, `sq64=4611686018427387904`, `umax64=18446744073709551614` |
| `sad` | `__sad` / `__usad`, widened absolute difference plus a wrapping add | `sneg=2`, `smin=2147483648`, `smin1=2147483649`, `swrap=0`, `umax=4294967295`, `uwrap=0` |
| `packed_sad` | sum of per-lane absolute differences, signed versus unsigned | `s2=2`, `u2=65536`, `smin=65536`, `pair=3`, `s4=131`, `u4=639` |
| `packed_cmp` | per-lane comparison masks, signed versus unsigned | `eq=4294901760`, `ne=65535`, `gts=65535`, `gtu=4294967295`, `lts=4294901760`, `gts4=255` |
| `packed_set` | per-lane predicates of 1, not an all-ones mask | `eq=65536`, `ne=1`, `gts=1`, `gtu=65537`, `gts4=1`, `gtu4=16777473` |
| `packed_avg` | packed averages, round-half-away-from-zero versus truncation | `spos=131074`, `sneg=4294836221`, `smin=2147516416`, `uround=131074`, `uplain=65537`, `uhalf=2147516416` |
| `bit_scan` | leading zeros, first set bit, population count, and bit reverse | `clz0=32`, `clzmin=0`, `ffs0=0`, `ffsmin=32`, `brev1=2147483648`, `clzll0=64`, `popc=32`, `brevll=9223372036854775808` |
| `mul24` | 24-bit product, high byte ignored, low 32 bits kept | `hi=2`, `sign=-8388608`, `neg1=1`, `sq=0`, `usq=4261412865`, `uhi=16777216` |
| `round_cvt` | float and 32-bit integer conversions, ties and saturation | `rn25=2`, `ru25=3`, `rdn=-3`, `big=2147483647`, `nan=0`, `ubig=4294967295`, `up=1266679809`, `nrd=3414163458` |
| `round_wide` | 64-bit and double conversions; NaN is the high bit | `llnan=-9223372036854775808`, `llbig=9223372036854775807`, `ullnan=9223372036854775808`, `llru=1509949441`, `dnan=-2147483648`, `dtrunc=1`, `dull=18446744073709551615` |
| `saturatef` | clamp to [0, 1]; NaN and negative zero become +0 | `neg=0`, `two=1065353216`, `half=1056964608`, `mzero=0`, `nan=0` |
| `farith` | directed float add, sub, mul, and fused multiply-add | `addru=1065353217`, `zrd=2147483648`, `zsame=0`, `addmax=2139095039`, `fmacancel=679477248`, `fmanan=2147483647` |
| `dround` | double to float and to integers, with a chosen rounding mode | `ftie=1065353216`, `ftru=1065353217`, `fsubru=1`, `fmax=2139095039`, `fnan=4290772992`, `fru=1266679809`, `inan=-2147483648` |
| `half_cvt` | float to half and half back to float | `halftie=15360`, `halfru=15361`, `halfsubru=1`, `halfmax=31743`, `halfnan=32767`, `halfmzero=32768`, `fnan=2147483647`, `finf=4286578688` |
| `graph_replay` | Stream capture, instantiate, replay | `sum=32` |
| `graph_async_alloc` | `cudaMallocAsync` / `cudaFreeAsync` inside a capture | `sum=32` |

## Run

Regenerate gold only from a real `nvcc` run, and only when the printed integers still match `corpus/manifest.json`:

```bash
python3 -m census run --toolchain nvcc --write-gold
```

Host-side tests, with no GPU:

```bash
python3 -m unittest discover -s tests -v
```

### Toolchain command lines

`toolchains.json` is the command line that will be executed. No other translator is consulted.

| Toolchain | What runs |
| --- | --- |
| `nvcc` | compiles the corpus file |
| `hipify` | `hipify-perl`, then `hipcc`. On a Hygon machine, `hipcc` is the DTK compiler on `PATH` |
| `musify` | copy the file, `musify-text --inplace`, then `mcc` |
| `cu-bridge` | no rewrite. `cucc` compiles the original `.cu` |

Override a binary by putting the official tool on `PATH`.

## Adding a case

Add one directory under `corpus/<id>/` with a single `main.cu` that prints integer lines. Add the source path and the expected integers to `corpus/manifest.json`. Produce gold with `nvcc --write-gold` on NVIDIA hardware. Do not commit a gold file from any other compiler.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
