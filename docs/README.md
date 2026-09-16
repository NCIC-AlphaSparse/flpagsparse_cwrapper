# 后端文档索引

三处文档，各回答一个问题：`docs/<BACKEND>.md` 怎么在那台机器上跑、
`capi/docs/<BACKEND>.md` C API 那一层、[`modified/<BACKEND>.md`](../modified/README.md)
这轮在实机上改了哪些文件（改动台账，省得后端文件来回传）。

**一个后端一个文件。** Python 侧在 `docs/`，C API 侧在 `capi/docs/`，同名对应：
`docs/MUSA.md` 讲怎么在摩尔线程上跑 Python/Triton 这套，`capi/docs/MUSA.md` 讲 C API
那一层。跨引用都按这个名字走。

| 后端名（`FLAGSPARSE_BACKEND`） | 厂商 / 平台 | Python 侧 | C API 侧 |
|---|---|---|---|
| `cuda` | NVIDIA | 见仓库根 `README.md` | [`capi/docs/CUDA.md`](../capi/docs/CUDA.md) |
| `rocm` | 海光 DCU / ROCm | [`DCU.md`](DCU.md) | [`capi/docs/DCU.md`](../capi/docs/DCU.md) |
| `metax` | 沐曦 MetaX / MACA | [`MACA.md`](MACA.md) | [`capi/docs/MACA.md`](../capi/docs/MACA.md) |
| `mthreads` | 摩尔线程 / MUSA | [`MUSA.md`](MUSA.md) | [`capi/docs/MUSA.md`](../capi/docs/MUSA.md) |
| `ascend` | 昇腾 / CANN | [`ASCEND.md`](ASCEND.md) | [`capi/docs/ASCEND.md`](../capi/docs/ASCEND.md) |
| `xpu` | 昆仑芯 | [`XPU.md`](XPU.md) | [`capi/docs/XPU.md`](../capi/docs/XPU.md) |
| `gcu` | 燧原 | — 尚无 | — 尚无 |
| `mlu` | 寒武纪 | — 尚无 | — 尚无 |

最后两行是**已注册但没有任何文档**的槽位：`_BACKEND_NAMES` 和 `_dispatch.py` 的
`_IMPLEMENTATION_BACKENDS` 都是 8 条，`backends/gcu`、`backends/mlu` 目录已建好但代码留空，
走共享实现。没有覆盖不等于不能跑，只是没人在真机上验过，也就没有可写的文档。

**`XPU.md` 与另外四份的性质不同**：DCU / MACA / MUSA / ASCEND 都有实机验证结果，
XPU 那份写的是已接好的槽位和首次上机的步骤，**没有实机记录**，文件顶部标了出来。

`cuda` 没有单独文件是因为它是参照路径 —— 环境、跑法、算子清单都在仓库根的
`README.md` / `README_cn.md` 里，后端文档只记"与 CUDA 不同的地方"。

## 每个文件里有什么

| 文件 | 内容 |
|---|---|
| [`DCU.md`](DCU.md) | ROCm 环境、诊断优先的排查顺序、hipSPARSE 基线覆盖范围、SpMV 走 rowpar 的原因、已知限制 |
| [`MACA.md`](MACA.md) | C550 bring-up（FlagTree metax 后端、环境指纹）、921 用例的验证范围、SpSV 的两个缺陷、SpMM COO 复数的 4 KB 私有内存上限、调优 A/B |
| [`MUSA.md`](MUSA.md) | 独立设备类型 `musa` 与兼容后端的区别、`_ACCEL` 抽象、实测能力矩阵（muDNN 的 gemv 缺口）、normal 回归结果、已解决问题的复现记录 |
| [`ASCEND.md`](ASCEND.md) | 910B 环境检查、Ascend fallback 分发表、算子能力探测、已知限制 |
| [`XPU.md`](XPU.md) | 昆仑芯：插件探测为什么不能只看 `torch.xpu`、probe-only 的性能路径、为什么没有厂商基线、首次上机顺序（**未实机验证**） |
