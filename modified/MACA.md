# 沐曦 MetaX / MACA —— 改动台账

**实机环境**：MetaX C550（`warp_size=64`、104 MP、64 GB），MACA SDK 3.8.2.6，
torch `2.10.0+metax3.8.1.0`，triton `3.6.0+metax3.8.1.0`，Python 3.12。
**基于版本**：上游 FlagSparse。
**最近回传**：2026-09-11（`/workspace/fork/metax`）。

跑法、921 用例的验证范围、SpSV 缺陷见 [`docs/MACA.md`](../docs/MACA.md)；
C API 那层见 [`capi/docs/MACA.md`](../capi/docs/MACA.md)。

---

## 1. 库内的后端分支（4 个文件，共 11 处）

| 文件 | 处数 | 锚点 | 改动 | 为什么 |
|---|---:|---|---|---|
| `sparse_operations/spsv.py` | 4 | `_MACA_SPSV_PROFILES` + `_maca_spsv_knob()` | 按**设备型号**取 knob：`alg3_warp_size` / `alg4_warp_size` / `enable_advanced_auto` / `cw_serial` / `smblk_persistent` | 只有 `warp_size=64` 是实测回填的；其余四个仍是 CUDA 平移值 |
| `sparse_operations/spmv_csr.py` | 2 | `_MACA_SPMV_PROFILES` | `csr_kernel` 默认 `segbin`（CUDA 版） | 未做 C550 A/B，DCU 实测是 `rowpar`，孰优未定 |
| `sparse_operations/spmm_coo.py` | 1 | `_MACA_SPMM_COO_COMPLEX_BLOCK_NNZ = 4` + `_resolve_spmm_coo_launch_config(value_dtype=...)` | MACA + 复数时把 `BLOCK_NNZ` 从 256 钳到 4 | 复数 SpMM COO **24 个用例全失败**：`tl.static_range(0, BLOCK_NNZ)` 是 constexpr、展开 256 次，复数带实虚两份，每线程私有内存要 8 KB，而系统上限 4 KB |
| `sparse_operations/spmm_csr.py` | 4 | 启动参数里的 `"is_maca"` 标志等 | 按后端选 launch 配置 | |

钳制那条的收益值得单独记：**51/51 通过，整组耗时从 23m48s 降到 11.8s** —— 256 的展开开销
就有这么大。实数和不传 dtype 的调用仍解析到 256，其他后端行为不变。

> 30 矩阵扫描测出的最优值本来就是 4，256 是被测量否定过的旧值，只是当初调优没改到函数
> 签名的默认参数。要全局生效，把公开默认值改成 `None` 即可。

## 2. 测试侧

| 文件 | 锚点 | 改动 | 为什么 |
|---|---|---|---|
| `tests/test_spgemm.py` | `isolate_matrices=ast_common._is_maca_runtime()` | MACA 上逐矩阵隔离进程 | 内核 fault 之后厂商 runtime 被污染，同进程后续结果全是垃圾 |
| `tests/test_spmv.py` | `_is_maca_runtime()` 分支 | 同上类别 | |

## 3. 已定位但**未修**的缺陷

**`_spsv_csr_cw_kernel`（ALG1 / `csr_cw`）在 C550 上有两个独立问题**，代码未改，只在文档里
记了现象和已排除项：

| 结构（n=8、fp32、unit 对角） | lower | upper |
|---|---|---|
| 仅对角（无依赖） | **非法访存** | OK |
| 双对角 / 稠密三角 | **非法访存** | **挂死** |

1. `LOWER=True` 特化在**零依赖**时就非法访存。单位阵对称、两次运行数据与控制流完全相同、
   依赖分支从不进入，只有 `LOWER`/`REVERSE_ORDER` 不同，而只有 lower 崩 ——
   **这是编译层面的差异**，不是 Python 侧索引算错；
2. ready-flag 自旋不推进（upper + 任意依赖），与 DCU/gfx936 同一故障模式。

**已排除**：`alg3/alg4_warp_size`（32、64 表现相同）、`cw_serial=True`（worker_count=1 仍崩）、
矩阵规模（n=2..64 全崩）、`enable_advanced_auto`（unit 对角下不可达）。

排查这个内核**必须**加 `CUDA_LAUNCH_BLOCKING=1` —— 非法访存异步上报，默认会在下一次同步
（通常是 `allclose`）才抛，traceback 指向完全无关的位置。

## 4. 仍开着

- **四个 SpSV knob 仍是 CUDA 平移值**，需要按 `docs/MACA.md` 第 8 节的 A/B 命令实测回填。
  它们是**行为选择**而非硬件事实，不能靠指纹推断。
- **`alpha_spmm_alg1` 缺 TLE**：沐曦的 triton 没有 `triton.experimental.tle`，
  FlagOS 的 flagtree（带 TLE）要 GLIBC 2.38，实机是 2.31。只影响这一个算子。
- **C API 侧的 mcSPARSE 基线未接** —— 名字（`mcsparse` / `<mcsparse.h>`）是按命名惯例推的、
  **未经实机验证**，CMake 探测不到就回落 `baseline/none`。与摩尔线程不同：muSPARSE 那边
  名字已实机核对、原生实现也已合入。
