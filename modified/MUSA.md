# 摩尔线程 MUSA —— 改动台账

**实机环境**：MTT S5000 ×1 / torch 2.7.1 / torch_musa 2.7.1 / muDNN v3105 / Triton 3.6.0。
**基于版本**：合并后的 `fs_merge`（2026-09-16 上午）。
**最近回传**：2026-09-16 16:39 + 16:51 + 16:57，共 12 个文件，**已全部合入**。

跑法、能力矩阵、排障记录见 [`docs/MUSA.md`](../docs/MUSA.md)；C API 那层见
[`capi/docs/MUSA.md`](../capi/docs/MUSA.md)。本文只记改了什么。

---

## 1. 库内的后端分支（3 个文件，共 4 处）

| 文件 | 锚点 | 改动 | 为什么 | 其他后端 |
|---|---|---|---|---|
| `sparse_operations/spgemm_csr.py` | `tle_hash_safe` | `tle_hash_safe = not _is_mthreads_runtime()`，MUSA 上禁用 TLE shared-memory hash/CAS 路径，改走已验证的 ESC | quick 下四个 dtype×索引宽度组合全部卡死，kill 前 stderr 为空；四个 Triton 原语探针单独全过，所以是**算法对原语的组合用法**踩到了同步阻塞，不是能力缺失 | 恒等变换（守卫为 False 时条件不变），CUDA `pytest -m spgemm_csr` 9 passed |
| `sparse_operations/spsv.py` | `effective_alg_num` | SELL 非转置 **ALG2 的实际 launch 降级到 ALG1**；descriptor/API 仍保留用户请求的 `alg_num=2` | `_spsv_sell_slice_kernel_alg2`（持久化 worker + ready flag 轮询）偶发失去前进性，`MUSA_ERROR_LAUNCH_TIMEOUT`，卡死位置固定在 n=64/slice=8 首次 solve | 恒等变换；CUDA 侧 `spsv_sell+spsv_csr+spsv_coo` 846 passed |
| `sparse_operations/spmv_csr.py` | `_spmv_csr_default_backend()` | 返回 `segbin`，"和 Ascend 一样从 CUDA 内核起步" | 没有 MUSA 专用内核，显式声明起点而不是靠 fallthrough | 恒等变换 |
| `sparse_operations/spmv_csr.py` | 后端名上报 | 环境报告里返回 `mthreads` | 结果文件要能自证跑在哪个后端 | 恒等变换 |

## 2. 按 dtype（不是按 backend）分支的一处

| 文件 | 锚点 | 改动 | 为什么 |
|---|---|---|---|
| `sparse_operations/_common.py` | `_gather_values()` | 复数重排改用 `view_as_real` 后 gather；SpGEMM / SpMM / SpMV / SpSM / SpSV 里所有 `data[order]` 路径都接入 | MUSA 的复数 **gather 方向**全挂（`a[order]` / `index_select` / `a[rows, cols]`），scatter 方向正常 |

**分支按 dtype 不按 backend** —— CUDA 上实数走原路径，复数走新路径且结果相同。这是有意的：
缺口是"复数索引内核"，不是"摩尔线程"。

## 3. 加速器抽象（影响所有后端，CUDA 上是恒等变换）

| 文件 | 锚点 | 改动 |
|---|---|---|
| `sparse_operations/_common.py` | `_resolve_accel()` / `_ACCEL` / `_ACCEL_DEVICE_TYPE` | torch 子模块与设备类型**成对**返回。只换模块不换设备类型会让 `_is_accel_tensor()` 拒绝掉每一个张量 |
| `sparse_operations/spmm_bell.py` | 输入检查 + timing/meta 路径 | `data.device.type != "cuda"` → `_is_accel_tensor()`；`torch.cuda.Event/synchronize` → `_ACCEL.*`（3 处） |
| `tests/pytest/accuracy_utils.py` | `accelerator_available()` / `accelerator_device()` / `golden_device()` | pytest 套件的守卫、设备字面量（148 处）、以及**参考值恒在 CPU** |
| `tests/pytest/test_spmm_bell_accuracy.py` | `pytestmark` | 移除 MUSA 整体 skip，改走 SciPy CPU 参考 |
| **`tests/benchmark_utils.py`（新增）** | `ACCEL` / `accelerator_device()` | 直接跑的 benchmark 脚本用的同一对名字；自己把 `src/` 加进 `sys.path`，因为脚本是先 import 它、后加 `src/` 的 |
| `tests/*.py` benchmark 共 20 个 | 全文件 | `torch.cuda.*` → `ACCEL.*`，`torch.device("cuda")` → `accelerator_device()` |

> `torch.cuda` 在摩尔线程上不是"选错了 GPU"，是"没有 GPU"：`torch.cuda.is_available()`
> 为 False，guard 直接跳过整个 benchmark、CSV 空着回来，而后端本身分发得好好的。

## 4. 基线与容差

| 文件 | 锚点 | 改动 | 为什么 |
|---|---|---|---|
| `sparse_operations/_common.py` | `_mthreads_vendor_sparse_library()` | 默认返回 `None`（此前是 `torch`） | 实测 MUSA 的 `torch.sparse` **能建 CSR/COO 张量但 sparse matmul 未注册**，四个 dtype 全挂。仍可用 `FLAGSPARSE_MTHREADS_VENDOR` 覆盖 |
| `tests/pytest/` 的 `spmv_coo` / `spmv_csc` | 容差 | fp32 与 complex64 放宽到 `1e-3`；fp64 / complex128 维持严格 | 累加顺序不同 |
| `capi/conf/operators.yaml` | `performance_baseline` | 加 `mthreads: musparse` | C API 侧基线已接上 |
| `capi/ctest/baseline/musa/baseline.cpp` | 整文件（420 行） | 前缀表 + `_template.inc` → **原生实现** | muSPARSE 4.3.5 的 SpMM/SpSV/SpSM/SpGEMM 是**一个入口加 stage 参数**，不是 cuSPARSE 的拆分函数，前缀表只能产出"看着能编、实则无效"的基线 |
| `capi/tools/matrix_sweep.cpp` | `Buf` / `upload()` / 计时循环 | `cudaMalloc`/`cudaMemcpy`/`cudaDeviceSynchronize` → `flagsparse::adaptor::*`，不再 include `<cuda_runtime.h>` | 同一个工具要在非 CUDA 后端上编译 |

## 5. 本轮回传清单（12 个文件，全部已合入）

```
flagsparse/tests/{test_gather,test_scatter,test_sddmm,test_spgemm,test_spmm,test_spmm_coo}.py
flagsparse/tests/benchmark_utils.py          ← 第二批补发（第一批漏了，6 个文件当时全 ImportError）
flagsparse/tests/test_spmm_csr.py            ← 回传的是未改版，本仓库按同样 pattern 补了
cwrapper/tools/matrix_sweep.cpp
cwrapper/conf/operators.yaml
cwrapper/ctest/baseline/CMakeLists.txt       ← 仅注释
cwrapper/baseline.cpp                        ← 第三批补发，装到 ctest/baseline/musa/
```

**本机没有 MUSA SDK，`baseline.cpp` 编不了。** 能静态核的都核了：11 个接口函数齐全、
9 个多参数签名与 `baseline.hpp` 逐个类型对上、用到的 28 个仓库符号（adaptor 4 +
`flagsparse.h` 14 + 接口结构体字段 10）全部存在、include 惯例与 `none/` 和 `_template.inc`
一致。**muSPARSE 那一侧用得对不对，只能在实机编译时才知道。**

## 6. 仍开着

- **Python 侧的 muSPARSE 基线未接** —— `FLAGSPARSE_MTHREADS_VENDOR=musparse` 只返回字符串，
  全仓搜不到 `import musparse`，基线列仍是 `N/A`。C API 侧已接上，两者是两回事。
- **`spmm_coo` 复数共轭用例稳定性** —— `conj-int64-complex128-8-4-16` 最近 3 次全过，
  但此前重复测试里出现过一次 `timeout -s KILL`，未宣告修复，`spmm_coo.py` 未改动。
- **`spsv_sell:404` 的 skip 理由含糊**（"MUSA SELL validation differs from CUDA"），没说差在哪。
