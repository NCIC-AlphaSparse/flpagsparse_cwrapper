# 昇腾 Ascend —— 改动台账

**实机环境**：Ascend 910B / CANN / torch_npu（性能测试只用 NPU 6、7 号卡）。
**基于版本**：上游 FlagSparse。
**最近回传**：2026-09-15，runner 的精度分支；`benchmark_ascend_probe.py` 未进上游。

跑法见 [`docs/ASCEND.md`](../docs/ASCEND.md)；C API 那层见
[`capi/docs/ASCEND.md`](../capi/docs/ASCEND.md)。

---

## 1. 库内的后端分支（6 个文件，共 17 处）

| 文件 | 处数 | 锚点 | 改动 | 为什么 |
|---|---:|---|---|---|
| `sparse_operations/gather_scatter.py` | 7 | `_is_ascend_runtime()` 守卫的 fallback | gather/scatter 走 torch_npu 路径 | CANN 侧 lower 不出对应内核形状 |
| `sparse_operations/spmv_csr.py` | 3 | `_spmv_csr_default_backend()` 等 | 起点 `segbin`；**`tl.associative_scan` 在 Ascend 上 lower 不了**，相关路径退回 torch_npu | 与摩尔线程明确不同：MUSA 的 associative_scan 是通的，不需要这个回退 |
| `sparse_operations/spmm_csr.py` | 3 | 厂商基线选择 + dispatch | `vendor in ("ops_sparse","torch") and _is_ascend_runtime()` | |
| `sparse_operations/sddmm_csr.py` | 2 | 同上 | | |
| `sparse_operations/spsm.py` | 1 | **`SPSM_ASCEND_DISPATCH`** | `spsm_csr` / `spsm_coo` 的 Ascend NPU fallback 分发表 | CANN lower 不出那两种内核形状 |
| `sparse_operations/spmm_coo.py` | 1 | **`SPMM_COO_ASCEND_DISPATCH`** | `spmm_coo` 走 torch_npu，绕开 shmem 编译问题 | |

两张分发表都从 `sparse_operations/__init__.py` 和 `flagsparse/__init__.py` 导出
（`__all__` 里各两条）。

> **导出这件事要确认，不要假设。** 这一轮的 drop 曾用了未导出的 `_IS_ASCEND_RUNTIME`，
> 留下一个只在特定分支才触发的 NameError。`from ._common import *` 只解析 `__all__` 里的名字。

## 2. runner 与 benchmark

| 文件 | 锚点 | 改动 | 为什么 |
|---|---|---|---|
| `run_flagsparse_pytest.py` | `ASCEND_BASELINE_OPS` | 五个算子（gather / scatter / spmv_csr / spmm_csr / sddmm_csr）走 `benchmark_ascend.py`，有厂商基线 | 只有这五个在 CANN 上有可用参照 |
| `run_flagsparse_pytest.py` | `ASCEND_ACCURACY_OPS` | **由 `ASCEND_BASELINE_OPS` 派生**，不重复写第二份名单 | 回传的版本里带了第二份硬编码副本，两份一旦改动就会漂移 |
| `run_flagsparse_pytest.py` | `ASCEND_PROBE_OPS` | 其余 17 个算子走能力探测 | 在可能 lower 不出内核的平台上，PASS/REJECTED/TRITON_COMPILE 才是有意义的测量 |
| `run_flagsparse_pytest.py` | `run_accuracy()` 的 Ascend 分支 | 这五个算子的精度改由 `benchmark/benchmark_ascend_accuracy.py` 跑，结果按原 `accuracy_result.json` 格式解析汇总 | |
| `benchmark/benchmark_ascend_probe.py` | 整文件 | 20 个矩阵、缓存清理、BSR/SELL/SpSV/SpSM 输入适配，区分 PASS / TRITON_COMPILE / REJECTED / ERROR / NO_ADAPTER | **每个用例前清空 Triton 缓存**，否则上一个用例的产物会让编译失败看起来像通过 |

> 这个探测脚本**名字带 ascend，实现是后端中立的**（通过 `_resolve_accel()` 跟随当前后端），
> 昆仑芯和后续 probe-only 后端复用同一份。昆仑芯那边提出把它改名
> `backend_capability_probe.py`，但文件没发过来，见 [`XPU.md`](XPU.md) §3。

## 3. 仍开着

- `benchmark_ascend_probe.py` **未进上游**，目前只在本仓库。
- 探测脚本只探能力不影响算子执行，所以它的改动不必逐次回传（已与维护者确认）。
