# 昆仑芯 XPU —— 改动台账

**实机环境**：P800 / XPU runtime（`benchmark_xpu_variants.py` 的 `MEASURED` 表注明逐条实跑过）。
**基于版本**：一份**早于 2026-09-16 交付变体改造**的 `run_flagsparse_pytest.py` 副本。
**最近回传**：2026-09-16 17:22，3 个文件；2 个脚本逐字节合入，runner **逐 hunk 移植**（原因见下）。

跑法见 [`docs/XPU.md`](../docs/XPU.md)；C API 那层（目前跑不起来）见
[`capi/docs/XPU.md`](../capi/docs/XPU.md)。

---

## 1. ⚠️ runner 不能整份覆盖

回传的 `run_flagsparse_pytest.py` 与仓库版差 **+148 / −230 行**。多出来的是 XPU 路由；
**少掉的 230 行里有 137 行是当天刚做完的交付变体投影**：

```
_DELIVERY_DTYPE_TOKENS / _DELIVERY_PERF_DTYPES
_delivery_not_configured_phase()  _delivery_accuracy_phase()
_delivery_performance_phase()     _delivery_results()
summarize_accuracy_cases()
from tools.delivery_variants import load_delivery_variants
write_summary() 里 ordered = _delivery_results(results)
```

整份覆盖的结果是 `summary.json` 从 **40 个交付变体**退回按算子名出结果，C API 与 Python
两侧的清单统一当场失效。所以只摘了下面第 2 节那几项移植，其余保留仓库版。

**下次回传 runner 请只写改了哪几个函数**，或者先 pull 一次再改。

## 2. 已移植的改动（都在 `run_flagsparse_pytest.py`）

| 锚点 | 改动 | 为什么 | 其他后端 |
|---|---|---|---|
| 模块顶部 | `csv.field_size_limit(sys.maxsize)`，`OverflowError` 时退到 `2**31-1` | XPU 编译诊断里可能带 MLIR reproducer，超过 csv 模块 128 KiB 默认上限。**它是执行结果的一部分，留不住就会把 TRITON_COMPILE 行变成 PASS** | 无影响 |
| `STATUS_TO_FLAGGEMS` | 新增 `REJECTED→Skipped`、`TRITON_COMPILE→Failed`、`MISMATCH→Failed`、`ERROR→Error`、`MIXED→Error` | 探测结论要能进 FlagGems 格式。后端拒绝的内核是 skip，编译不出来的是 fail，**两者都不能读成 pass** | 纯新增键 |
| `PROBE_ONLY_BACKENDS` | `("xpu","gcu","mlu")` → `("gcu","mlu")` | XPU 有了自己的脚本 | gcu/mlu 不变 |
| `XPU_BASELINE_OPS`（新增） | `gather`/`scatter`/`spmv_csr`/`spmm_csr`/`sddmm_csr` 五个走 `benchmark_xpu.py`，其余留在探测路径 | XPU SDK 没有 cuSPARSE 形状的 generic 稀疏 API | 仅 `backend == "xpu"` 分支 |
| `_xpu_baseline_command()`（新增） | 渲染 `benchmark/benchmark_xpu.py --op/--device/--csv-summary/--warmup/--iters` | | |
| `run_performance()` | 加 `elif backend == "xpu":` 路由 | | 其他分支不变 |
| `_capability_probe_status()`（新增） | 把 CSV 行里的状态折叠成一个：全同取之，混合取 `MIXED`，无行取 `NO_TESTS` | **探测脚本无论内核跑通、被拒还是编译失败都 exit 0**，verdict 在行里。拿退出码当结论会把每一行都报成 pass | 只在探测/XPU 脚本上启用 |
| `write_benchmark_json_from_csv()` | 加 `status=` / `test_case=` 透传（`benchmark_json_from_csv` 早就有这两个参数） | 上面那条要用 | 默认值不变 |
| `load_operator_catalog()` | 除 `ops:` 外支持 `operators:` 列表，提取 `id`/`labels`/`reporting`/`delivery_dtypes`/`dtypes` | 能直接读 C API 的交付清单 `capi/conf/operators.yaml` | `conf/operators.yaml` 走原分支 |
| `_parse_operator_catalog_fallback()` | 认 `reporting:` 字段 | 没装 yaml 时的退化解析要一致 | |
| `read_ops(delivery_only=...)` + `--delivery-only` | 只选 `reporting: delivery` 的条目 | | 默认 False |

实测（本机，无 XPU 硬件，只验路由与解析）：`--delivery-only` 从 `capi/conf/operators.yaml`
的 22 条里选出 **11** 条 delivery，不带该参数是 15 条；`conf/operators.yaml` 仍解析出 25 条；
五个 baseline 算子路由到 `benchmark_xpu.py`，其余到探测脚本；状态折叠四种情况全部符合预期。

## 3. 未移植 / 缺口

| 项 | 情况 |
|---|---|
| **`benchmark/backend_capability_probe.py`** | 回传的 runner 把探测脚本指向这个**后端中立的新名字**，并改用 `--mode quick` 取代 `--warmup/--iters`。**这个文件不在回传里**，仓库中也没有。因此 `_probe_command()` 仍指向现有的 `benchmark/benchmark_ascend_probe.py`（它没有 `--mode` 参数）。**要么把新脚本发过来，要么确认沿用旧的** |
| 大量括号/换行重排 | 回传版本未经 `ruff format`，diff 里相当一部分是这类噪声，未采纳 |

## 4. 新增脚本（逐字节合入）

| 文件 | 内容 |
|---|---|
| `benchmark/benchmark_xpu.py` | 在 XPU 上构造并执行 FlagSparse 内核并计时。CSV 列：`dtype,shape,triton_ms,pytorch_ms,speedup,max_abs_err,status,baseline,reason` |
| `benchmark/benchmark_xpu_variants.py` | 按清单跑变体，带 `MEASURED` 表（注明每条都在 P800 上单独启动过），支持 `--manifest` / `--matrix-dir` |

> **`benchmark_xpu.py` 名字叫 baseline，实际不做对比。** 五个算子各自建了一个
> PyTorch-XPU 参考闭包 `baseline()`，但**一次都没有被调用**；`candidate_result = candidate()`
> 执行一次后丢弃，`pytorch_ms` / `speedup` / `max_abs_err` / `baseline` 四列恒为空字符串，
> `status` 是 PASS（没抛异常）或 ERROR（带 reason）。脚本自己的 docstring 写明了
> "intentionally has no baseline"，但回传 runner 里的注释说这五个算子"get a measured
> baseline" —— 两者矛盾。仓库这边的注释已按**实际行为**改写。
>
> 也就是说这条路径给的是**时延 + 能否执行**，不是加速比，也不是精度比对。要真做对比，
> 把那五个 `baseline()` 调起来即可，结构都在。

## 5. 库内改动：零

`src/flagsparse/` 下**没有任何 `_is_xpu_runtime()` 分支**，`backends/xpu/` 只有
`__init__.py`，算子走共享实现。能不能跑取决于 FlagTree 的 xpu target 能否编出这些 Triton
内核，不取决于本仓库有没有 XPU 代码。
