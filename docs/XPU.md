# 昆仑芯 XPU

Python 侧的昆仑芯后端：槽位已就位，实机（P800）脚本已回传并合入。本文写的是怎么跑、
以及哪些结论现在还不能下。改了哪些文件见 [`modified/XPU.md`](../modified/XPU.md)。
C API 那一层见
`capi/docs/XPU.md`（结论是 C API 目前在昆仑芯上跑不起来，可达的只有 Python 侧）；
各后端文档的对应关系见 [README.md](README.md)。

> **状态：有实机脚本，没有实机通过率。** 与 MACA（921 用例实机全绿）、MUSA（1564 用例
> 实机全绿）不同，昆仑芯这边回传的是**执行路径**（`benchmark/benchmark_xpu.py` 与
> `benchmark_xpu_variants.py`，后者的 `MEASURED` 表注明每条都在 P800 上单独启动过），
> **不是一轮成建制的精度/性能结果**。本文第 1、2 节仍来自代码与注册表；下面凡是"期望"
> 的地方，都要在真机上确认后再当结论用。

---

## 1. 探测：插件必须真的装了，`torch.xpu` 存在不算数

后端名是 **`xpu`**，注册在 `_common.py` 的 `_BACKEND_SPECS`：

| 字段 | 值 |
|---|---|
| 厂商 | Kunlunxin XPU |
| torch 命名空间 | `torch.xpu` |
| 插件模块 | `torch_xmlir`、`torch_xpu`（任一可 import 即可） |
| 设备名 token | `kunlun`、`xpu` |

**探测要求插件模块可 import，光有 `torch.xpu` 命名空间不认。** 这不是多余的谨慎：
上游 PyTorch 给 Intel GPU 也装了 `torch.xpu` 命名空间，只看命名空间会把一台 Intel 机器
当成昆仑芯 —— 要么在 Intel 硅片上跑却报告昆仑芯，要么在有命名空间没设备的机器上于
第一次分配时报一个既不指明厂商也不指明原因的错。逻辑在 `_vendor_plugin_present()`。

`FLAGSPARSE_BACKEND=xpu` 的显式指定**优先于探测**，这是让没法探测的机器也能进到这个
槽位的办法。

### 自检

```bash
export PYTHONPATH=$PWD/src
export FLAGSPARSE_BACKEND=xpu

python3 - <<'PY'
import importlib, torch
for m in ("torch_xmlir", "torch_xpu"):
    try:
        importlib.import_module(m); print(f"{m}: ok")
    except Exception as exc:
        print(f"{m}: 不可用 - {exc}")
print("torch.xpu 命名空间:", hasattr(torch, "xpu"))
if hasattr(torch, "xpu"):
    print("torch.xpu.is_available():", torch.xpu.is_available())
PY

python3 - <<'PY'
import flagsparse.sparse_operations._common as C
print("backend          :", C._backend_name())        # 期望 xpu
print("is_xpu_runtime   :", C._is_xpu_runtime())      # 期望 True
print("accel device type:", C._accel_device_type())   # 期望 xpu
print("fallback reason  :", C._accel_fallback_reason())
PY
```

`accel device type` 若是 `cuda` 而不是 `xpu`，说明插件没找到、整轮会跑在 CUDA 语义下 ——
和 MUSA 那边同一个坑，见 `MUSA.md` 第 0 节。

---

## 2. 内核：没有 XPU 专用实现，走共享那一份

`src/flagsparse/sparse_operations/` 下的算子文件里**没有任何 `_is_xpu_runtime()` 分支**
（可以自己 grep 确认）。`backends/xpu/` 目录已建好但只有 `__init__.py`，算子代码留空，
因此解析到共享实现。

这是有意的：`_dispatch.py` 的覆盖层默认为空，**只有真正分叉的文件才会出现在里面**。
真在昆仑芯上要改，优先级从低到高是——加 profile 表项（launch 参数、`num_warps` 之类）
→ 同文件内加内核变体 → fallback 分发表 → 最后才是往 `backends/xpu/` 放整份文件。

所以昆仑芯上能不能跑，取决于 **FlagTree 的 xpu target 能不能把这些 Triton 内核编出来**，
而不是取决于本仓库有没有 XPU 代码。

---

## 3. 跑测试

### 按后端 suite 跑

`tests/backends/xpu/suite.json` 已就位，`tools/run_backend_tests.py` 会据此设好
`FLAGSPARSE_BACKEND`：

```bash
python tools/run_backend_tests.py --backend xpu --phase accuracy --mode quick
```

精度用例与其他后端**共用** `tests/pytest`，不另起一套 —— 六份同样的 oracle 各自漂移
是这个仓库明确要避免的。

### 性能阶段：五个算子走自己的脚本，其余走能力探测

```python
PROBE_ONLY_BACKENDS: tuple[str, ...] = ("gcu", "mlu")     # xpu 已经不在里面
XPU_BASELINE_OPS = ("gather", "scatter", "spmv_csr", "spmm_csr", "sddmm_csr")
```

这五个走 `benchmark/benchmark_xpu.py`，其余算子仍走**能力探测**，按算子报
`PASS` / `REJECTED` / `TRITON_COMPILE` / `ERROR` / `NO_ADAPTER`。在一个内核可能根本
lower 不出来的平台上，这才是有意义的测量 —— 替代方案是一个空的性能阶段，而空阶段
读起来和通过一模一样。

> **`benchmark_xpu.py` 给的是时延和"能否执行"，不是加速比。** 它为五个算子各建了一个
> PyTorch-XPU 参考闭包，但**一次都没调用**，`pytorch_ms` / `speedup` / `max_abs_err`
> 三列恒为空，`status` 只有 PASS（没抛异常）或 ERROR（带 reason）。脚本 docstring 写明了
> 这是有意的。要真做对比，把那几个 `baseline()` 调起来即可，结构都在。

探测脚本和 `benchmark_xpu.py` **无论内核跑通、被拒还是编译失败都 exit 0**，结论在 CSV 行
里。runner 因此用 `_capability_probe_status()` 折叠行状态（全同取之、混合取 `MIXED`、
无行取 `NO_TESTS`）而不是看退出码 —— 看退出码会把每一行都报成 pass。

```bash
export PYTHONPATH=$PWD/src FLAGSPARSE_BACKEND=xpu
python run_flagsparse_pytest.py --phase both --mode quick --gpus 0 \
  --results-dir pytest_results_xpu
```

> 探测脚本叫 `benchmark/benchmark_ascend_probe.py`，**名字有历史包袱，实现是后端中立的**
> —— 它通过 `_resolve_accel()` 跟随当前后端，不绑昇腾。每个用例都清空 Triton 缓存后再跑，
> 因为上一个用例留下的缓存产物会让编译失败看起来像通过。

---

## 4. 没有厂商基线，而且不是"等一个库名"

昆仑芯的数学库是 **XDNN**，提供的是**固定的稀疏算子**，不是描述符式的 generic API。
所以 speedup 列会是空的，`reason` 字段里写明原因。

这和 MACA / MUSA 的情况**不是一回事**，不要混：

| | 昆仑芯 XPU | MACA / MUSA |
|---|---|---|
| 厂商有稀疏库吗 | 有 XDNN，但不是 generic API | 有，且是 cuSPARSE 克隆 |
| 缺什么 | 缺的是 **API 形状**，补不了 | 只缺头路径和库名 |
| 怎么解 | 需要厂商出 generic API，或本仓库为 XDNN 写逐算子适配 | 填两个 CMake 变量 |

这是**诚实的空**，不是没做完。详见 `capi/docs/XPU.md`。

---

## 5. C API 在昆仑芯上跑不起来

两条独立阻塞，补上其中一条不解锁另一条：

1. **`deps/libtriton_jit` 的后端清单里没有 XPU**（有 CUDA / GCU / HCU / MACA / MLU /
   MUSA / NPU）。这个桥负责编译和启动每一个内核，所以写一个 adaptor 不会有帮助 ——
   adaptor 解决的是 runtime 调用，不是内核从哪来；
2. **没有 cuSPARSE 形状的稀疏库可做基线**（见第 4 节）。

`tests/ci/test_backend_test_profiles.py` 里 `xpu` 的 `capi_buildable` 因此是 `False`，
这是被断言住的状态而不是待办。要在昆仑芯上测算子，今天的入口就是本文第 3 节的 Python 侧。

---

## 6. 首次上机时按这个顺序走

1. 第 1 节的自检 —— `backend=xpu` 且 `accel device type=xpu` 再往下；
2. 确认 FlagTree 的 xpu target 能编译执行一个**最小 Triton kernel**（写成真实 .py 文件，
   不要用 stdin，Triton 要读源码；参考 `MACA.md` 2.1 那段）；
3. 单算子精度：先 `gather` / `scatter`（最简单、无需矩阵文件），再 `spmv_csr`；
4. 全量精度 `--mode normal`（`quick` 会砍掉约四成覆盖，且恰好跳过历史上出过问题的两个
   用例，见 `MACA.md` 7.1）；
5. 性能阶段的能力探测，拿到逐算子的 PASS / TRITON_COMPILE 分布；
6. 把第 1 节自检的实际输出、以及探测结果回填到本文 —— 那时才能把顶部的"未验证"去掉。

排查时通用的几条（跨后端适用）：**一律 `timeout -s KILL`**、**一个配置一个进程**、
**先看 stderr 的厂商诊断**、**"没输出"不等于"通过"**，展开见 `MUSA.md` 第 10 节。
