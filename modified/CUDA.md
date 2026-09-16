# NVIDIA CUDA —— 改动台账

**这个文件的内容注定是空的，那正是它要表达的意思。**

CUDA 是参照路径：共享的那份 Triton 实现就是在 NVIDIA 上验证过的代码，
`src/flagsparse/sparse_operations/` 下**没有任何 `_is_cuda_runtime()` 守卫** ——
不需要，因为所有后端分支都写成"别的后端做什么"，CUDA 是 fallthrough。

所以本文的作用是给其他后端立一条验收线：

## 每一条后端改动都要证明它在 CUDA 上是恒等变换

| 形式 | 怎么证 |
|---|---|
| `if _is_<backend>_runtime():` 新增分支 | 守卫为 False 时控制流逐字不变，跑一遍该算子的 marker |
| 改了某个默认值/profile | CUDA 走的是另一个 key，贴出两个 key 的取值 |
| 按 **dtype** 而不是 backend 分支（如 `_gather_values()`） | CUDA 上也走新路径，必须给出**结果相同**的证据，不能只说"没报错" |
| 测试文件整体换设备抽象 | `_ACCEL is torch.cuda`、`_ACCEL_DEVICE_TYPE == "cuda"`，CPU 上的 fp64 oracle 只会比设备端更准 |

本仓库已有的回归基线（供对比，跑之前先确认自己的基线是什么）：

```
tests/ci        80 passed / 3 skipped
tests/pytest    1613 passed / 3 failed   （3 个是既有失败，与后端改动无关）
```

三个既有 flaky（CUDA 上同样复现，随机输入未固定种子）：
`test_spmv_csc_matches_dense_reference[float32-160-1024]`、
`test_spsv_csr_upper_optimized_route_analysis_workspace_matches_direct[csr_cw_levelschd]`、
`test_spsv_sell_non_unit_rejects_malformed_structure[duplicate_diagonal]`。

## 唯一一类"CUDA 侧改动"

不是后端适配，而是**所有后端共用的东西在 CUDA 上改**：加速器抽象（`_ACCEL`）、
交付变体投影、结果格式。这类改动要反过来证明它在**其他后端**上也成立 —— 举证责任在
改动者，不在各后端。它们不属于任何一个后端的台账，写在被改模块自己的注释和
`docs/` 里。
