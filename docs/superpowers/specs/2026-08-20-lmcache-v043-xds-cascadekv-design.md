# LMCache v0.4.3 xDS 与 CascadeKV 后端设计

**状态：** 已于 2026-08-20 在对话中确认

## 目标

将本地 xDS 修改向前移植到上游 LMCache v0.4.3，同时保持仓库名、Python
发行包名和导入包名均为 `lmcache`。最终提供两个互斥的存储后端，二者都连接
到当前 worker/rank 对应的 xDS 服务客户端：

- `XDSBackend`：通过服务注册的 GPU buffer，实现 SSD 到 GPU 的直接传输。
- `CascadeKVBackend`：先从 SSD 读入由服务管理的 CPU staging 内存，再通过
  LMCache 的推理引擎 GPU connector 从 CPU 传输到 GPU。

两个路径都必须兼容源码安装的 vLLM 0.18.0。SGLang 同样从源码安装，并在
LMCache 后端迁移完成后进行集成验证。三者安装在同一个 Conda 环境中，统一
以 PyTorch 2.10.0 为优先约束。

## 版本与兼容性约束

- LMCache 基线：上游标签 `v0.4.3`。
- LMCache 工作分支：`v0.4.3-xds`。
- vLLM 目标版本：上游标签 `v0.18.0`。
- SGLang 基线：上游标签 `v0.5.9`；工作分支为
  `v0.5.9-torch210-xds`。
- PyTorch 目标版本固定为 `2.10.0`，优先级高于 SGLang v0.5.9 原始依赖
  元数据中的 `torch==2.9.1`。
- SGLang 源码分支将 PyTorch 家族依赖调整到 2.10 对应版本，并保留
  `transformers==4.57.1`，满足 vLLM 0.18.0 的 `transformers>=4.56,<5`
  约束。若预编译 `sgl-kernel` 出现 ABI 不兼容，则针对 PyTorch 2.10 重新
  构建或替换该 kernel，不允许因此降级 PyTorch。
- 唯一的新 Conda 环境名为 `xds043`。LMCache、xDS SDK、vLLM 和 SGLang 都
  在该环境中从各自源码进行 editable 安装。
- 现有 `/home/yangxc/workspace/LMCache` 工作区保持不变，只作为本地实现和
  测试的迁移来源。
- xDS Python SDK 从 `/home/yangxc/workspace/xDS/sdk` 源码安装，并提供
  `XDSClient`。
- 后端选择和资源所有权均以 LMCache worker/rank 为单位。

## 选定架构

### 共享服务客户端

当任一后端开启时，LMCache 为每个 worker rank 创建一个已连接的
`XDSClient`。连接参数包括 `xds_service_name`、LMCache worker rank、world
size，以及由后端决定的 `require_direct`。客户端拥有服务连接状态；被选中的
后端负责管理通过该客户端申请的 blob handle、buffer 注册和 buffer 分配。

共享连接辅助逻辑放在新的独立文件 `lmcache/v1/xds_utils.py` 中。通用 xDS
blob 元数据也放在该文件中，避免将 xDS 专用类型加入 LMCache 的通用
`lmcache/utils.py`。

### 直接 xDS 后端

`XDSBackend` 要求目标设备为 CUDA，并要求 `xds_require_direct: true`。它使用
`XDSClient.register_gpu_buffer()` 将 LMCache 的 GPU 内存注册一次。每条 I/O
描述符通过 `buffer_ref`、buffer offset、blob offset 和 size 指定传输范围。
读取路径为：

```text
SSD -> xDS 服务 -> 已注册的 LMCache GPU buffer
```

写入路径与之相反。从本地实现向前移植窗口化 I/O、队列深度限制、分阶段写入
lease、完成结果校验、超时处理，以及确定性的 GPU buffer 注销逻辑。失败或
短 I/O 不得将相应 blob 元数据发布为已持久化状态。

### CascadeKV 后端

`CascadeKVBackend` 使用相同的 xDS 服务协议，但不要求 GPU direct 能力。它
通过 `XDSClient.alloc_buffer()` 申请由 xDS 服务管理的 CPU DMA buffer。返回
的 `memoryview` 和 `buffer_ref` 在整个生命周期中必须保持一一对应。读取路径
为：

```text
SSD -> xDS 服务 -> 服务管理的 CPU staging buffer -> GPU
```

写入路径为：

```text
GPU -> 服务管理的 CPU staging buffer -> xDS 服务 -> SSD
```

CPU memoryview 直接适配成 LMCache memory object，不引入额外的 CPU 到 CPU
拷贝。CPU/GPU 传输使用 LMCache 现有的 GPU connector。后端维护一个有界的
staging buffer 池，容量按完整且对齐的 blob slot 计算。在 xDS I/O 和对应的
GPU copy 都完成前，slot 不得被复用。

关闭后端时，必须先完成或取消所有 I/O，释放未提交的 blob handle，通过
`XDSClient.free_buffer()` 释放全部 CPU buffer，最后才允许关闭客户端。

### CascadeKV 流水线性能恢复

移植后的第一版只保留了功能语义：每个 key 依次执行一次 xDS I/O，并用同一组
CPU staging slots 承担读写。这与移植前实现不同，会把 SSD I/O、CPU/GPU copy
和 Python 调度串行化；长 prompt 还会因为 memory object 生命周期过长而要求
staging slots 接近 prompt 的 chunk 数量。本轮恢复移植前的有界流水线语义，
但仍使用外部 xDS 服务 API，不恢复旧的进程内 xDS SDK。

服务分配的 hugepage buffer 必须在生成 `torch.frombuffer()` 视图后通过
`cudaHostRegister(..., cudaHostRegisterDefault)` 注册为 CUDA pinned host
memory。移植前的 `PinMemoryAllocator` 和 `MixedMemoryAllocator` 已执行同等注册；
当前 service staging 绕过了这两个 allocator，因此必须在 CascadeKV buffer
生命周期内显式补齐。注册失败时后端构造失败并释放 service buffer；关闭时先
排空 xDS I/O 和 CUDA event，再 `cudaHostUnregister()`，最后调用
`XDSClient.free_buffer()`。

读写使用互不占用的 slot bank：

```text
写：GPU -> write slots -> 后台 xDS 写队列 -> SSD
读：SSD -> read slots -> GPU connector -> GPU
```

写路径恢复旧版的聚合与异步提交行为。前台 GPU 到 CPU copy 完成后，把包含
memory object、blob handle 和 CUDA event 的 lease 放入有界后台队列；后台按
write window 聚合，最多维持配置的 queue depth 个未完成 xDS write。只有完整
完成的 write 才发布持久化 metadata。尚未完成的 key 保存在 pending map 中，
使同进程的立即读取可以命中写入中的 staging 数据。队列和 write slots 用 credit
施加背压，禁止覆盖尚未完成的 lease。

读路径使用有界 completion-driven ring。调度线程持续填满 read queue depth，
并按任意完成顺序返回已读 slot；主线程一收到完成项就立即通过 GPU connector
发起 CPU 到 GPU copy，同时调度线程继续进行后续 SSD 到 CPU 读取。每个 read
slot 直到对应 CUDA event 完成后才能回收，因此多个 chunk 形成
`SSD -> CPU` 与 `CPU -> GPU` 的双级重叠流水线。短读、超时或 connector 失败
只影响对应 chunk，并按 LMCache 的连续 prefix 规则截断命中结果。

第一版以完整 LMCache chunk 作为一个 slot，不恢复旧实现中“单个 blob 再切片”
的实验路径。当前 256-token chunk 对应的 blob 已足够小，跨 chunk ring 能在更少
状态和复制次数下提供所需 overlap。

为了让 vLLM 调用 completion-driven 的整批路径，`StorageManager` 的 direct
transfer hook 从只允许 `XDSBackend` 扩展为同时允许 `CascadeKVBackend`。
CascadeKV 性能配置使用 `use_layerwise: false`；layerwise 路径保留兼容，但不作为
本轮性能基准，因为其按 layer 分配 memory objects 的生命周期会削弱有界 slot
流水线。

### 后端工厂与 StorageManager

`CreateStorageBackends` 只构造 `XDSBackend` 和 `CascadeKVBackend` 中被选中的
一个。直接 xDS 后端作为 allocator backend，因为其对象位于已注册的 GPU
内存中。CascadeKV 使用 LMCache 的常规本地 CPU 分配与存储流程，并作为持久化
存储后端。其他 LMCache 通用后端保持 v0.4.3 原有行为。

## 配置接口

公开 YAML 配置键如下：

```yaml
xds_service_name: xds0

# SSD -> GPU 直接路径
xds_backend: false
xds_require_direct: false
xds_windowed_io: true
xds_read_window_size: 64
xds_write_window_size: 64
xds_io_queue_depth: 1
xds_read_slot_count: 2
xds_write_slot_count: 8
xds_async_write: false

# SSD -> CPU -> GPU 分级路径
cascadekv_backend: false
cascadekv_overlap: true
cascadekv_async_write: true
cascadekv_read_slot_count: 8
cascadekv_write_slot_count: 64
cascadekv_io_queue_depth: 4
cascadekv_write_window_size: 64
```

配置校验规则如下：

1. `xds_backend` 和 `cascadekv_backend` 不能同时为 true。
2. 开启任一后端时，必须提供非空的 `xds_service_name`。
3. `xds_backend` 要求 `xds_require_direct: true`。
4. `cascadekv_backend` 使用 `require_direct: false` 连接；CascadeKV 配置中设置
   `xds_require_direct: true` 将直接报错，避免数据路径含义不明确。
5. window size、queue depth 和 slot count 必须为正数。
6. 直接后端的 queue depth 不能超过 read slot count 或 write slot count。
7. CascadeKV read/write slot count、queue depth 和 write window size 必须为正数；
   queue depth 不能超过相应 slot count。
8. 旧配置 `cascadekv_slot_count` 仅作为 read/write slot count 均未显式配置时的
   兼容回退值，新配置优先。

旧进程内实现使用的 `bdev_name`、`rpc_addr` 和 `xds_max_size` 等配置不再由
两个新后端使用。存储设备和容量由外部 xDS 服务负责管理。

## 代码边界

实现范围限定为以下职责：

- `lmcache/v1/storage_backend/xds_backend.py`：GPU direct 后端及其直接 I/O
  调度。
- `lmcache/v1/storage_backend/cascadekv_backend.py`：CPU staging 后端、staging
  slot 池和 CPU memory object 适配。
- `lmcache/v1/xds_utils.py`：创建客户端、对齐辅助函数和通用 xDS blob 元数据。
- `lmcache/v1/storage_backend/xds_read_ring.py`：仅供 `XDSBackend` 使用的有界
  direct-read slot 调度。
- `lmcache/v1/storage_backend/xds_write_scheduler.py`：仅供 `XDSBackend` 使用
  的 direct async-write lease。
- `lmcache/v1/storage_backend/transfer_result.py`：当 v0.4.3 通用接口无法表示
  部分成功时，提供明确的逐 chunk 传输结果。
- `lmcache/v1/config.py`：实现上述配置项与校验规则。
- `lmcache/v1/storage_backend/__init__.py`：惰性导入并互斥构造两个后端。
- `lmcache/v1/storage_backend/storage_manager.py`：最小化修改 allocator backend
  选择与传输结果消费逻辑。
- `lmcache/v1/cache_engine.py`：仅加入直接后端必需的 xDS client 和 allocator
  生命周期钩子。

默认保留上游 v0.4.3 的 vLLM connector。只有当失败的集成测试证明 xDS buffer
生命周期或关闭逻辑无法通过上游 connector 表达时，才允许修改 connector。
本次工作不进行全项目重构，也不修改包名。

## 错误处理与资源所有权

- 服务连接失败时中止后端构造，错误信息包含 service name 和 worker rank。
- 在提交 I/O 前完成对齐、offset 和注册范围检查。
- 超时和短完成均视为失败；失败 chunk 保持 cache miss，新申请的 blob handle
  必须被释放。
- CascadeKV staging slot 使用明确的状态转换：
  `free -> xds_io -> gpu_copy -> free`。发生异常时，必须先经过清理流程再归还
  slot。
- 直接写入 lease 使用 `reserved -> submitted -> committed`，失败路径为
  `reserved/submitted -> rolled_back`。
- 两个后端的 `close()` 都必须幂等。开始关闭后拒绝新的操作。
- 后端必须先关闭自己的 buffer 和 worker，再关闭 rank-local client。
  `XDSClient.close()` 执行时不得存在未释放的 CPU buffer ref。

## 测试策略

开发采用测试先行方式，单元测试不依赖真实 SSD 硬件。

1. 配置测试覆盖后端互斥、service name 必填、direct capability 规则、正数范围、
   YAML 解析和环境变量覆盖。
2. 后端工厂测试证明只惰性构造被选中的后端，并且不会导入未使用后端的可选
   依赖。
3. `XDSBackend` mock-client 单元测试覆盖 GPU 注册、描述符边界、成功读写、短
   完成回滚、超时清理、异步写入排空和幂等关闭。
4. `CascadeKVBackend` mock-client 单元测试覆盖 `alloc_buffer`、无中间 CPU
   拷贝、pinned 注册和注销、SSD 到 CPU 到 GPU 的调用顺序、GPU 到 CPU 到 SSD
   的调用顺序、读写 overlap、聚合写、pending read、slot 背压、超时清理、
   `free_buffer` 和关闭顺序。测试必须证明首个 I/O 完成前已经提交多个请求，并
   证明 CUDA event 完成前 slot 不会被复用。
5. 现有 LMCache v0.4.3 存储与配置测试保持通过。
6. 源码安装 smoke test 在同一个 `xds043` 环境中从预期源码路径导入
   `lmcache`、`xds`、vLLM 和 SGLang，确认 `torch==2.10.0`、
   `transformers==4.57.1`，并输出各源码版本和加载路径。
7. 硬件验证启动 xDS 服务，分别使用两个后端通过 vLLM 0.18.0 连续发送两次
   相同的长 prompt。第二次请求必须报告外部缓存命中，日志必须明确标识所选
   数据路径。
8. 适配 SGLang 源码集成后重复一次 prompt 对，不修改 LMCache 后端接口。

## 验收标准

- 仓库名和导入包名保持 `lmcache`。
- `xds_backend: true` 选择并实际执行 SSD 到 GPU 的直接 I/O。
- `cascadekv_backend: true` 选择并实际执行通过外部 xDS 服务的 SSD 到 CPU 到
  GPU 路径。
- CascadeKV 的 service staging 是 CUDA pinned host memory；测试和运行指标证明
  read ring 与 async write queue 的最大并发度大于 1。
- 40K prompt 不要求 slot 数量覆盖全部 chunk；默认 8 read + 64 write slots 时
  能依靠背压完成存取，不出现 staging allocation failure。
- 同时开启两个后端的无效配置在资源分配前失败。
- 单元测试证明成功、超时、短 I/O 和关闭场景都能确定性清理资源。
- LMCache、vLLM 和 SGLang 在同一个新的 `xds043` Conda 环境中均为 editable
  源码安装；环境中的 PyTorch 固定为 2.10.0，现有已验证的 `xds` 环境保持
  可恢复。
- vLLM 和 SGLang 各完成一次重复 prompt 验证，确认命中 xDS 后端，而不是仅
  命中本地 GPU prefix cache。

## 未采用的方案

- 将 v0.4.3 直接合并进旧的脏分支：这会把 836 个上游提交与旧接口混合，难以
  判断问题来自接口迁移还是冲突处理。
- 保留进程内 `xds.init_blob_pool()` 的 CascadeKV 路径：这不符合两个后端都
  使用外部 xDS 服务的要求。
- 将服务 CPU buffer 再复制到第二个 LMCache CPU buffer：这会产生不必要的
  CPU 到 CPU 拷贝，并使 staging 内存压力翻倍。
- 将 Python 发行包或 import 改名为 `cascadekv`：这会破坏推理引擎集成，并且
  对后端选择没有帮助。
