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
LMCache 后端迁移完成后进行集成验证。

## 版本与兼容性约束

- LMCache 基线：上游标签 `v0.4.3`。
- LMCache 工作分支：`v0.4.3-xds`。
- vLLM 目标版本：上游标签 `v0.18.0`。
- PyTorch 目标版本：`2.10.0`，与两个源码项目的构建元数据保持一致。
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
cascadekv_slot_count: 8
```

配置校验规则如下：

1. `xds_backend` 和 `cascadekv_backend` 不能同时为 true。
2. 开启任一后端时，必须提供非空的 `xds_service_name`。
3. `xds_backend` 要求 `xds_require_direct: true`。
4. `cascadekv_backend` 使用 `require_direct: false` 连接；CascadeKV 配置中设置
   `xds_require_direct: true` 将直接报错，避免数据路径含义不明确。
5. window size、queue depth 和 slot count 必须为正数。
6. 直接后端的 queue depth 不能超过 read slot count 或 write slot count。
7. `cascadekv_slot_count` 决定有界 CPU staging 容量，且必须为正数。

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
   拷贝、SSD 到 CPU 到 GPU 的调用顺序、GPU 到 CPU 到 SSD 的调用顺序、slot
   背压、超时清理、`free_buffer` 和关闭顺序。
5. 现有 LMCache v0.4.3 存储与配置测试保持通过。
6. 源码安装 smoke test 从预期源码路径导入 `lmcache`、`xds`、vLLM 和 SGLang，
   并输出固定版本。
7. 硬件验证启动 xDS 服务，分别使用两个后端通过 vLLM 0.18.0 连续发送两次
   相同的长 prompt。第二次请求必须报告外部缓存命中，日志必须明确标识所选
   数据路径。
8. 适配 SGLang 源码集成后重复一次 prompt 对，不修改 LMCache 后端接口。

## 验收标准

- 仓库名和导入包名保持 `lmcache`。
- `xds_backend: true` 选择并实际执行 SSD 到 GPU 的直接 I/O。
- `cascadekv_backend: true` 选择并实际执行通过外部 xDS 服务的 SSD 到 CPU 到
  GPU 路径。
- 同时开启两个后端的无效配置在资源分配前失败。
- 单元测试证明成功、超时、短 I/O 和关闭场景都能确定性清理资源。
- LMCache、vLLM 和 SGLang 在新的 Conda 环境中均为 editable 源码安装；现有
  已验证的 `xds` 环境保持可恢复。
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
