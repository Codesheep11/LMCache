# LMCache v0.4.3 CascadeKV 旧版行为等价迁移设计

**状态：** 已于 2026-08-22 在对话中确认

## 目标

将
`/home/yangxc/workspace/LMCache/lmcache/v1/storage_backend/cascadekv_backend.py`
作为 CascadeKV 行为的唯一基准，迁移到 LMCache v0.4.3 工作分支。除 LMCache
v0.4.3 接口适配和外部 xDS 服务 request-ID 接口适配外，后端的功能、调度、
并发、配置和异常语义均与旧版一致。

本设计取代
`2026-08-20-lmcache-v043-xds-cascadekv-design.md` 中 CascadeKV 的固定 slot、
有界 ring、queue-depth 和额外生命周期加固设计。`XDSBackend` 的 GPU direct
设计不受影响。

## 已确认约束

- `xds_backend` 与 `cascadekv_backend` 继续作为两个互斥的独立后端。
- CascadeKV 继续使用外部 xDS 服务，数据路径为
  `SSD -> xDS hugepage CPU memory -> GPU`。
- CascadeKV 不使用 Python `Future` 兼容层。
- CascadeKV 直接提交 xDS request ID，并按完成 ID 驱动旧版流水线。
- 不保留当前实现额外增加的固定 staging slot、全生命周期 `_io_lock`、
  poison、quarantine、active-operation close gate 和超时槽位保护。
- 已知旧版竞态、超时和关闭异常语义不作为本轮修复对象。
- xDS 服务端、共享内存协议和 SPDK I/O hot path 不因本迁移改变。

## 选定架构

### 旧后端为主体

直接以旧版 `CascadeKVBackend` 为实现主体，保留下列原始结构：

- `dict`、`put_tasks`、`prefetch_tasks` 及其原始锁边界；
- `write_buffer`、`flush_condition` 和 `xDS-FlushThread`；
- `LRUEvictor`、usage、pin/unpin、worker admit/evict、lookup 和 I/O trace；
- `batched_get_blocking`、connector overlap、blob overlap 和 chunk overlap；
- `get_non_blocking`、LocalCPU prefetch/writeback 和旧版关闭顺序。

只允许在方法签名、类型、构造参数和 xDS 调用表达上进行必要适配。当前
LMCache v0.4.3 CascadeKV 的 slot scheduler、read ring 和 lease 状态机不进入
迁移后的实现。

### 无 Future 的 xDS 完成路径

新版 xDS 服务 Python 接口以 request ID 表示异步请求：

```text
submit_read/write -> request_id
completion ring   -> completion(request_id, status, completed_bytes, ...)
```

CascadeKV 为每次旧版批量操作建立 request group。group 保存 request ID 到旧版
任务上下文的映射，并提供两种等待语义：

- `wait_all`：供旧版整批写和非 overlap 整批读使用；
- `wait_next`：供旧版 `FIRST_COMPLETED` blob/chunk overlap 使用。

一个 CascadeKV completion pump 是同一 rank-local `XDSClient` 的唯一完成队列
消费者。pump 非阻塞批量取得完成项，并将它们投递到所属 request group。
request group 使用 condition 和普通结果记录唤醒调用者，不创建
`concurrent.futures.Future`。

```text
read caller ---- submit IDs ----\
write flush ---- submit IDs -----+--> xDS service
                                  |
completion pump <-- poll_burst ---+
       |
       +--> read group: wait_next -> CPU-to-GPU copy
       +--> write group: wait_all -> publish metadata
```

客户端锁只覆盖短暂的 SDK 调用：批量 submit、blob control 操作和
`poll_burst`。等待 SSD、处理完成项和 CUDA copy 时不持有该锁，因此读写请求
可以同时在 xDS 服务中处于 inflight 状态。禁止恢复当前实现中覆盖整个 I/O
生命周期的全局 `_io_lock`。

### xDS SDK 最小修改

xDS C client 已实现 `xds_client_poll_burst()`，但 Python SDK 尚未公开。本轮只
补齐已有能力：

- `sdk/csrc/py_wrapper.c` 导出 `client_poll_burst(max_count)`；
- `sdk/src/xds/__init__.py` 增加 `XDSClient.poll_burst(max_count)`；
- Python 返回 `list[XDSCompletion]`，空完成队列返回空列表；
- 参数范围与 C 常量保持一致，C 错误按现有 Python 包装规则抛出。

不新增 Future、回调线程、服务命令或共享内存字段，也不修改 completion ring
生产逻辑。SDK 测试覆盖空 poll、单项、多项、最大 burst 和错误参数。

## 内存与 allocator

CascadeKV 不再按 `cascadekv_read_slot_count` 和
`cascadekv_write_slot_count` 划分固定 staging bank。后端通过
`XDSClient.alloc_buffer()` 从服务申请一块由 hugepage 支撑的大型 CPU DMA
buffer，并在 LMCache v0.4.3 中恢复与旧 `XDSDmaAllocator` 等价的显式空闲链表
allocator。

每个 `MemoryObj` 保存逻辑形状、dtype、物理大小以及相对该 service buffer 的
offset。提交 xDS I/O 时转换为同一个 `buffer_ref + buffer_offset`；CPU tensor
必须直接 view 到该 service memoryview，不允许额外 CPU-to-CPU copy。

为支持旧版 CPU-to-GPU 异步 copy，service buffer 原地址使用
`cudaHostRegister` 注册为 pinned host memory。释放顺序只做 LMCache 0.4.3 和
service API 所必需的转换，不增加旧版没有的操作门禁：停止旧 flush/prefetch
流程后注销 CUDA host mapping、释放 service buffer，再关闭 client。

LocalCPUBackend 必须使用这一 DMA allocator，使旧版 pending read、prefetch 和
writeback 操作仍然作用于 xDS 可访问的 hugepage 内存。CascadeKV 的
`get_allocator_backend()` 返回该 LocalCPU allocator backend，满足 LMCache
v0.4.3 的 allocator 选择接口。

## 保持一致的写路径

`batched_submit_put_task` 按旧版顺序完成去重、LRU 容量检查、引用计数增加和
`put_tasks` 发布，然后把任务追加到跨调用 `write_buffer`。

后台 flush 条件保持为：

- 缓冲项达到 `xds_write_window_size`；或
- 最老任务等待超过 `xds_max_flush_delay`；或
- 后端关闭时强制清空。

一次 flush 先为整批任务申请 blob，再连续调用 `submit_write()` 提交全部
request ID，最后通过 request group `wait_all` 等待整批完成。只有整批成功后才
执行旧版 `insert_key`、worker admit、usage 和统计更新。失败回滚、引用计数和
blob 释放顺序与旧版保持一致。

`contains()` 和 get 路径继续把 `put_tasks` 中的对象视为可命中，因此 SSD 写入
尚未完成时，同进程读取仍可直接取得 pending object。

## 保持一致的读路径

### 普通批量读取

非 overlap 路径先为所有命中项分配 CPU object，提交全部 read request ID，再
通过 `wait_all` 等待；返回顺序仍与输入 key 顺序一致。

### 多 blob overlap

为每个 blob 分配 host staging 和 GPU result，先提交全部 read request ID。
调用线程通过 request group `wait_next` 按真实完成顺序取得任务，并在旧版 copy
stream 上立即启动 CPU-to-GPU copy。所有 copy 完成后再同步 stream。

### 单 blob chunk overlap

按 `xds_chunked_size_kb` 和 xDS I/O unit 切分 blob，先提交全部 chunk request
ID。每个 chunk 完成后立即在 copy stream 上复制对应范围，保持旧版 SSD 读取和
H2D copy overlap。

### connector 与 prefetch

`batched_get_to_gpu_blocking`、`get_non_blocking`、异步 device read、LocalCPU
prefetch、metadata pin/unpin 和完成 callback 均保留旧版控制流。LMCache v0.4.3
新增的 `transfer_spec` 和完成回调仅作为接口参数接入，不改变旧版缓存状态机。

## 配置兼容

继续使用独立开关和服务连接配置：

```yaml
cascadekv_backend: true
xds_service_name: xds0
```

恢复并使用旧版 CascadeKV 配置：

```yaml
xds_max_size: 500
xds_write_window_size: 64
xds_idle_flush_time: 0.5
xds_max_flush_delay: 1.0
xds_enable_chunked_overlap: true
xds_chunked_size_kb: 4096
```

其中 `xds_max_size` 保持旧版单位和 LRU 容量含义，`max_local_cpu_size` 控制 DMA
hugepage allocator 的 CPU 容量。

当前实现新增的下列配置不再参与 CascadeKV 调度：

```text
cascadekv_slot_count
cascadekv_read_slot_count
cascadekv_write_slot_count
cascadekv_io_queue_depth
cascadekv_write_window_size
cascadekv_async_write
cascadekv_overlap
```

为避免旧配置文件解析失败，这些字段可暂时保留为 deprecated，但必须记录为
ignored，且不得改变运行行为。

## LMCache v0.4.3 接口适配边界

允许修改的兼容点只有：

- storage backend 构造参数和 factory 注入；
- `AllocatorBackendInterface` 所需 allocator 方法；
- `transfer_spec`、per-key completion callback 和异步批量方法签名；
- LMCache v0.4.3 `MemoryObj` 的 shapes/dtypes/format 表达；
- worker message sender、lookup server和 stats API 的名称变化；
- StorageManager 对 partial result、direct transfer hook 和 backend location 的
  v0.4.3 调用约定。

适配代码不得引入新的写窗口、slot backpressure、请求重排、完成顺序或失败
恢复策略。

## 测试策略

### xDS SDK

- C client 现有 `poll_burst` 测试保持通过；
- Python wrapper fake-C-API 测试验证 `XDSCompletion` 转换；
- 真实 service client smoke test 验证一次 burst 可返回多个完成项；
- 验证 poll 不会阻塞新的 submit。

### CascadeKV 行为等价测试

使用可控 mock client 记录提交和完成轨迹，逐项验证：

- 跨多次 `batched_submit_put_task` 的窗口聚合；
- threshold、max delay 和 close 三种 flush；
- 首次等待完成前已提交整批请求；
- read 和 write 可同时 inflight；
- `wait_next` 按完成顺序驱动 blob/chunk H2D；
- pending write 命中、prefetch 去重和 LocalCPU writeback；
- LRU 淘汰、pin/unpin、usage、worker、lookup、stats 和 trace；
- 超时、短 I/O、异常和 close 的结果与旧版一致；
- 超过当前 slot 默认数量的长 batch 不因 staging slot 数不足而失败。

测试以旧文件的外部行为和事件轨迹为断言基准，不以当前 v0.4.3 CascadeKV 的
slot/ring 私有状态为断言目标。

### 集成验证

在 `xds043` Conda 环境中运行 LMCache 单元测试和 lint，然后启动外部 xDS
服务与源码安装的 vLLM 0.18.0。连续发送两次相同长 prompt，第二次必须出现
CascadeKV 命中；日志与 xDS 统计需证明走过
`SSD -> CPU hugepage -> GPU`，并证明读写 overlap 已启用。

## 验收标准

- CascadeKV 实现主体和旧版方法/状态结构可逐项对应；
- 旧版公开配置及默认值生效；
- 当前固定 slot、queue-depth 和全生命周期 `_io_lock` 不再影响 CascadeKV；
- Python 代码中不使用 `Future` 表示 xDS service I/O；
- xDS 仅增加 Python `poll_burst` 暴露，不修改服务端和 hot path；
- 旧版写聚合、读取 overlap、prefetch、LRU、worker、lookup、stats 和 trace 均有
  行为等价测试；
- 长 prompt 不因固定 staging slot 数量不足而失败；
- vLLM 0.18.0 端到端重复 prompt 验证 CascadeKV 命中。
