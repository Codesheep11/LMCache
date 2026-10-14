# CascadeKV Backend Design

## 配置

```yaml
chunk_size: 256
use_layerwise: false
local_cpu: false
max_local_cpu_size: 40       # 每 rank 的 Block Buffer，GiB

xds_service_name: xds0
xds_backend: false
cascadekv_backend: true
xds_max_size: 256            # 每 rank 的 SSD 容量，GiB

# xDS service
xds_blob_size_bytes: 6291456
xds_cluster_size_kb: 6144
```

CascadeKV 只支持 whole-block 路径。一个 6 MiB block 对应一个 6 MiB blob 和
一个 6 MiB cluster。Block Buffer 等待时间和 I/O 超时是后端内部常量。

## 每 rank 结构

```text
CascadeKVBackend
├── Block Buffer  ── XDSServiceMemoryAllocator
├── Buffer Table ── key -> BlockMemory
├── Block Table  ── key -> SSDBlock
└── StorageIOEngine ── xDS client owner thread
```

每个 rank 独占以上对象、shared-memory ring 和 SSD 容量状态。

## Block Buffer

一个 KV block 申请一个完整的 HugePage/DMA CPU `MemoryObj`，包含该 block 的
所有 layer。一个 block 对应一个 SSD blob、一次 SSD read/write 和一次完整
block H2D/D2H。

Block Buffer 只负责：

- `allocate` / `allocate_many`；
- 引用归零后释放；
- 内存不足时有界等待，超时报错；
- 批量读先原子预留全部 block，避免多个请求各占一部分后死锁。

## 两张 Table

`Buffer Table` 保存 D2H 已完成、SSD write 尚未完成的 block。命中后可立即 H2D，
不等待持久化。

`Block Table` 保存已持久化的 blob handle。`SSDBlock` 自己记录 reader、pin 和
LRU 时间；淘汰选择最旧且无 reader/pin 的 block。

两张表分别使用 RWLock。映射查询用读锁，增删用写锁；引用计数和 `SSDBlock`
状态使用对象短锁。所有 D2H、H2D、SSD I/O、xDS 调用及等待均在 table lock 外。

## 写路径

```text
GPU --D2H--> BlockMemory
                 |
                 +-- publish Buffer Table
                 +-- submit async SSD write --> return
                                                |
                                                +-- callback publish Block Table
                                                +-- remove Buffer Table entry
```

数据完整后才发布元数据。并发写相同 key 可以重复 D2H，但只有第一个成功发布
Buffer Table 的 block 提交 SSD write。callback 用对象身份检查，不能覆盖 remove
后同 key 的新 block。先发布 Block Table，再删除 Buffer Table，不产生不可见窗口。

## 读路径

1. 先查 Buffer Table；命中后持有 block 引用并直接 H2D。
2. 未命中则从 Block Table 获取 reader 引用。
3. 在锁外分配 request-private BlockMemory 并提交 SSD read。
4. 每个请求维持最多 4 个在途 block；任一 read completion 在 owner 线程立即
   提交该 block 的 H2D，并立刻补交下一个 block，形成滚动小窗口。
5. 全部 block 的 H2D 提交后记录一个 stream-tail event，只同步一次，再批量释放
   本请求的 BlockMemory；SSD reader 在各自 read completion 后释放。

vLLM 一个 forward batch 内的多个 retrieve 由 rank-local 线程池并发执行。每个
请求形成一条 4-block 滚动流水线；线程只准备和等待任务，不直接访问
shared-memory ring。

## StorageIOEngine

每 rank 一个 owner 线程，只有它调用 `get_blob`、`release_blob`、
`submit_read`、`submit_write` 和 `poll_burst`。业务线程通过 MPSC queue 提交任务，
因此 xDS shared-memory ring 保持单生产者/单消费者，不需要 client lock。

owner 线程轮转 ready task，并为每个读任务填满最多 4 个 descriptor。任一
completion 完成 H2D host-side 提交后，立即补齐该任务的窗口。多个请求同时存在时，
各自独立占用最多 4 个在途 block；CascadeKV 不设置全局请求数或 I/O QD 上限。
shared-memory ring 满时先 poll completion，再重试。超时按连续无 completion
进展计算。

xDS service 仅在所属 worker 有空闲 request context 时才消费 ring；请求进入 xDS
后由底层 pending queue 排队，设备侧并发由 xDS 内部控制。资源暂时饱和只让 ring
形成背压，不能向 CascadeKV 返回 `ENOMEM`。

SSD handle 未达到本 rank 容量时调用 `get_blob()`；达到容量后直接复用 Block Table
中最旧的可淘汰 handle。`remove` 产生的空闲 handle 由 owner 线程归还。

## 不变量

- 只存在 whole-block 数据路径，不存在 Layerwise 兼容分支；
- 数据先完成，元数据后发布；
- SSD write 异步提交，前台不等待持久化；
- 每个 SSD block 一个 descriptor；
- I/O 和 GPU copy 全部在 table lock 外；
- read/write 请求立即提交，可跨 task、跨 block overlap；
- close 排空已接受任务，再释放 blob 和 Block Buffer。
