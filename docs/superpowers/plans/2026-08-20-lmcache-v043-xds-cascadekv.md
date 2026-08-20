# LMCache v0.4.3 xDS 与 CascadeKV 后端实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 LMCache v0.4.3 上实现共享外部 xDS 服务的 `XDSBackend`（SSD→GPU）和 `CascadeKVBackend`（SSD→CPU→GPU），并在单一 PyTorch 2.10 源码环境中验证 vLLM 0.18.0 与 SGLang 0.5.9。

**Architecture:** 两个后端复用 rank-local `XDSClient` 和 blob 元数据，但各自拥有 buffer 生命周期。`XDSBackend` 是 allocator backend，直接管理并注册 GPU tensor pool；`CascadeKVBackend` 同时实现 storage/allocator backend，使用 `XDSClient.alloc_buffer()` 建立有界 CPU staging pool，并让 LMCache 现有 GPU connector 完成 CPU/GPU 复制。

**Tech Stack:** Python 3.12、PyTorch 2.10.0+cu130、LMCache 0.4.3、vLLM 0.18.0、SGLang 0.5.9、pytest、xDS Python SDK/C API、CUDA、SPDK。

## Global Constraints

- 工作目录固定为 `/home/yangxc/workspace/LMCache-v043-xds`，分支固定为 `v0.4.3-xds`。
- 旧目录 `/home/yangxc/workspace/LMCache` 只读参考，不修改、不清理其暂存区或工作区。
- 仓库、发行包和 import 名均保持 `lmcache`。
- 配置 `xds_backend` 与 `cascadekv_backend` 互斥。
- 两个后端都连接 `xds_service_name` 指向的外部 xDS 服务；不调用旧进程内 `xds.init_blob_pool()`。
- `XDSBackend` 使用 `require_direct=True`；`CascadeKVBackend` 使用 `require_direct=False`。
- 唯一新环境为 `xds043`；PyTorch 固定 `2.10.0`，不得为 SGLang 降级。
- LMCache、xDS SDK、vLLM 和 SGLang 都必须 editable 源码安装。
- 生产代码遵循测试先行：每项行为先观察到预期失败，再实现最小代码并观察通过。

---

### Task 1: 建立干净基线和 xDS 配置契约

**Files:**
- Modify: `lmcache/v1/config.py`
- Modify: `tests/v1/test_config.py`

**Interfaces:**
- Produces: `LMCacheEngineConfig.xds_service_name: Optional[str]`
- Produces: `LMCacheEngineConfig.xds_backend: bool`
- Produces: `LMCacheEngineConfig.cascadekv_backend: bool`
- Produces: direct I/O 和 CascadeKV slot 的整数配置项。
- Consumes: v0.4.3 `create_config_class()`、YAML/env 加载和 `validate()`。

- [ ] **Step 1: 运行未修改基线测试**

Run:

```bash
conda run -n xds python -m pytest tests/v1/test_config.py -q
```

Expected: 当前 v0.4.3 配置测试通过；如果旧环境依赖导致收集失败，记录完整错误，并用 `PYTHONPATH=$PWD` 重跑确认错误来自依赖而非源码。

- [ ] **Step 2: 写入失败的配置测试**

在 `tests/v1/test_config.py` 增加：

```python
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {
                "xds_service_name": "xds0",
                "xds_backend": True,
                "cascadekv_backend": True,
                "xds_require_direct": True,
            },
            "cannot both be enabled",
        ),
        ({"xds_backend": True}, "xds_service_name"),
        (
            {"xds_service_name": "xds0", "xds_backend": True},
            "xds_require_direct",
        ),
        (
            {
                "xds_service_name": "xds0",
                "cascadekv_backend": True,
                "xds_require_direct": True,
            },
            "require_direct=false",
        ),
        (
            {
                "xds_service_name": "xds0",
                "cascadekv_backend": True,
                "cascadekv_slot_count": 0,
            },
            "cascadekv_slot_count",
        ),
    ],
)
def test_xds_backend_config_rejects_invalid_combinations(overrides, message):
    config = LMCacheEngineConfig.from_defaults()
    for name, value in overrides.items():
        setattr(config, name, value)
    with pytest.raises(ValueError, match=message):
        config.validate()


def test_xds_backend_config_accepts_each_service_path(monkeypatch):
    direct = LMCacheEngineConfig.from_defaults()
    direct.xds_service_name = "xds0"
    direct.xds_backend = True
    direct.xds_require_direct = True
    direct.validate()

    cascade = LMCacheEngineConfig.from_defaults()
    cascade.xds_service_name = "xds0"
    cascade.cascadekv_backend = True
    cascade.cascadekv_slot_count = 4
    cascade.validate()

    monkeypatch.setenv("LMCACHE_CASCADEKV_BACKEND", "true")
    monkeypatch.setenv("LMCACHE_XDS_SERVICE_NAME", "xds-env")
    from_env = LMCacheEngineConfig.from_env()
    assert from_env.cascadekv_backend is True
    assert from_env.xds_service_name == "xds-env"
```

- [ ] **Step 3: 验证测试因配置项缺失而失败**

Run:

```bash
python -m pytest tests/v1/test_config.py -k xds_backend_config -q
```

Expected: FAIL，错误指向 `LMCacheEngineConfig` 缺少 `xds_service_name`、`xds_backend` 或 `cascadekv_backend`。

- [ ] **Step 4: 在单一配置定义中心加入字段和校验**

在 `_CONFIG_DEFINITIONS` 加入：

```python
    "xds_service_name": {
        "type": Optional[str],
        "default": None,
        "env_converter": str,
    },
    "xds_backend": {"type": bool, "default": False, "env_converter": _to_bool},
    "cascadekv_backend": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
    },
    "xds_require_direct": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
    },
    "xds_windowed_io": {
        "type": bool,
        "default": True,
        "env_converter": _to_bool,
    },
    "xds_read_window_size": {"type": int, "default": 64, "env_converter": int},
    "xds_write_window_size": {"type": int, "default": 64, "env_converter": int},
    "xds_io_queue_depth": {"type": int, "default": 1, "env_converter": int},
    "xds_read_slot_count": {"type": int, "default": 2, "env_converter": int},
    "xds_write_slot_count": {"type": int, "default": 8, "env_converter": int},
    "xds_async_write": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
    },
    "cascadekv_slot_count": {"type": int, "default": 8, "env_converter": int},
```

在 `_validate_config()` 末尾、`return self` 之前加入显式 `ValueError` 校验，错误文本与测试完全一致；正数检查仅在相应后端启用时执行。

- [ ] **Step 5: 验证配置测试并提交**

Run:

```bash
python -m pytest tests/v1/test_config.py -q
git diff --check
git add lmcache/v1/config.py tests/v1/test_config.py
git commit -m "feat: define xds backend configuration contract"
```

Expected: 配置测试全部通过，提交只包含两个文件。

---

### Task 2: 建立共享 xDS client 和 blob 元数据

**Files:**
- Create: `lmcache/v1/xds_utils.py`
- Create: `tests/v1/test_xds_utils.py`

**Interfaces:**
- Produces: `XDSBlobMetadata` dataclass。
- Produces: `align_to_io_unit(size: int, io_unit_size: int) -> int`。
- Produces: `connect_xds_client(config, metadata, connect_fn=None) -> Any`。
- Consumes: `LMCacheMetadata.get_shapes()`、`get_dtypes()` 与 xDS `connect()`。

- [ ] **Step 1: 写入失败的 helper 测试**

测试使用一个返回 metadata 的假客户端：

```python
from types import SimpleNamespace
import torch

from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.xds_utils import align_to_io_unit, connect_xds_client


def test_align_to_io_unit_rejects_non_positive_values():
    assert align_to_io_unit(4097, 4096) == 8192
    with pytest.raises(ValueError, match="size must be positive"):
        align_to_io_unit(0, 4096)
    with pytest.raises(ValueError, match="io_unit_size must be positive"):
        align_to_io_unit(4096, 0)


def test_connect_xds_client_binds_rank_and_path(test_metadata):
    calls = []
    config = LMCacheEngineConfig.from_defaults()
    config.xds_service_name = "xds0"
    config.cascadekv_backend = True

    def connect(service_name, rank_id, world_size, require_direct):
        calls.append((service_name, rank_id, world_size, require_direct))
        return SimpleNamespace(
            metadata=SimpleNamespace(
                queue_id=test_metadata.worker_id,
                io_unit_size=4096,
                blob_size_bytes=4096,
                capacity_bytes=4096 * 32,
                gpu_buffer_size_bytes=4096 * 8,
            )
        )

    client = connect_xds_client(config, test_metadata, connect_fn=connect)
    assert client.metadata.queue_id == test_metadata.worker_id
    assert calls == [
        ("xds0", test_metadata.worker_id, test_metadata.world_size, False)
    ]
```

- [ ] **Step 2: 观察模块缺失失败**

Run:

```bash
python -m pytest tests/v1/test_xds_utils.py -q
```

Expected: FAIL with `ModuleNotFoundError: lmcache.v1.xds_utils`。

- [ ] **Step 3: 实现独立 helper 模块**

实现的核心类型和函数签名为：

```python
@dataclass
class XDSBlobMetadata:
    blob_handle: int
    size: int
    shapes: list[torch.Size]
    dtypes: list[torch.dtype]
    fmt: MemoryFormat
    pin_count: int = 0

    def pin(self) -> bool:
        self.pin_count += 1
        return True

    def unpin(self) -> bool:
        if self.pin_count == 0:
            return False
        self.pin_count -= 1
        return True

    @property
    def is_pinned(self) -> bool:
        return self.pin_count > 0


def align_to_io_unit(size: int, io_unit_size: int) -> int:
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    if io_unit_size <= 0:
        raise ValueError(f"io_unit_size must be positive, got {io_unit_size}")
    return ((size + io_unit_size - 1) // io_unit_size) * io_unit_size
```

`connect_xds_client()` 必须：惰性 `import xds`；根据所选后端计算 `require_direct`；校验 queue/rank；校验 blob size 至少能够容纳 `metadata.get_shapes(metadata.chunk_size)` 的总字节数并满足 I/O 对齐；连接失败信息包含 service name 和 rank。

- [ ] **Step 4: 验证 helper 测试并提交**

Run:

```bash
python -m pytest tests/v1/test_xds_utils.py -q
git diff --check
git add lmcache/v1/xds_utils.py tests/v1/test_xds_utils.py
git commit -m "feat: add shared xds client utilities"
```

Expected: 新 helper 测试全部通过。

---

### Task 3: 实现 CascadeKV 的服务 CPU staging allocator

**Files:**
- Create: `lmcache/v1/storage_backend/cascadekv_backend.py`
- Create: `tests/v1/storage_backend/test_cascadekv_backend.py`

**Interfaces:**
- Produces: `CascadeKVBackend(AllocatorBackendInterface)`。
- Produces: `initialize_allocator(config, metadata) -> TensorMemoryAllocator`。
- Consumes: `XDSClient.alloc_buffer()`/`free_buffer()`、`torch.frombuffer()`、`TensorMemoryAllocator`。

- [ ] **Step 1: 写 staging pool 生命周期失败测试**

```python
def test_cascadekv_allocator_wraps_service_cpu_buffer_without_copy(
    cascade_config, test_metadata, fake_xds_client
):
    backend = CascadeKVBackend(
        cascade_config,
        test_metadata,
        fake_xds_client,
        dst_device="cpu",
    )
    obj = backend.allocate(
        test_metadata.get_shapes(),
        test_metadata.get_dtypes(),
        MemoryFormat.KV_2LTD,
    )
    assert obj is not None
    assert obj.raw_data.data_ptr() >= backend.staging_base_ptr
    assert obj.raw_data.data_ptr() < backend.staging_base_ptr + backend.staging_size
    assert fake_xds_client.alloc_sizes == [backend.staging_size]
    obj.ref_count_down()
    backend.close()
    assert fake_xds_client.free_refs == [backend.staging_buffer_ref]
```

`fake_xds_client.alloc_buffer(size)` 返回 `memoryview(bytearray(size))` 和固定正整数 ref，并记录调用。

- [ ] **Step 2: 观察类缺失失败**

Run:

```bash
python -m pytest tests/v1/storage_backend/test_cascadekv_backend.py -k allocator -q
```

Expected: FAIL because `CascadeKVBackend` 尚不存在。

- [ ] **Step 3: 实现 staging pool 和 allocator delegation**

构造函数必须按以下顺序完成：

```python
self.client = xds_client
self.io_unit_size = int(xds_client.metadata.io_unit_size)
self.blob_size = align_to_io_unit(
    int(xds_client.metadata.blob_size_bytes), self.io_unit_size
)
self.staging_size = config.cascadekv_slot_count * self.blob_size
self.staging_view, self.staging_buffer_ref = xds_client.alloc_buffer(
    self.staging_size
)
self.staging_tensor = torch.frombuffer(self.staging_view, dtype=torch.uint8)
self.staging_base_ptr = self.staging_tensor.data_ptr()
self.memory_allocator = TensorMemoryAllocator(
    self.staging_tensor,
    align_bytes=self.io_unit_size,
)
```

`allocate()`、`batched_allocate()`、`get_memory_allocator()` 委托给该 allocator；`get_allocator_backend()` 返回 `self`；`calculate_chunk_budget()` 返回 `cascadekv_slot_count`。`close()` 幂等，并在所有 MemoryObj 释放后调用一次 `client.free_buffer(staging_buffer_ref)`。

- [ ] **Step 4: 验证 allocator 测试并提交**

Run:

```bash
python -m pytest tests/v1/storage_backend/test_cascadekv_backend.py -k allocator -q
git add lmcache/v1/storage_backend/cascadekv_backend.py tests/v1/storage_backend/test_cascadekv_backend.py
git commit -m "feat: add cascadekv cpu staging allocator"
```

Expected: allocator 生命周期测试通过。

---

### Task 4: 实现 CascadeKV SSD→CPU→GPU 的存取和回收

**Files:**
- Modify: `lmcache/v1/storage_backend/cascadekv_backend.py`
- Modify: `tests/v1/storage_backend/test_cascadekv_backend.py`

**Interfaces:**
- Produces: `batched_submit_put_task(keys, objs, transfer_spec=None, on_complete_callback=None)`。
- Produces: `batched_get_blocking(keys) -> list[Optional[MemoryObj]]`。
- Produces: contains/pin/unpin/remove/touch/async wrappers/close 的完整 `StorageBackendInterface`。

- [ ] **Step 1: 写读写顺序和失败清理测试**

新增三个独立测试：

```python
def test_cascadekv_read_uses_service_cpu_buffer(cascade_backend, stored_key):
    objects = cascade_backend.batched_get_blocking([stored_key])
    obj = objects[0]
    assert obj is not None
    assert cascade_backend.client.calls[:2] == ["submit_read", "wait_any_result"]
    assert obj.raw_data.data_ptr() >= cascade_backend.staging_base_ptr
    obj.ref_count_down()


def test_cascadekv_short_write_releases_blob_and_does_not_publish_key(
    cascade_backend, memory_obj, cache_key
):
    cascade_backend.client.next_completion.completed_bytes = memory_obj.get_size() - 1
    futures = cascade_backend.batched_submit_put_task([cache_key], [memory_obj])
    for future in futures or []:
        with pytest.raises(RuntimeError, match="short xDS write"):
            future.result()
    assert not cascade_backend.contains(cache_key)
    assert cascade_backend.client.released_blob_handles


def test_cascadekv_close_frees_buffers_before_client_close(cascade_backend):
    cascade_backend.close()
    assert cascade_backend.client.calls[-1] == "free_buffer"
    cascade_backend.close()
    assert cascade_backend.client.calls.count("free_buffer") == 1
```

- [ ] **Step 2: 观察接口未实现失败**

Run:

```bash
python -m pytest tests/v1/storage_backend/test_cascadekv_backend.py -k 'read or short_write or close' -q
```

Expected: FAIL due未实现的存取、完成校验或回收逻辑。

- [ ] **Step 3: 实现最小完整存取状态机**

实现规则：

```python
descriptor = (
    blob_handle,
    self.staging_buffer_ref,
    memory_obj.raw_data.data_ptr() - self.staging_base_ptr,
    0,
    memory_obj.get_physical_size(),
)
request_id = self.client.submit_read(*descriptor)
completion = self.client.wait_any_result(timeout=IO_TIMEOUT_SECS)
if completion.request_id != request_id:
    raise RuntimeError("unexpected xDS completion request id")
if completion.status != 0 or completion.completed_bytes != descriptor[-1]:
    raise RuntimeError("short xDS read")
```

写入使用同一描述符结构和 `submit_write()`。只有完整成功后才把 `XDSBlobMetadata` 放入受锁保护的 `OrderedDict`。失败时释放新 blob handle 并让 MemoryObj refcount 回到调用者预期状态。`remove()` 先从字典摘除，再释放 handle；pin count 非零时，非强制 remove 返回 false。

- [ ] **Step 4: 验证 CascadeKV 全套单测并提交**

Run:

```bash
python -m pytest tests/v1/storage_backend/test_cascadekv_backend.py -q
git diff --check
git add lmcache/v1/storage_backend/cascadekv_backend.py tests/v1/storage_backend/test_cascadekv_backend.py
git commit -m "feat: implement cascadekv service storage path"
```

Expected: CascadeKV 测试全部通过，无进程内 xDS API 调用。

---

### Task 5: 移植直接 XDSBackend 及窗口化 I/O

**Files:**
- Create: `lmcache/v1/storage_backend/xds_backend.py`
- Create: `lmcache/v1/storage_backend/xds_read_ring.py`
- Create: `lmcache/v1/storage_backend/xds_write_scheduler.py`
- Create: `lmcache/v1/storage_backend/transfer_result.py`
- Create: `tests/v1/storage_backend/test_xds_backend.py`
- Create: `tests/v1/storage_backend/test_xds_read_ring.py`
- Create: `tests/v1/storage_backend/test_xds_write_scheduler.py`

**Interfaces:**
- Produces: `XDSBackend(AllocatorBackendInterface)`，拥有 GPU `TensorMemoryAllocator`。
- Produces: 有界读 slot ring 和异步写 lease scheduler。
- Produces: 明确的逐 chunk 成功/失败结果。

- [ ] **Step 1: 写 GPU 注册、范围校验和短 I/O 回滚测试**

```python
def test_xds_backend_registers_exact_gpu_pool(direct_config, test_metadata, fake_client):
    backend = XDSBackend(direct_config, test_metadata, fake_client, "cuda:0")
    allocator = backend.get_memory_allocator()
    assert fake_client.registered == [
        (allocator.buffer.data_ptr(), allocator.buffer.numel())
    ]
    backend.close()
    assert fake_client.unregistered == [backend.buffer_ref]


def test_xds_backend_rejects_descriptor_outside_registered_pool(direct_backend):
    with pytest.raises(ValueError, match="registered range"):
        direct_backend._buffer_descriptor(
            direct_backend.registered_size - 4096,
            8192,
        )


def test_xds_backend_short_write_rolls_back_lease(
    direct_backend, memory_obj, cache_key
):
    direct_backend.client.next_completion.completed_bytes = 0
    futures = direct_backend.batched_submit_put_task([cache_key], [memory_obj])
    for future in futures or []:
        with pytest.raises(RuntimeError, match="short xDS write"):
            future.result()
    assert not direct_backend.contains(cache_key)
    assert direct_backend.client.released_blob_handles
```

- [ ] **Step 2: 观察模块缺失失败**

Run:

```bash
python -m pytest tests/v1/storage_backend/test_xds_backend.py -q
```

Expected: FAIL because direct backend 模块不存在。

- [ ] **Step 3: 移植通用完成校验和 GPU allocator**

`XDSBackend` 构造 GPU pool：

```python
self.registered_size = int(xds_client.metadata.gpu_buffer_size_bytes)
self.gpu_tensor = torch.empty(
    self.registered_size,
    dtype=torch.uint8,
    device=dst_device,
)
self.memory_allocator = TensorMemoryAllocator(
    self.gpu_tensor,
    align_bytes=int(xds_client.metadata.io_unit_size),
)
self.buffer_ref = xds_client.register_gpu_buffer(
    self.gpu_tensor.data_ptr(), self.registered_size
)
```

`_buffer_descriptor(memory_obj)` 使用 `raw_data.data_ptr() - gpu_tensor.data_ptr()` 计算 offset，并拒绝负 offset、越界和非 I/O unit 对齐。读写仅调用 service client API。

- [ ] **Step 4: 测试先行移植 read ring 和 write scheduler**

从旧工作区只迁移算法，不迁移旧 API。测试必须覆盖：slot 不重复占用、完成后归还、异常归还、write budget 背压、commit/rollback 只能发生一次、close 排空未完成 lease。每个模块先运行各自测试观察失败，再写最小实现：

```bash
python -m pytest tests/v1/storage_backend/test_xds_read_ring.py -q
python -m pytest tests/v1/storage_backend/test_xds_write_scheduler.py -q
```

Expected after implementation: 两个测试文件全部通过。

- [ ] **Step 5: 完成 direct backend 接口并验证**

Run:

```bash
python -m pytest \
  tests/v1/storage_backend/test_xds_backend.py \
  tests/v1/storage_backend/test_xds_read_ring.py \
  tests/v1/storage_backend/test_xds_write_scheduler.py -q
git diff --check
git add lmcache/v1/storage_backend/xds_backend.py \
  lmcache/v1/storage_backend/xds_read_ring.py \
  lmcache/v1/storage_backend/xds_write_scheduler.py \
  lmcache/v1/storage_backend/transfer_result.py \
  tests/v1/storage_backend/test_xds_backend.py \
  tests/v1/storage_backend/test_xds_read_ring.py \
  tests/v1/storage_backend/test_xds_write_scheduler.py
git commit -m "feat: port direct xds storage backend"
```

Expected: direct backend 相关测试全部通过，`close()` 只注销一次 GPU buffer。

---

### Task 6: 接入后端工厂、StorageManager 和 CacheEngine 生命周期

**Files:**
- Modify: `lmcache/v1/storage_backend/__init__.py`
- Modify: `lmcache/v1/storage_backend/storage_manager.py`
- Modify: `lmcache/v1/cache_engine.py`
- Create: `tests/v1/storage_backend/test_xds_backend_factory.py`
- Modify: `tests/v1/storage_backend/test_storage_manager.py`
- Modify: `tests/v1/test_cache_engine.py`

**Interfaces:**
- Consumes: Task 2 的 `connect_xds_client()`。
- Produces: factory 对两个后端的互斥惰性构造。
- Produces: direct 模式选择 `XDSBackend` 为全局 allocator；Cascade 模式保留 `LocalCPUBackend` 为全局 allocator。
- Produces: backend buffers 先于共享 client 关闭。

- [ ] **Step 1: 写失败的 factory 选择测试**

```python
@pytest.mark.parametrize(
    ("direct", "cascade", "expected"),
    [(True, False, "XDSBackend"), (False, True, "CascadeKVBackend")],
)
def test_factory_constructs_only_selected_xds_backend(
    direct, cascade, expected, config, test_metadata, monkeypatch
):
    config.xds_service_name = "xds0"
    config.xds_backend = direct
    config.cascadekv_backend = cascade
    config.xds_require_direct = direct
    fake_client = FakeXDSClient(require_direct=direct)
    monkeypatch.setattr(
        "lmcache.v1.storage_backend.connect_xds_client",
        lambda *_args, **_kwargs: fake_client,
    )
    backends = CreateStorageBackends(config, test_metadata, asyncio.new_event_loop())
    assert expected in backends
    assert ({"XDSBackend", "CascadeKVBackend"} & backends.keys()) == {expected}
```

- [ ] **Step 2: 观察 factory 不认识配置失败**

Run:

```bash
python -m pytest tests/v1/storage_backend/test_xds_backend_factory.py -q
```

Expected: FAIL because `CreateStorageBackends` 尚未构造新后端。

- [ ] **Step 3: 实现惰性构造和 allocator 选择**

在 factory 中只在配置启用时导入类：

```python
if config.xds_backend or config.cascadekv_backend:
    client = connect_xds_client(config, metadata)
    if config.xds_backend:
        from lmcache.v1.storage_backend.xds_backend import XDSBackend

        backend = XDSBackend(config, metadata, client, dst_device)
    else:
        from lmcache.v1.storage_backend.cascadekv_backend import CascadeKVBackend

        backend = CascadeKVBackend(config, metadata, client, dst_device="cpu")
    storage_backends[str(backend)] = backend
```

`StorageManager._get_allocator_backend()` 在 direct 模式返回 `storage_backends["XDSBackend"]`；其他模式保持上游逻辑。将本地 CPU writeback 的硬编码名称判断改为后端属性：当 `backend.skip_local_cpu_writeback` 为 true 时不写回。两个 xDS 后端设置该属性，避免 staging slot 被本地 cache 长期持有。

- [ ] **Step 4: 测试关闭顺序和 CacheEngine GPU copy**

新增测试断言 `StorageManager.close()` 先调用 backend.close，再调用 client.close；direct 返回 GPU MemoryObj 时，现有 `gpu_connector.batched_to_gpu()` 仍收到对象；Cascade 返回 CPU MemoryObj 时走相同 connector 接口。

- [ ] **Step 5: 验证工厂/管理器/引擎并提交**

Run:

```bash
python -m pytest \
  tests/v1/storage_backend/test_xds_backend_factory.py \
  tests/v1/storage_backend/test_storage_manager.py \
  tests/v1/test_cache_engine.py -q
git diff --check
git add lmcache/v1/storage_backend/__init__.py \
  lmcache/v1/storage_backend/storage_manager.py \
  lmcache/v1/cache_engine.py \
  tests/v1/storage_backend/test_xds_backend_factory.py \
  tests/v1/storage_backend/test_storage_manager.py \
  tests/v1/test_cache_engine.py
git commit -m "feat: wire xds backends into lmcache engine"
```

Expected: 选择和关闭顺序测试通过。

---

### Task 7: LMCache 回归测试、静态检查和源码包验证

**Files:**
- Modify only if tests prove required: `lmcache/integration/vllm/vllm_v1_adapter.py`
- Modify only if tests prove required: `lmcache/integration/vllm/lmcache_connector_v1.py`
- Modify: backend tests from Tasks 1–6 as required by discovered regressions。

**Interfaces:**
- Ensures: 上游 v0.4.3 connector 保持默认实现。
- Ensures: `lmcache` editable install 从新工作树加载。

- [ ] **Step 1: 运行聚焦回归集合**

Run:

```bash
python -m pytest \
  tests/v1/test_config.py \
  tests/v1/test_xds_utils.py \
  tests/v1/storage_backend/test_cascadekv_backend.py \
  tests/v1/storage_backend/test_xds_backend.py \
  tests/v1/storage_backend/test_xds_backend_factory.py \
  tests/v1/storage_backend/test_storage_manager.py \
  tests/v1/test_cache_engine.py -q
```

Expected: 全部通过。如果 connector 测试未失败，不修改 connector 文件。

- [ ] **Step 2: 运行格式和类型相关检查**

Run:

```bash
ruff check lmcache/v1/xds_utils.py \
  lmcache/v1/storage_backend/xds_backend.py \
  lmcache/v1/storage_backend/cascadekv_backend.py \
  tests/v1/test_xds_utils.py \
  tests/v1/storage_backend/test_xds_backend.py \
  tests/v1/storage_backend/test_cascadekv_backend.py
git diff --check
```

Expected: 无 lint 和 whitespace 错误。

- [ ] **Step 3: 提交回归修正**

Run:

```bash
git status --short
git add \
  lmcache/integration/vllm/vllm_v1_adapter.py \
  lmcache/integration/vllm/lmcache_connector_v1.py \
  lmcache/v1/xds_utils.py \
  lmcache/v1/storage_backend/xds_backend.py \
  lmcache/v1/storage_backend/cascadekv_backend.py \
  lmcache/v1/storage_backend/storage_manager.py \
  tests/v1/test_xds_utils.py \
  tests/v1/storage_backend/test_xds_backend.py \
  tests/v1/storage_backend/test_cascadekv_backend.py \
  tests/v1/storage_backend/test_storage_manager.py \
  tests/v1/test_cache_engine.py
git diff --cached --stat
if ! git diff --cached --quiet; then
  git commit -m "test: cover xds backend integration regressions"
fi
```

Expected: 若没有修正则不创建空提交；若有修正，提交只包含测试证明必需的文件。

---

### Task 8: 创建单一 PyTorch 2.10 源码环境并安装四个源码项目

**Files:**
- Create worktree: `/home/yangxc/workspace/vllm-v0180`
- Create worktree: `/home/yangxc/workspace/sglang-v059-torch210`
- Modify: `/home/yangxc/workspace/sglang-v059-torch210/python/pyproject.toml`
- Environment: `/home/yangxc/miniconda3/envs/xds043`

**Interfaces:**
- Produces: `xds043` 单一环境。
- Produces: editable `xds`、`lmcache`、`vllm`、`sglang`。
- Pins: torch 2.10.0、torchvision 0.25.0、torchaudio 2.10.0、torchao 0.16.0、torchcodec 0.10.0、transformers 4.57.1。

- [ ] **Step 1: 创建环境并安装 PyTorch 2.10 CUDA 13.0 组合**

Run:

```bash
conda create -n xds043 python=3.12 -y
conda run -n xds043 python -m pip install --upgrade pip setuptools wheel
conda run -n xds043 python -m pip install \
  torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu130
conda run -n xds043 python -m pip install \
  torchao==0.16.0 torchcodec==0.10.0 transformers==4.57.1
```

Expected: `python -c 'import torch; print(torch.__version__, torch.version.cuda)'` 输出 `2.10.0` 和 CUDA 13.0 变体。

- [ ] **Step 2: 创建 vLLM 与 SGLang 隔离源码工作树**

Run:

```bash
git -C /home/yangxc/workspace/vllm fetch origin refs/tags/v0.18.0:refs/tags/v0.18.0
git -C /home/yangxc/workspace/vllm worktree add \
  /home/yangxc/workspace/vllm-v0180 -b v0.18.0-xds v0.18.0
git -C /home/yangxc/workspace/sglang fetch origin refs/tags/v0.5.9:refs/tags/v0.5.9
git -C /home/yangxc/workspace/sglang worktree add \
  /home/yangxc/workspace/sglang-v059-torch210 \
  -b v0.5.9-torch210-xds v0.5.9
```

Expected: 两个新工作树 clean，原 vLLM/SGLang 工作区保持不变。

- [ ] **Step 3: 先写 SGLang 依赖元数据检查，再修改版本**

Run before modification:

```bash
rg -n 'torch==|torchaudio==|torchao==|torchcodec==|transformers==|flashinfer' \
  /home/yangxc/workspace/sglang-v059-torch210/python/pyproject.toml
```

Expected: 显示 torch/torchaudio 2.9.1、torchao 0.9.0、torchcodec 0.8.0、transformers 4.57.1、FlashInfer 0.6.3。

将依赖元数据改为：

```toml
  "flashinfer_python==0.6.6",
  "flashinfer_cubin==0.6.6",
  "torch==2.10.0",
  "torchao==0.16.0",
  "torchaudio==2.10.0",
  "torchcodec==0.10.0 ; sys_platform != 'linux' or (sys_platform == 'linux' and platform_machine != 'aarch64' and platform_machine != 'arm64' and platform_machine != 'armv7l')",
  "torchvision==0.25.0",
  "transformers==4.57.1",
```

Run after modification:

```bash
git -C /home/yangxc/workspace/sglang-v059-torch210 diff --check
git -C /home/yangxc/workspace/sglang-v059-torch210 add python/pyproject.toml
git -C /home/yangxc/workspace/sglang-v059-torch210 commit \
  -m "build: align sglang with torch 2.10 stack"
```

Expected: SGLang 源码分支记录明确的依赖适配提交。

- [ ] **Step 4: 按顺序 editable 安装源码**

Run:

```bash
conda run -n xds043 python -m pip install -e /home/yangxc/workspace/xDS/sdk \
  --no-build-isolation
conda run -n xds043 python -m pip install -e /home/yangxc/workspace/LMCache-v043-xds \
  --no-build-isolation
conda run -n xds043 python -m pip install -r \
  /home/yangxc/workspace/vllm-v0180/requirements/build.txt
conda run -n xds043 python -m pip install -e /home/yangxc/workspace/vllm-v0180 \
  --no-build-isolation
conda run -n xds043 python -m pip install -e \
  /home/yangxc/workspace/sglang-v059-torch210/python --no-build-isolation
```

Expected: 四个项目均安装成功，未替换 torch 2.10.0。

- [ ] **Step 5: 验证版本、加载路径和依赖一致性**

Run:

```bash
conda run -n xds043 python -c \
  'import inspect,lmcache,sglang,torch,transformers,vllm,xds; print(torch.__version__); print(transformers.__version__); print(vllm.__version__); print(inspect.getfile(lmcache)); print(inspect.getfile(vllm)); print(inspect.getfile(sglang)); print(inspect.getfile(xds))'
conda run -n xds043 python -m pip check
```

Expected: torch 2.10.0、transformers 4.57.1、vLLM 0.18.0；三个源码包路径指向新工作树。`pip check` 无版本冲突。若 `sgl-kernel` 在 import 时出现 PyTorch ABI 错误，只针对该 kernel 进行 PyTorch 2.10 源码重建，不修改 torch 版本。

---

### Task 9: xDS 服务硬件验证和重复 prompt 验收

**Files:**
- Create: `examples/xds/cascadekv.yaml`
- Create: `examples/xds/direct.yaml`
- Create: `examples/xds/README.md`
- Runtime logs: `/tmp/lmcache-xds-validation/`

**Interfaces:**
- Consumes: 运行中的外部 xDS 服务、可用模型路径、GPU。
- Produces: vLLM 和 SGLang 对 direct/Cascade 路径的可复现启动命令和 cache-hit 证据。

- [ ] **Step 1: 写两份互斥配置和运行说明**

`direct.yaml`：

```yaml
chunk_size: 256
local_cpu: false
max_local_cpu_size: 0
xds_service_name: xds0
xds_backend: true
xds_require_direct: true
xds_windowed_io: true
xds_read_window_size: 64
xds_write_window_size: 64
xds_io_queue_depth: 1
xds_read_slot_count: 2
xds_write_slot_count: 8
xds_async_write: false
cascadekv_backend: false
```

`cascadekv.yaml`：

```yaml
chunk_size: 256
local_cpu: true
max_local_cpu_size: 5
xds_service_name: xds0
xds_backend: false
xds_require_direct: false
cascadekv_backend: true
cascadekv_slot_count: 8
```

- [ ] **Step 2: 运行 unit/regression 验证后再启动服务**

Run:

```bash
conda run -n xds043 python -m pytest \
  tests/v1/test_config.py \
  tests/v1/test_xds_utils.py \
  tests/v1/storage_backend/test_cascadekv_backend.py \
  tests/v1/storage_backend/test_xds_backend.py \
  tests/v1/storage_backend/test_xds_backend_factory.py -q
nvidia-smi
```

Expected: 测试通过，GPU 可见。

- [ ] **Step 3: 用 vLLM 0.18.0 分别验证两个后端**

对每份配置设置 `LMCACHE_CONFIG_FILE`，使用：

```bash
vllm serve "$MODEL_PATH" \
  --host 127.0.0.1 \
  --port 8000 \
  --kv-transfer-config \
  '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

连续发送两次完全相同且超过一个 chunk 的 prompt。Expected: 第一次日志包含 store，第二次包含 LMCache retrieved/hit tokens；direct 日志包含 `XDSBackend`，Cascade 日志包含 `CascadeKVBackend`。禁用 vLLM 内部 prefix cache 或根据 LMCache 日志区分外部命中。

- [ ] **Step 4: 用 SGLang v0.5.9 验证 LMCache 集成**

Run:

```bash
python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --host 127.0.0.1 \
  --port 30000 \
  --enable-lmcache
```

连续发送同一 prompt 两次。Expected: 第二次由 LMCache 报告 xDS 后端 cache hit，生成结果正常，无 torch/sgl-kernel ABI 错误。

- [ ] **Step 5: 保存验证命令、日志摘要并提交文档**

Run:

```bash
git add examples/xds/direct.yaml examples/xds/cascadekv.yaml examples/xds/README.md
git commit -m "docs: add xds backend source validation guide"
git status --short --branch
```

Expected: LMCache 工作树 clean；README 记录实际模型、命令、服务名和 direct/Cascade 两种命中证据。

---

## 完成前检查

- [ ] `git log --oneline v0.4.3..HEAD` 中每个提交职责单一。
- [ ] `git diff v0.4.3...HEAD --check` 无 whitespace 错误。
- [ ] 新后端测试、配置测试、StorageManager 和 CacheEngine 测试通过。
- [ ] `pip check` 在 `xds043` 中通过。
- [ ] `inspect.getfile()` 证明四个项目均从预期源码路径加载。
- [ ] vLLM direct、vLLM Cascade、SGLang 至少各有一次重复 prompt 的外部 cache-hit 证据。
- [ ] 原 `/home/yangxc/workspace/LMCache`、`vllm`、`sglang` 和 Conda `xds` 环境未被清理或覆盖。
