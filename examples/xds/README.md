# xDS 与 CascadeKV 独立后端

这两个配置连接同一个名为 `xds0` 的外部 xDS 服务，但数据路径互斥：

- `direct.yaml`：SSD 直接读写服务注册的 GPU buffer。
- `cascadekv.yaml`：SSD 先读写服务管理的 CPU staging buffer，再由 LMCache
  connector 在 CPU 与 GPU 之间传输。

不要同时开启 `xds_backend` 和 `cascadekv_backend`。直接后端要求服务配置中的
`xds_enable_direct` 为 `true`；CascadeKV 不要求 GPU direct，但仍然使用同一个
xDS 服务和 SSD blob pool。

## 环境

```bash
conda activate xds043
export LMCACHE_CONFIG_FILE=/home/yangxc/workspace/LMCache-v043-xds/examples/xds/direct.yaml
```

切换到 CascadeKV 时只需修改配置文件环境变量：

```bash
export LMCACHE_CONFIG_FILE=/home/yangxc/workspace/LMCache-v043-xds/examples/xds/cascadekv.yaml
```

`use_layerwise: true` 同时满足 SGLang connector 的要求；vLLM 0.18.0 也支持该
模式。CascadeKV 的 staging 容量由 `cascadekv_slot_count × xDS blob_size`
决定；保持 `local_cpu: false` 和 `max_local_cpu_size: 0`，可以避免额外创建
LocalCPUBackend，确保命中必须经过 xDS。

示例把 `chunk_size` 设为 32。xDS direct fast-map 要求一个 blob 对应一个 SPDK
cluster；本机服务的 blob/cluster 都是 6 MiB。在 `use_layerwise: true` 下，
每个 xDS blob 保存一个 layer 的一个 LMCache chunk。更换模型、TP、KV dtype
或关闭 layerwise 后，必须保证每次提交的对象不大于
`xds_blob_size_bytes`；否则应减小 `chunk_size`，不要直接修改已经格式化的
blobstore cluster 大小。

direct 后端还把 xDS 服务配置的 `xds_gpu_buffer_size_bytes` 作为注册 GPU
allocator。启用 `xds_windowed_io` 后，注册池只需至少容纳
`(xds_read_slot_count + xds_write_slot_count) × 对齐后的 xDS blob_size`，不再按
prompt 的总 chunk 数预分配 staging 显存。例如读槽位为 2、写槽位为 8 时，
注册池至少需要 10 个 blob；4K 或更长 prompt 会循环复用这些槽位。服务侧 SSD
blob pool 仍需有足够的持久化容量。

`xds_read_window_size` 和 `xds_write_window_size` 只用于逻辑窗口统计，实际同时
驻留的 staging 对象数量由 read/write slot count 限制，xDS 同时在途请求数由
`xds_io_queue_depth` 限制。`xds_async_write: true` 时，每个异步 lease 同样不超过
写槽位预算，并在下一批 GPU gather 前施加反压。

当前有界 direct hook 由非 layerwise 的 `store()` / `retrieve()` 路径调用。
`use_layerwise: true` 的 SGLang 路径继续使用原有逐层接口，不应把本节的长
prompt 有界 staging 结论直接套用到逐层模式。

## 启动外部 xDS 服务

以下命令使用本机已有的 TP4/NUMA1 服务配置：

```bash
sudo -E env XDS_SERVICE_ALLOWED_UID="$(id -u)" \
  /home/yangxc/workspace/xDS/src/service/xds_service \
  --config /home/yangxc/workspace/xDS/configs/setup_qwen3_30b_tp4_numa1.json \
  --service-name xds0
```

服务成功后应出现 `/tmp/xds_xds0.sock` 和 `/tmp/xds_xds0.shm`。服务配置中的
queue 数至少要覆盖推理 rank 数；GPU/SSD PCIe 拓扑、blob 大小也必须与实际
部署匹配。退出推理进程后再用 `Ctrl-C` 正常关闭服务。

## vLLM 0.18.0

```bash
PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=4 \
vllm serve /mnt/adminserver-nfsrdma/Model/Qwen3-0.6B \
  --max-model-len 4096 \
  --no-enable-prefix-caching \
  --kv-transfer-config \
  '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

## SGLang 0.5.9

```bash
PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=4 \
python -m sglang.launch_server \
  --model-path /mnt/adminserver-nfsrdma/Model/Qwen3-0.6B \
  --tp 1 \
  --page-size 32 \
  --enable-lmcache
```

验证外部缓存时，连续发送两次相同且超过一个 LMCache chunk 的 prompt。关闭
vLLM prefix caching；SGLang 则在第一次请求完成后调用一次 `/flush_cache` 清空
RadixCache，再发送第二次请求。`/flush_cache` 不会删除 LMCache/xDS 中已持久化的
KV，因此第二次请求应从 xDS 恢复。不要给 SGLang 加 `--disable-radix-cache`，
因为 LMCache connector 本身挂载在 `LMCRadixCache` 上。最后同时检查 LMCache
与 xDS 服务日志中的 store/retrieve 完成记录。
