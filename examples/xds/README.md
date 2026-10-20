# xDS 与 CascadeKV 独立后端

这两个配置连接同一个名为 `xds0` 的外部 xDS 服务，但数据路径互斥：

- `direct.yaml` / `readv.yaml`：默认通过 readv/writev 直接读写最终 vLLM HBM KV pages，
  不分配 KV staging pool。
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

默认直接读写要求非 layerwise、完整且对齐的 NHD KV blocks，以及同步存储。
保持 `local_cpu: false`、`max_local_cpu_size: 0`、`store_async: false` 和
`xds_async_write: false`。不支持的布局或未对齐 chunk 会报错，不会自动回退到 staging。
当前双向路径已在 Ascend 验证；不能据此认定所有 CUDA/SGLang connector 均已支持。

`xds_io_queue_depth` 控制每个客户端同时在途的 I/O 数量，不分配 payload buffer。
`xds_read_slot_count` 和 `xds_write_slot_count` 已从配置中移除。
KV 内存池在启动时一次注册，关闭时等待 DMA 完成再注销。
服务的 blob 必须容纳一个 chunk 的全部 KV；更换模型、TP 或 dtype 时需重新计算。

旧 staging 路径仅在 `extra_config.xds_direct_kv_io: false` 时启用，读写缓冲预算
各为 `xds_io_queue_depth × 对齐后的 blob_size`。`layerwise_readv.yaml` 显式选择
此兼容路径；它不能作为双向无 staging 的示例。CascadeKV 的 CPU staging 仍由
`cascadekv_slot_count × xDS blob_size` 控制，不受本次更改影响。

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

## SGLang 0.5.9（旧 layerwise 路径）

该路径需单独配置 `use_layerwise: true` 和
`extra_config.xds_direct_kv_io: false`，不能直接使用默认无 staging 的配置。

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
