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
allocator。它必须足以容纳一次 store 阶段同时收集的 KV；容量不足时 LMCache
会记录 `no enough memory`，并只保存能容纳的前缀。例如本机 64 MiB 配置配合
Qwen3-0.6B 可完整验证约 500 token 的短请求；若要缓存 4K prompt，应在启动
xDS 服务前把该注册池调大到覆盖对应 KV 总字节数。

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
