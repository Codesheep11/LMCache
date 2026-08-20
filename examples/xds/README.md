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
模式。CascadeKV 的 `max_local_cpu_size` 必须大于 0，因为它为 CPU staging
分配器提供容量；`local_cpu: false` 表示不把它作为独立热缓存层使用。

## 启动外部 xDS 服务

以下命令使用本机已有的 TP4/NUMA1 服务配置：

```bash
sudo -E env XDS_SERVICE_ALLOWED_UID=0 \
  /home/yangxc/workspace/xDS/src/service/xds_service \
  --config /home/yangxc/workspace/xDS/configs/setup_qwen3_30b_tp4_numa1.json \
  --service-name xds0
```

服务成功后应出现 `/tmp/xds_xds0.sock` 和 `/tmp/xds_xds0.shm`。服务配置中的
queue 数、GPU BDF、blob 大小与推理的 TP 数必须匹配。退出推理进程后再用
`Ctrl-C` 正常关闭服务。

## vLLM 0.18.0

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
vllm serve /mnt/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled \
  --tensor-parallel-size 4 \
  --disable-prefix-caching \
  --kv-transfer-config \
  '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

## SGLang 0.5.9

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python -m sglang.launch_server \
  --model-path /mnt/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled \
  --tp 4 \
  --page-size 32 \
  --enable-lmcache
```

验证外部缓存时，连续发送两次相同且超过一个 LMCache chunk 的 prompt。关闭
vLLM prefix caching；SGLang 则在第一次请求完成后调用一次 `/flush_cache` 清空
RadixCache，再发送第二次请求。`/flush_cache` 不会删除 LMCache/xDS 中已持久化的
KV，因此第二次请求应从 xDS 恢复。不要给 SGLang 加 `--disable-radix-cache`，
因为 LMCache connector 本身挂载在 `LMCRadixCache` 上。最后同时检查 LMCache
与 xDS 服务日志中的 store/retrieve 完成记录。
