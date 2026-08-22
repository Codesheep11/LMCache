# LMCache 0.4.3 CascadeKV vs Local Disk 端到端测试设计

## 目标

在同一套 vLLM 0.18.0 端到端长提示词负载下，对比当前 LMCache
`0.4.3+xds` 的 CascadeKV 后端和原生 Local Disk 后端。首轮只验证测试链路、
正确命中后端并获得一组可比较数据，不纳入移植前 CascadeKV。

## 固定环境

- Conda 环境：`xds043`
- LMCache 源码：`/home/yangxc/workspace/LMCache-v043-xds`
- vLLM 源码：`/home/yangxc/workspace/vllm-v0180`（`v0.18.0-xds`）
- 模型：`/mnt/adminserver-nfsrdma/Model/Qwen3-30B-A3B`
- GPU：4、5、6、7，tensor parallel size 为 4
- SSD：两个后端均使用 `0000:aa:00.0`、`0000:d3:00.0`、
  `0000:ec:00.0`、`0000:ed:00.0` 这四块已授权测试的 NUMA1 NVMe

## 后端配置

两个后端统一使用 `chunk_size: 256`、`use_layerwise: true`、
`save_unfull_chunk: true` 和 `local_cpu: false`。

CascadeKV 使用外部 xDS 服务和服务分配的有界 CPU staging slots；测试前验证
单个 layer chunk 的实际物理大小不超过现有 6 MiB blob。Local Disk 阶段使用
`switch_lmcache_test_ssds.sh` 将同四块 SSD 绑定到内核 NVMe 驱动，组成四盘 mdadm
RAID0 并格式化为 ext4；LMCache 在 `/mnt/lmcache-raid0/disk-cache` 上使用原生
buffered file I/O。该结果保留 Linux page cache 的默认行为，并在报告中明确标记为
buffered Local Disk。

## 工作负载

复用 `/home/yangxc/workspace/test` 的 `benchmark_prefix_caching.py` 方法：

- 固定随机种子 0
- 5 个不同 prompt，每个重复 20 次
- 输入长度固定为 40,000 tokens
- 输出 1 token
- `max_num_seqs=1`
- eager mode，最大模型长度 40,960
- vLLM automatic prefix caching 不启用，使重复请求必须经过 LMCache

第一次出现的 prompt 负责写入，后续重复 prompt 负责读取。模型初始化时间不计入
`llm.generate()` 的端到端耗时。

## 执行顺序与隔离

每个后端只执行一组，顺序固定为 Local Disk 后执行 CascadeKV。每组使用独立进程。
测试开始时通过已有 SSD 切换脚本重建四盘 mdadm RAID0/ext4，并清空专用缓存目录；
Local Disk 完成后卸载并销毁文件系统 RAID 元数据，再把同四块 SSD 绑定回
`uio_pci_generic`，最后启动干净的 xDS 服务实例执行 CascadeKV。测试结束后设备
保持为 xDS 状态。

## 验收与输出

每个后端必须同时满足：vLLM 请求成功、LMCache 日志显示完成写入和缓存读取、重复
请求存在非零缓存命中、没有 short I/O、越界、blob 尺寸或 xDS completion 错误。

结果写入 `/home/yangxc/workspace/test/results/` 的时间戳目录，保存：

- 完整 vLLM/LMCache 日志
- xDS 服务日志（CascadeKV）
- 实际使用的两份 LMCache 配置
- 总耗时、请求吞吐、每 prompt 平均耗时和缓存命中证据
- 环境版本、Git commit、GPU/SSD 映射及异常说明

单组数字作为功能验证和初步性能数据，不将其表述为多轮稳定性能结论。
