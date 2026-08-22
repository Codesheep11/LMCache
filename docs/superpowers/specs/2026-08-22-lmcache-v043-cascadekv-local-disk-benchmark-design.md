# LMCache 0.4.3 CascadeKV vs Local Disk 端到端测试设计

## 目标

在同一套 vLLM 0.18.0 端到端长提示词负载下，对比当前 LMCache
`0.4.3+xds` 的 CascadeKV 后端和原生 Local Disk 后端。首轮只验证测试链路、
正确命中后端并获得一组可比较数据，不纳入移植前 CascadeKV。

## 固定环境

- Conda 环境：`xds043`
- LMCache 源码：`/home/yangxc/workspace/LMCache-v043-xds`
- vLLM 源码：`/home/yangxc/workspace/vllm`
- 模型：`/mnt/adminserver-nfsrdma/Model/Qwen3-30B-A3B`
- GPU：4、5、6、7，tensor parallel size 为 4
- xDS 服务：`xds0`，使用已授权测试的 NUMA1 四块原始 NVMe

## 后端配置

两个后端统一使用 `chunk_size: 256`、`use_layerwise: true`、
`save_unfull_chunk: true` 和 `local_cpu: false`。

CascadeKV 使用外部 xDS 服务和服务分配的有界 CPU staging slots；测试前验证
单个 layer chunk 的实际物理大小不超过现有 6 MiB blob。Local Disk 使用 LMCache
原生 buffered file I/O，缓存目录为测试专用目录，保留 Linux page cache 的默认行为，
并在报告中明确标记为 buffered Local Disk。

## 工作负载

复用 `/home/yangxc/workspace/test` 的 `benchmark_prefix_caching.py` 方法：

- 固定随机种子 0
- 5 个不同 prompt，每个重复 20 次
- 输入长度 26,530--30,000 tokens
- 输出 1 token
- `max_num_seqs=1`
- eager mode，最大模型长度 40,000
- vLLM automatic prefix caching 不启用，使重复请求必须经过 LMCache

第一次出现的 prompt 负责写入，后续重复 prompt 负责读取。模型初始化时间不计入
`llm.generate()` 的端到端耗时。

## 执行顺序与隔离

首版先各执行一轮：Local Disk 后执行 CascadeKV。每轮使用独立进程；Local Disk
开始前清空专用缓存目录，CascadeKV 开始前启动干净的 xDS 服务实例。若首轮无错误，
再单独扩展为交错顺序的三轮稳定性测试，不把额外轮次混入首版结果。

## 验收与输出

每个后端必须同时满足：vLLM 请求成功、LMCache 日志显示完成写入和缓存读取、重复
请求存在非零缓存命中、没有 short I/O、越界、blob 尺寸或 xDS completion 错误。

结果写入 `/home/yangxc/workspace/test/results/` 的时间戳目录，保存：

- 完整 vLLM/LMCache 日志
- xDS 服务日志（CascadeKV）
- 实际使用的两份 LMCache 配置
- 总耗时、请求吞吐、每 prompt 平均耗时和缓存命中证据
- 环境版本、Git commit、GPU/SSD 映射及异常说明

首轮数字只作为功能与初步性能数据；只有完成后续交错三轮后才报告稳定性能结论。
