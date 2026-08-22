# LMCache 0.4.3 CascadeKV vs Local Disk Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run one reproducible 40K-token vLLM end-to-end comparison between LMCache 0.4.3 CascadeKV and buffered Local Disk on the same four SSDs.

**Architecture:** A repository-owned benchmark launcher supplies two fixed LMCache configurations to the installed editable vLLM/LMCache sources. A second launcher reuses the existing guarded SSD switcher to run ext4/mdadm Local Disk first, destroy that filesystem, bind the same devices to `uio_pci_generic`, start xDS, and run CascadeKV. All commands and evidence are copied into one timestamped result directory.

**Tech Stack:** Bash, YAML, Conda `xds043`, vLLM 0.18.0, LMCache 0.4.3+xds, xDS service, mdadm RAID0, ext4.

## Global Constraints

- Use `/home/yangxc/miniconda3/envs/xds043/bin/python`.
- Use editable vLLM source `/home/yangxc/workspace/vllm-v0180` and LMCache source `/home/yangxc/workspace/LMCache-v043-xds`.
- Use `chunk_size: 256`, `use_layerwise: true`, and do not enable vLLM automatic prefix caching.
- Use only GPUs 4, 5, 6, and 7 with tensor parallel size 4.
- Use only SSD BDFs `0000:aa:00.0`, `0000:d3:00.0`, `0000:ec:00.0`, and `0000:ed:00.0`.
- Run exactly one group per backend: 5 prompts, 20 repeats, exactly 40,000 input tokens, 1 output token, seed 0.
- Run Local Disk before CascadeKV and leave the SSDs in xDS state.

---

### Task 1: Fixed backend configurations and backend launcher

**Files:**
- Create: `benchmarks/xds_e2e/disk_40k.yaml`
- Create: `benchmarks/xds_e2e/cascadekv_40k.yaml`
- Create: `benchmarks/xds_e2e/run_backend_40k.sh`
- Create: `benchmarks/xds_e2e/test_run_backend_40k.sh`

**Interfaces:**
- Consumes: environment variables `BACKEND`, `BENCHMARK_LOG`, and optional `BENCHMARK_DRY_RUN`.
- Produces: one vLLM benchmark log and exit status; dry-run prints the fully escaped command.

- [ ] **Step 1: Write the launcher contract test**

Create a shell test that invokes `run_backend_40k.sh` with `BENCHMARK_DRY_RUN=1` for `disk` and `cascadekv`, then asserts both commands contain the exact Python/source/model/TP/workload arguments; assert the selected YAML differs and an unknown backend exits 2.

```bash
disk=$(BACKEND=disk BENCHMARK_LOG=/tmp/disk.log BENCHMARK_DRY_RUN=1 bash "$RUNNER")
cascade=$(BACKEND=cascadekv BENCHMARK_LOG=/tmp/cascade.log BENCHMARK_DRY_RUN=1 bash "$RUNNER")
[[ "$disk" == *'disk_40k.yaml'* ]]
[[ "$cascade" == *'cascadekv_40k.yaml'* ]]
for output in "$disk" "$cascade"; do
  [[ "$output" == *'--input-length-range 40000:40000'* ]]
  [[ "$output" == *'--num-prompts 5'* ]]
  [[ "$output" == *'--repeat-count 20'* ]]
  [[ "$output" == *'--tensor-parallel-size 4'* ]]
  [[ "$output" != *'--enable-prefix-caching'* ]]
done
```

- [ ] **Step 2: Run the contract test and verify it fails**

Run: `bash benchmarks/xds_e2e/test_run_backend_40k.sh`

Expected: FAIL because the launcher and configurations do not exist.

- [ ] **Step 3: Add the two fixed LMCache configurations**

Use this disk configuration:

```yaml
chunk_size: 256
local_cpu: false
max_local_cpu_size: 1
local_disk: "file:///mnt/lmcache-raid0/disk-cache"
max_local_disk_size: 1000
use_layerwise: true
save_unfull_chunk: true
```

Use this CascadeKV configuration:

```yaml
chunk_size: 256
local_cpu: false
max_local_cpu_size: 0
use_layerwise: true
save_unfull_chunk: true
xds_service_name: xds0
xds_backend: false
cascadekv_backend: true
xds_require_direct: false
cascadekv_slot_count: 8
```

The disk-only 1 GiB LocalCPU allocation is the allocator required by LMCache's LocalDisk backend; `local_cpu: false` prevents it from becoming a cache tier.

- [ ] **Step 4: Implement the backend launcher**

Build an array command around `/home/yangxc/workspace/vllm-v0180/benchmarks/benchmark_prefix_caching.py` with these exact engine arguments:

```bash
--model /mnt/adminserver-nfsrdma/Model/Qwen3-30B-A3B
--enforce-eager
--max-model-len 40960
--trust-remote-code
--cpu-offload-gb 0
--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
--prefix-len 0
--tensor-parallel-size 4
--input-length-range 40000:40000
--num-prompts 5
--repeat-count 20
--output-len 1
--max-num-seqs 1
--no-enable-prefix-caching
--gpu-memory-utilization 0.85
--seed 0
```

Set `CUDA_VISIBLE_DEVICES=4,5,6,7`, `PYTHONHASHSEED=0`, `LMCACHE_USE_EXPERIMENTAL=True`, `LMCACHE_CONFIG_FILE` to the selected YAML, and `PYTHONPATH=/home/yangxc/workspace/xDS/sdk/src:/home/yangxc/workspace/LMCache-v043-xds:/home/yangxc/workspace/vllm-v0180`.

- [ ] **Step 5: Run the launcher contract test**

Run: `bash benchmarks/xds_e2e/test_run_backend_40k.sh`

Expected: `LMCACHE_V043_BACKEND_RUNNER_TESTS_OK`.

- [ ] **Step 6: Commit Task 1**

```bash
git add benchmarks/xds_e2e
git commit -m "bench: add 40k CascadeKV and disk launchers"
```

### Task 2: Same-SSD orchestration and safety checks

**Files:**
- Create: `benchmarks/xds_e2e/run_same_ssd_compare_40k.sh`
- Create: `benchmarks/xds_e2e/test_run_same_ssd_compare_40k.sh`

**Interfaces:**
- Consumes: existing `/home/yangxc/workspace/test/scripts/switch_lmcache_test_ssds.sh` and xDS service configuration `/home/yangxc/workspace/xDS/configs/setup_qwen3_30b_tp4_numa1.json`.
- Produces: timestamped result directory under `/home/yangxc/workspace/test/results/`; returns SSDs to xDS state on both success and failure.

- [ ] **Step 1: Write a dry-run orchestration test**

The test runs the orchestrator with `BENCHMARK_DRY_RUN=1` and asserts the ordered output contains:

```text
switch_lmcache_test_ssds.sh filesystem --yes --confirm ERASE-aa-d3-ec-ed
BACKEND=disk
switch_lmcache_test_ssds.sh xds --yes --confirm ERASE-aa-d3-ec-ed
xds_service --config /home/yangxc/workspace/xDS/configs/setup_qwen3_30b_tp4_numa1.json --service-name xds0
BACKEND=cascadekv
```

Also assert that the script names only the four approved BDFs and refuses to continue if GPU 4--7 are busy or an xDS service/IPC namespace already exists.

- [ ] **Step 2: Run the dry-run test and verify it fails**

Run: `bash benchmarks/xds_e2e/test_run_same_ssd_compare_40k.sh`

Expected: FAIL because the orchestrator does not exist.

- [ ] **Step 3: Implement preflight and cleanup**

Before destructive switching, capture `git rev-parse HEAD`, package versions, `nvidia-smi`, `lspci -nnk` for the four approved BDFs, switcher status, service config, and both YAML files. Require the exact switch confirmation token. Reject unexpected drivers, mounted/busy RAID filesystems, busy GPUs, or stale xDS service/IPC state.

Install an EXIT/INT/TERM trap that stops xDS, removes only `/tmp/xds_xds0.sock` and `/tmp/xds_xds0.shm` after the service has stopped, and calls the existing switcher `xds` transition if the devices are not already in xDS state.

- [ ] **Step 4: Implement the ordered run**

Perform these exact phases:

```text
1. switch approved SSDs to filesystem mode
2. create/chown /mnt/lmcache-raid0/disk-cache
3. run BACKEND=disk once
4. sync and switch approved SSDs to xDS mode
5. remount /dev/hugepages read-write if required
6. start xds_service and wait up to 120 seconds for /tmp/xds_xds0.sock
7. run BACKEND=cascadekv once
8. stop xDS and restore the original hugepage mount flags
```

- [ ] **Step 5: Run dry-run and static checks**

Run:

```bash
bash -n benchmarks/xds_e2e/run_same_ssd_compare_40k.sh
bash benchmarks/xds_e2e/test_run_same_ssd_compare_40k.sh
```

Expected: syntax check succeeds and test prints `LMCACHE_V043_SAME_SSD_ORCHESTRATOR_TESTS_OK`.

- [ ] **Step 6: Commit Task 2**

```bash
git add benchmarks/xds_e2e
git commit -m "bench: orchestrate same-SSD CascadeKV comparison"
```

### Task 3: Preflight, execute, and report one group per backend

**Files:**
- Create at runtime: `/home/yangxc/workspace/test/results/lmcache-v043-cascadekv-disk-40k-<UTC>/manifest.txt`
- Create at runtime: `/home/yangxc/workspace/test/results/lmcache-v043-cascadekv-disk-40k-<UTC>/disk.log`
- Create at runtime: `/home/yangxc/workspace/test/results/lmcache-v043-cascadekv-disk-40k-<UTC>/cascadekv.log`
- Create at runtime: `/home/yangxc/workspace/test/results/lmcache-v043-cascadekv-disk-40k-<UTC>/xds-service.log`
- Create at runtime: `/home/yangxc/workspace/test/results/lmcache-v043-cascadekv-disk-40k-<UTC>/summary.md`

**Interfaces:**
- Consumes: Task 2 orchestrator and authorized root privileges for the four SSDs, mdadm, mounts, driver binding, hugepages, and xDS service.
- Produces: one evidence-backed end-to-end measurement per backend and final xDS device state.

- [ ] **Step 1: Run non-destructive preflight**

Verify the env reports `vllm 0.18.0+cu130`, `lmcache 0.4.3+xds`, `torch 2.10.0+cu130`; all four SSDs resolve to the expected BDFs; GPU 4--7 are idle; and the xDS binary/config exist.

- [ ] **Step 2: Run the complete comparison**

Run the orchestrator once with the explicit destructive confirmation already granted for the four test SSDs. Stream logs to the timestamped result directory and keep the user updated at the disk completion and SSD transition checkpoints.

- [ ] **Step 3: Validate backend evidence**

For both logs, require a successful `cost time` line and no traceback, CUDA OOM, short I/O, blob-size, or xDS completion error. Require the disk log to contain `LocalDiskBackend`, the Cascade log to contain `CascadeKVBackend`, and LMCache metrics/logs to show non-zero retrieved tokens after the first occurrence of each prompt.

- [ ] **Step 4: Calculate the one-group comparison**

Write `summary.md` with exact total seconds, 100-request throughput, mean seconds/request, relative speedup, prompt parameters, backend paths, page-cache caveat, and a clear statement that a single group is preliminary rather than a stability result.

- [ ] **Step 5: Verify final hardware state and repository state**

Run the SSD switcher status and require `"state":"xds"`; confirm no xDS service or stale IPC object remains; run `git status --short`; and link the result directory and summary in the handoff.

---

Plan self-review: every design requirement maps to a task; no unapproved SSD appears; 40K input plus one output fits `max_model_len=40960`; Local Disk has the allocator required by LMCache 0.4.3 without making LocalCPU a cache tier; failure cleanup returns the four devices to xDS state.
