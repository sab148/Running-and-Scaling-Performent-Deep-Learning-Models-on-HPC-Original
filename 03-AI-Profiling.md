---
author: Javad Kasravi
title: AI Profiling
date: Sep. 10, 2026
---
<style>
.reveal h3 {
  font-size: 40px;
}

.reveal ul li {
  margin-bottom: 20px;  /* adjust as needed */
}
</style>


## Who’s Stealing My Speed?



![](images/profiling/profiling_questions.png){style="width: 90%; max-width: 1200px; margin-left: 0; margin-right: auto;"}


## Agenda

- Node Communications
- NVIDIA AI Profiling tools
- Single GPU training
- Distributed Data Parallel (DDP)
    - Single node training
    - Multi node training
- DDP scaling

---


## Single Node Communications

---

###  Base System


![](images/profiling/base_system.png)

---

###  Naive communication

**Data Path:** 

<div style="font-weight: bold; background-color: #ffcccc;">
```bash
GPU 0  →  PCI Bus  →  System Memory →  PCI Bus →  GPU 1
```
</div>

![](images/profiling/host_staging_copy.png)

---

###  PCI Bus Peer-to-Peer (P2P) communication

**Data Path:** 


<div style="font-weight: bold; background-color: #ffcccc;">
```bash
GPU 0  →  PCI Bus →  GPU 1
```
</div>

![](images/profiling/p2p-memory-access.png)


---

### GPUDirect P2P communication (NVLink)

**Data Path** 

<div style="font-weight: bold; background-color: #ffcccc;">
```bash
GPU 0  →  NVLink  →  GPU 2
```
</div>

![](images/profiling/NVswitch_01.png)


---

### Throughput Comparison

| Communication Type | Throughput   |
|---------------------|------------------------|
| Naive communication               | ~16 GB/s per GPU             <span style="font-size:2em">🐢</span> |
| PCIe Bus P2P communication       | ~32 GB/s per GPU             <span style="font-size:2em">🚗</span> |
| GPUDirect P2P communication             | 300 GB/s total per GPU          <span style="font-size:2em">🏎️</span> |

---

## Multinode Commumications

---

### GPUDirect <span style="color: red;">Without</span> RDMA Communication

![](images/profiling/multi_node_without_rdma.png)

---

### GPUDirect <span style="color: blue;">With</span> RDMA Communication

![](images/profiling/multi_node_with_rdma.png)

---

### Throughput Comparison

| Communication Type | Throughput   |
|---------------------|------------------------|
| GPUDirect Without RDMA              | <50 GB/s              <span style="font-size:2em">🐢</span> |
| GPUDirect With RDMA             | ~ 50 GB/s per node (2 HDR InfiniBand)          <span style="font-size:2em">🏎️</span> |

---


## AI Profiling?


---

### NVIDIA profiling Tools

![](images/profiling/nsight_flowchart_systemcomputegraphics.png)


---

### Nsight Systems GUI
Go to the following link and download the **Nsight System 2025.3.1**:

<div style="font-weight: bold; background-color: #ffcccc;">
```bash
https://developer.nvidia.com/tools-downloads
```
</div>

Download a sample **Nsight Systems** trace:

<div style="font-weight: bold; background-color: #ffcccc; padding: 12px 16px; border-radius: 8px;">
<a href="https://drive.google.com/file/d/1fhYkZ4LkyqbB9-_PKYXcU6dDkpIpwtPB/view" target="_blank" style="color:#8a0000; text-decoration: underline;">Download Nsight Systems trace (Google Drive)</a>
</div>

---


### Run Profiling

<div style="font-weight: bold; background-color: #ffcccc;">
```bash
    srun env -u CUDA_VISIBLE_DEVICES bash -c 'torchrun \
       --nproc-per-node=gpu \
       --nnodes="$SLURM_JOB_NUM_NODES" \
       --rdzv-id="$SLURM_JOB_ID" \
       --rdzv-endpoint="$MASTER_ADDR":"$MASTER_PORT" \
       --rdzv-backend=c10d \
       --rdzv-conf=is_host="$(if ((SLURM_NODEID)); then echo 0; else echo 1; fi)" \
       --local-addr="$(if ((SLURM_NODEID)); then echo $MASTER_ADDR; else hostname; fi)" \
       --no-python ./run_profile.sh train/ddp_training.py --profile'
```
</div>


<div style="font-weight: bold; background-color: #ffcccc;">
```bash
nsys profile \
    --duration=30 \
    --delay=200 \
    --gpu-metrics-device=all \
    --nic-metrics=true \
    --stop-on-exit=false \
    --trace=nvtx,cuda,osrt \
    --cuda-memory-usage=true \
    --force-overwrite=true \
    --python-functions-trace=profiler/config/profiling.json \
    --output=nsys_logs/nsys_logs_rank_${RANK} \
    --python-backtrace=cuda \
    --cudabacktrace=all \
    python -u "$SCRIPT_NAME" "$@"
```
</div>


---


### Run Profiling

Inside of `profiler/config/profiling.json`

<div style="font-weight: bold; background-color: #ffcccc;">
```bash
    {
        "domain": "PyTorch",
        "color": "E8B795",
        "module": "torch.amp",
        "functions": [
            "GradScaler.scale",
            "GradScaler.unscale",
            "GradScaler.unscale_",
            "GradScaler.step",
            "GradScaler.update"
        ]
    },
    ...
```
</div>

We also trace the NVTX trace

<div style="font-weight: bold; background-color: #ffcccc;">
```bash
for step in range(num_steps):
    with ExecutionTimer("data_loading (to Sys. mem)", profile=True) as t:
        src, tgt = next(train_iter)
    with ExecutionTimer("data_movement (to GPU mem)", profile=True) as t:
        src, tgt = src.to(device, non_blocking=False), tgt.to(device, non_blocking=False)
    with ExecutionTimer("forward_step", profile=True) as t:
        output = model(src)
    ...
```
</div>



---

### Nsight Systems (Deep Learning App.)

![](images/profiling/nsys.svg){style="width: 200%; max-width: 1200px; margin-left: -100px; margin-right: auto;"}


---

### what information is inside?

<img src="images/profiling/nsight_timeline_parts.png" alt="Nsight Timeline parts" style="width: 96%; max-width: 1180px; background: #fff;"/>

<div style="font-size: 0.34em; margin-top: 0.1em;">
Shared time axis: <b>GPU metrics</b> · <b>NVLink</b> (spikes at each allreduce) · <b>PCIe</b> (flat) · <b>NCCL</b> · <b>NIC</b> row
</div>

---

### What information each training label gives you

<div style="font-size: 0.45em; text-align: left; max-width: 1150px; margin: 0 auto;">

| NVTX label (CPU) | Information on the CPU row | Information to read on CUDA HW |
|---|---|---|
| `:data_loading` | Time to load the batch into system memory | Usually little/no GPU activity |
| `:data_movement` | Time the host spends in `tensor.to(device)` (can include waiting) | Length of H2D **memcpy** (often much shorter than the CPU bar) |
| `:forward_step` | Time Python stays in the forward phase | Forward **kernels** (GPU compute; may be longer than the CPU bar) |
| `:backward_step` | Autograd on the host (graph walk + launches) | Gradient **kernels** on the GPU |
| `:optimizer_step` | Time in the optimizer step on the host | Optimizer **kernels** on the GPU |

</div>

---

### From the Timeline to an optimization choice

<img src="images/profiling/optimization_decision_tree.png" alt="Optimization decision tree" style="width: 100%; max-width: 1280px; max-height: 720px; object-fit: contain; background: #fff; margin: 0;"/>

---

## Focus: DATA LOADING

---

### Focus: DATA LOADING — Single_GPU laboratory

<img src="images/profiling/data_loading_focus.png" alt="DATA LOADING focus and Single_GPU experiment ladder" style="width: 100%; max-width: 1800px; max-height: 920px !important; height: auto; object-fit: contain; background: #fff; margin: 0;"/>

---

### How DataLoader works with 0 workers

<iframe src="images/profiling/dataloader_zero_workers_anim.html" title="Animated DataLoader with 0 workers" style="width:100%;max-width:1600px;height:760px;border:0;border-radius:12px;background:#fff;display:block;margin:0 auto;"></iframe>

---

### Single GPU without dataloader Worker

<div style="max-width:1100px;margin:0 auto;text-align:left;font-size:0.48em;">
<div style="background:#ffcccc;border-radius:10px;padding:10px 14px;margin-bottom:12px;">
<div style="font-size:0.85em;color:#8a0000;font-weight:700;margin-bottom:4px;">Open this trace</div>
<code style="font-size:0.92em;">Single_GPU/Report_00_zero_woker_unpined_non_blocking_False/nsys_logs/540_WVB_725_nsys_logs_rank.nsys-rep</code>
</div>
<div style="display:flex;gap:12px;margin-bottom:12px;">
<div style="flex:1;background:#fff8ea;border:2px solid #f0d965;border-radius:10px;padding:10px 14px;">
<div style="font-weight:700;color:#6d4c00;margin-bottom:4px;">Zoom</div>
Use <b>+</b> / <b>&minus;</b> and pick a steady middle iteration (skip startup).
</div>
<div style="flex:1;background:#f0d965;border-radius:10px;padding:10px 14px;">
<div style="font-weight:700;color:#6d4c00;margin-bottom:4px;">Answer the form</div>
<a href="https://forms.gle/LMyJAzm3Q5tfWJqc7" target="_blank" style="color:#023d6b;font-weight:700;">Single GPU without dataloader Worker</a>
</div>
</div>
<div style="padding:12px 16px;background:#e8f5e9;border-left:5px solid #4caf50;border-radius:0 10px 10px 0;">
<div style="font-weight:700;color:#1b5e20;margin-bottom:8px;">Guided exploration</div>
<ul style="margin:0 0 0 1.1em;padding:0;color:#2e7d32;">
<li>Find the Python process that has a <b>CUDA HW</b> row.</li>
<li>NVTX labels (<code>:data_loading</code>, ...) live on the <b>CPU</b> timeline &mdash; compare the same window on CUDA HW.</li>
<li>Measure on the CPU row; use CUDA HW to see if the GPU is busy, copying, or idle.</li>
<li>Then answer the form from what you observe.</li>
</ul>
</div>
</div>

---

## Single GPU Multiworkers

---

### How DataLoader workers work

<iframe src="images/profiling/dataloader_workers_anim.html" title="Animated DataLoader with 2 workers" style="width:100%;max-width:1600px;height:760px;border:0;border-radius:12px;background:#fff;display:block;margin:0 auto;"></iframe>

---

### Enable workers in code

<div style="font-weight: bold; background-color: #ffcccc;">
```python
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    num_workers=4,
    pin_memory=False,
)
```
</div>

<div style="font-weight: bold; background-color: #ffcccc;">
```bash
sbatch --disable-dcgm single_gpu_training_profiling.sbatch --profile
```
</div>

---

### Single GPU Multiworkers — explore the trace

<div style="max-width:1100px;margin:0 auto;text-align:left;font-size:0.48em;">
<div style="background:#ffcccc;border-radius:10px;padding:10px 14px;margin-bottom:12px;">
<div style="font-size:0.85em;color:#8a0000;font-weight:700;margin-bottom:4px;">Open this trace</div>
<code style="font-size:0.92em;">Single_GPU/Report_01_multi_woker_unpined_non_blocking_False/nsys_logs/647_ATA_566_nsys_logs_rank.nsys-rep</code>
</div>
<div style="display:flex;gap:12px;margin-bottom:12px;">
<div style="flex:1;background:#fff8ea;border:2px solid #f0d965;border-radius:10px;padding:10px 14px;">
<div style="font-weight:700;color:#6d4c00;margin-bottom:4px;">Zoom</div>
Use <b>+</b> / <b>&minus;</b> and pick a steady middle iteration.
</div>
<div style="flex:1;background:#f0d965;border-radius:10px;padding:10px 14px;">
<div style="font-weight:700;color:#6d4c00;margin-bottom:4px;">Answer the form</div>
<a href="https://forms.gle/oxNaBnjP8cA9kPJw5" target="_blank" style="color:#023d6b;font-weight:700;">Single GPU Multiworkers</a>
</div>
</div>
<div style="padding:12px 16px;background:#e8f5e9;border-left:5px solid #4caf50;border-radius:0 10px 10px 0;">
<div style="font-weight:700;color:#1b5e20;margin-bottom:8px;">Guided exploration</div>
<ul style="margin:0 0 0 1.1em;padding:0;color:#2e7d32;">
<li>Find <code>pt_data_worker</code> rows — how many are there? Do they have CUDA HW?</li>
<li>On the <b>main</b> process, measure <code>data_loading</code> and compare with Report 00.</li>
<li>Which NVTX stage is still longest? Did the full iteration get much faster?</li>
<li>Under <code>data_movement</code>, is CUDA HW busy with kernels or only a short H2D copy?</li>
</ul>
</div>
</div>

---

### Multiworkers vs zero workers

<img src="images/profiling/multiworker_vs_zeroworkers.png" alt="Multiworkers vs zero workers" style="width:70%;max-width:1200px;max-height:860px !important;height:auto;object-fit:contain;margin:0 auto;display:block;"/>


<!--
---

### Single GPU Multiworkers (Asyn. Transfer)

![](images/profiling/Asynchronous_Transfer.png){style="width: 45%; max-width: 1200px; margin-left: 0; margin-right: auto;"}

-->
---

## DMA (Direct Memory Access)

<iframe src="images/profiling/dma_anim.html" title="CPU transfer vs DMA: stepped comparison" style="width:100%;max-width:1400px;height:620px;border:0;border-radius:12px;background:#fff;display:block;margin:0 auto;"></iframe>

---

## 

![](images/profiling/cuda_q.png){style="width: 60%; max-width: 1200px; margin-left: 0; margin-right: auto;"}

How does `cudaMemcpy` copy data from host to device?

---



## cudaMemcpy


<div style="padding: 15px 15px 15px 25px; background-color: #e8f5e9; border-left: 5px solid #4caf50; margin: 10px 0; font-size: 0.7em;">
  <ul style="margin: 5px 0 0 20px; padding-left: 0;">
    <li class="fragment"><strong>What happens during <code>cudaMemcpy</code>:</strong></li>

    <li class="fragment" style="margin-left: 20px;"><strong>Host to Device:</strong></li>
    <li class="fragment" style="margin-left: 40px;">The CPU copies data to a pinned memory buffer.</li>
    <li class="fragment" style="margin-left: 40px;">DMA transfers the data from the pinned memory buffer to GPU memory.</li>

    <li class="fragment" style="margin-left: 20px;"><strong>Device to Host:</strong></li>
    <li class="fragment" style="margin-left: 40px;">DMA transfers data from GPU memory to a pinned memory buffer.</li>
    <li class="fragment" style="margin-left: 40px;">The CPU copies data from the pinned memory buffer.</li>

    <li class="fragment" style="margin-top: 10px; color: #c62828; font-weight: bold;">Disadvantage:</li>
    <li class="fragment" style="margin-left: 20px; color: #c62828; font-weight: bold;">Every <code>cudaMemcpy</code> actually involves two copies.</li>
  </ul>
</div>


## Memory pinning

<iframe src="images/profiling/memory_pinning_anim.html" title="Pageable vs pinned host memory" style="width:100%;max-width:1400px;height:620px;border:0;border-radius:12px;background:#fff;display:block;margin:0 auto;"></iframe>

---


### Single GPU (Asyn. Transfer)

<div style="font-weight: bold; background-color: #ffcccc;">
```python
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    num_workers=4,
    pin_memory=True,
)
```
</div>

<div style="font-weight: bold; background-color: #ffcccc;">
```python
src, tgt = src.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
```
</div>

<div style="font-weight: bold; background-color: #ffcccc;">
```bash
sbatch --disable-dcgm single_gpu_training_profiling.sbatch --profile
```
</div>

---

### Single GPU Multiworkers (Asyn. Transfer)

<div style="max-width:1100px;margin:0 auto;text-align:left;font-size:0.48em;">
<div style="background:#ffcccc;border-radius:10px;padding:10px 14px;margin-bottom:12px;">
<div style="font-size:0.85em;color:#8a0000;font-weight:700;margin-bottom:4px;">Open this trace</div>
<code style="font-size:0.92em;">Single_GPU/Report_02_multiwoker_pined_non_blocking_True/nsys_logs/107_RKN_402_nsys_logs_rank.nsys-rep</code>
</div>
<div style="display:flex;gap:12px;margin-bottom:12px;">
<div style="flex:1;background:#fff8ea;border:2px solid #f0d965;border-radius:10px;padding:10px 14px;">
<div style="font-weight:700;color:#6d4c00;margin-bottom:4px;">Zoom</div>
Use <b>+</b> / <b>&minus;</b> and pick a steady middle iteration.
</div>
<div style="flex:1;background:#f0d965;border-radius:10px;padding:10px 14px;">
<div style="font-weight:700;color:#6d4c00;margin-bottom:4px;">Answer the form</div>
<a href="https://forms.gle/Fi1EVARNaVRZcpsR8" target="_blank" style="color:#023d6b;font-weight:700;">Single GPU Multiworkers (Asyn. Transfer)</a>
</div>
</div>
<div style="padding:12px 16px;background:#e8f5e9;border-left:5px solid #4caf50;border-radius:0 10px 10px 0;">
<div style="font-weight:700;color:#1b5e20;margin-bottom:8px;">Guided exploration</div>
<ul style="margin:0 0 0 1.1em;padding:0;color:#2e7d32;">
<li>Check one iteration of training (PyTorch / NVTX stages).</li>
<li>Which stage is longest: data loading, data movement, forward, or backward?</li>
<li>Is there any CUDA synchronization in the trace?</li>
</ul>
</div>
</div>

<!--

---

### Single GPU Multiworkers (Asyn. Transfer)
![](images/profiling/data_transfer_GPU_mem.png){style="width: 70%; max-width: 1200px; margin-left: 0; margin-right: auto;"}

---
-->

## DDP 

---

### DDP — single node (4 GPUs)

Reproduce this trace:

<div style="font-weight: bold; background-color: #ffcccc;">
```bash
sbatch --disable-dcgm --nodes 1 ddp_training.sbatch --profile
```
</div>

<div style="max-width:1100px;margin:0 auto;text-align:left;font-size:0.48em;">
<div style="background:#ffcccc;border-radius:10px;padding:10px 14px;margin-bottom:12px;">
<div style="font-size:0.85em;color:#8a0000;font-weight:700;margin-bottom:4px;">Open this trace</div>
<code style="font-size:0.92em;">Multi_GPUs/DDP/Single_node/Report_03_DDP_one_node/nsys_logs/nsys_logs_rank_2.nsys-rep</code>
</div>
<div style="display:flex;gap:12px;margin-bottom:12px;">
<div style="flex:1;background:#fff8ea;border:2px solid #f0d965;border-radius:10px;padding:10px 14px;">
<div style="font-weight:700;color:#6d4c00;margin-bottom:4px;">Zoom</div>
Use <b>+</b> / <b>&minus;</b> and pick a steady middle iteration (skip startup / NCCL init).
</div>
<div style="flex:1;background:#f0d965;border-radius:10px;padding:10px 14px;">
<div style="font-weight:700;color:#6d4c00;margin-bottom:4px;">Answer the form</div>
<a href="https://forms.gle/z6VoTiQ1gZjwZTRDA" target="_blank" style="color:#023d6b;font-weight:700;">DDP (4 GPUs)</a>
</div>
</div>
<div style="padding:12px 16px;background:#e8f5e9;border-left:5px solid #4caf50;border-radius:0 10px 10px 0;">
<div style="font-weight:700;color:#1b5e20;margin-bottom:8px;">Guided exploration</div>
<ul style="margin:0 0 0 1.1em;padding:0;color:#2e7d32;">
<li>On <b>GPU metrics</b>, compare <b>NVLink</b> vs <b>PCIe</b> during training — which spikes with collectives?</li>
<li>On <b>CUDA HW</b>, find NCCL kernels (e.g. <code>ncclDevKernel_AllReduce_...</code>) / the NCCL row.</li>
<li>Count how many times <b>AllReduce</b> runs in this report (NCCL / CUDA HW).</li>
<li>In a steady window: do non-NCCL <b>compute kernels</b> run at the same time as AllReduce, or only before/after?</li>
</ul>
</div>
</div>



---


### DDP — multi-node with GPUDirect RDMA (8 GPUs)

Reproduce this trace:

<div style="font-weight: bold; background-color: #ffcccc;">
```bash
sbatch --disable-dcgm --nodes 2 ddp_training.sbatch --profile
```
</div>

<div style="max-width:1100px;margin:0 auto;text-align:left;font-size:0.48em;">
<div style="background:#ffcccc;border-radius:10px;padding:10px 14px;margin-bottom:12px;">
<div style="font-size:0.85em;color:#8a0000;font-weight:700;margin-bottom:4px;">Open this trace</div>
<code style="font-size:0.92em;">Multi_GPUs/DDP/Multi_nodes/Report_03_DDP_multi_nodes/GPU_Direct_RDMA_enable/nsys_logs/nsys_logs_rank_1.nsys-rep</code>
</div>
<div style="display:flex;gap:12px;margin-bottom:12px;">
<div style="flex:1;background:#fff8ea;border:2px solid #f0d965;border-radius:10px;padding:10px 14px;">
<div style="font-weight:700;color:#6d4c00;margin-bottom:4px;">Zoom</div>
Use <b>+</b> / <b>&minus;</b> and pick a steady middle iteration (skip startup / NCCL init).
</div>
<div style="flex:1;background:#f0d965;border-radius:10px;padding:10px 14px;">
<div style="font-weight:700;color:#6d4c00;margin-bottom:4px;">Answer the form</div>
<a href="https://forms.gle/KvvPEqvEnVLhVNrL8" target="_blank" style="color:#023d6b;font-weight:700;">DDP (8 GPUs)</a>
</div>
</div>
<div style="padding:12px 16px;background:#e8f5e9;border-left:5px solid #4caf50;border-radius:0 10px 10px 0;">
<div style="font-weight:700;color:#1b5e20;margin-bottom:8px;">Guided exploration</div>
<ul style="margin:0 0 0 1.1em;padding:0;color:#2e7d32;">
<li>On <b>GPU metrics</b>, compare <b>NVLink</b> vs <b>PCIe</b> — which path carries <i>intra-node</i> collectives?</li>
<li>Also glance at <b>NIC / InfiniBand</b> metrics — multi-node traffic should appear here (GPUDirect RDMA).</li>
<li>On <b>CUDA HW</b>, find <code>ncclDevKernel_AllReduce_...</code>; note how long AllReduce bars are vs the single-node case.</li>
<li>In a steady window: do non-NCCL <b>compute kernels</b> overlap AllReduce, or only run before/after? Use the same overlap options as before.</li>
</ul>
</div>
</div>



---


### DDP — multi-node without GPUDirect RDMA (`NCCL_P2P_LEVEL=LOC`)

Reproduce this trace:

<div style="font-weight: bold; background-color: #ffcccc;">
```bash
NCCL_P2P_LEVEL=LOC sbatch --disable-dcgm --nodes 2 ddp_training.sbatch --profile
```
</div>

<div style="max-width:1100px;margin:0 auto;text-align:left;font-size:0.48em;">
<div style="background:#ffcccc;border-radius:10px;padding:10px 14px;margin-bottom:12px;">
<div style="font-size:0.85em;color:#8a0000;font-weight:700;margin-bottom:4px;">Open this trace</div>
<code style="font-size:0.92em;">Multi_GPUs/DDP/Multi_nodes/Report_03_DDP_multi_nodes/GPU_Direct_RDMA_disable/nsys_logs/nsys_logs_rank_2.nsys-rep</code>
</div>
<div style="display:flex;gap:12px;margin-bottom:12px;">
<div style="flex:1;background:#fff8ea;border:2px solid #f0d965;border-radius:10px;padding:10px 14px;">
<div style="font-weight:700;color:#6d4c00;margin-bottom:4px;">Zoom</div>
Use <b>+</b> / <b>&minus;</b> and pick a steady middle iteration (skip startup / NCCL init).
</div>
<div style="flex:1;background:#f0d965;border-radius:10px;padding:10px 14px;">
<div style="font-weight:700;color:#6d4c00;margin-bottom:4px;">Answer the form</div>
<a href="https://forms.gle/XXF5EHMLQkRaM88k8" target="_blank" style="color:#023d6b;font-weight:700;">DDP (8 GPUs without GPU_Direct_RDMA)</a>
</div>
</div>
<div style="padding:12px 16px;background:#e8f5e9;border-left:5px solid #4caf50;border-radius:0 10px 10px 0;">
<div style="font-weight:700;color:#1b5e20;margin-bottom:8px;">Guided exploration</div>
<ul style="margin:0 0 0 1.1em;padding:0;color:#2e7d32;">
<li><code>NCCL_P2P_LEVEL=LOC</code> restricts GPU P2P to PCIe locality — compare <b>NVLink</b> vs <b>PCIe</b> on GPU metrics vs the previous RDMA-enable slide.</li>
<li>On <b>CUDA HW</b>, find <code>ncclDevKernel_AllReduce_...</code>; are AllReduce bars longer than with GPUDirect RDMA?</li>
<li>In a steady window: do non-NCCL <b>compute kernels</b> overlap AllReduce? Use the same overlap options as before.</li>
</ul>
</div>
</div>



---



### DDP — multi-node without InfiniBand (`NCCL_IB_DISABLE=1`)

Reproduce this trace:

<div style="font-weight: bold; background-color: #ffcccc;">
```bash
NCCL_IB_DISABLE=1 sbatch --disable-dcgm --nodes 2 ddp_training.sbatch --profile
```
</div>

<div style="max-width:1100px;margin:0 auto;text-align:left;font-size:0.48em;">
<div style="background:#ffcccc;border-radius:10px;padding:10px 14px;margin-bottom:12px;">
<div style="font-size:0.85em;color:#8a0000;font-weight:700;margin-bottom:4px;">Open this trace</div>
<code style="font-size:0.92em;">Multi_GPUs/DDP/Multi_nodes/Report_03_DDP_multi_nodes/NOT_IB_USAGE/nsys_logs/nsys_logs_rank_0.nsys-rep</code>
</div>
<div style="display:flex;gap:12px;margin-bottom:12px;">
<div style="flex:1;background:#fff8ea;border:2px solid #f0d965;border-radius:10px;padding:10px 14px;">
<div style="font-weight:700;color:#6d4c00;margin-bottom:4px;">Zoom</div>
Use <b>+</b> / <b>&minus;</b> and pick a steady middle iteration (skip startup / NCCL init).
</div>
<div style="flex:1;background:#f0d965;border-radius:10px;padding:10px 14px;">
<div style="font-weight:700;color:#6d4c00;margin-bottom:4px;">Answer the form</div>
<a href="https://forms.gle/zn58hSUSLZZWPzLX8" target="_blank" style="color:#023d6b;font-weight:700;">DDP (8 GPUs without Infiniband usage)</a>
</div>
</div>
<div style="padding:12px 16px;background:#e8f5e9;border-left:5px solid #4caf50;border-radius:0 10px 10px 0;">
<div style="font-weight:700;color:#1b5e20;margin-bottom:8px;">Guided exploration</div>
<ul style="margin:0 0 0 1.1em;padding:0;color:#2e7d32;">
<li><code>NCCL_IB_DISABLE=1</code> forces NCCL off InfiniBand — check <b>NIC</b> metrics (Ethernet / sockets vs IB) vs the RDMA-enable slide.</li>
<li>On <b>CUDA HW</b>, find <code>ncclDevKernel_AllReduce_...</code>; how long are the bars vs previous multi-node runs?</li>
<li>Look at <b>SMs Active</b> / compute during AllReduce — is the GPU busy or mostly waiting on communication?</li>
<li>Use the same overlap options as before; then decide if this setup looks scalable.</li>
</ul>
</div>
</div>



---


### DDP scaling

![](images/profiling/DDP_scaling.png)

---

