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

### How to read an Nsight trace (before the exercise)

Open the `.nsys-rep` and stay on the **Timeline**.  
Below is what information each part of the Timeline contains.

---

### Timeline parts — what information is inside?

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
| `:calculating_loss` | Short host-side loss phase | Loss / softmax **kernels** (often async) |
| `:backward_step` | Autograd on the host (graph walk + launches) | Gradient **kernels** on the GPU |
| `:optimizer_step` | Time in the optimizer step on the host | Optimizer **kernels** on the GPU |

</div>

---

### From the Timeline to an optimization choice

<img src="images/profiling/optimization_decision_tree.png" alt="Optimization decision tree" style="width: 100%; max-width: 1280px; max-height: 720px; object-fit: contain; background: #fff; margin: 0;"/>

---

### Single GPU without dataloader Worker

Move the trace folder to your local machine by running:
<div style="font-weight: bold; background-color: #ffcccc; font-size: 0.8em; min-width: 1050px">
```bash
scp -r -4 <user>@jureca.fz-juelich.de:/p/project1/training2643/AI_profiling/Nsys_trace_update_Jan_2026 .
```
</div>

<div style="font-weight: bold; background-color: #ffcccc;; font-size: 0.8em; min-width: 1050px;">
```bash
File -> Open -> Single_GPU/Report_00_zero_woker_unpined_non_blocking_False/nsys_logs/540_WVB_725_nsys_logs_rank.nsys-rep
```
</div>

Use + and − keys to zoom in and out

Go to this link to answer the questions:

<div style="font-weight: bold; background-color: #f0d965; font-size: 0.6em; min-width: 800px; padding: 8px;">
  <a href="https://forms.gle/LMyJAzm3Q5tfWJqc7" target="_blank">
    Single GPU without dataloader Worker
  </a>
</div>

<div style="padding: 15px 15px 15px 25px; background-color: #e8f5e9; border-left: 5px solid #4caf50; margin: 10px 0; font-size: 0.40em;">
<div style="font-weight: bold; margin-bottom: 8px; margin-left: -850px;">📘 Guided exploration (then answer the form)</div>
<ol style="margin: 5px 0 0 20px;">
<li><b>Orient:</b> Find the Python process with a <b>CUDA HW</b> row. Expand it. Zoom (+) to a steady middle iteration (skip startup).</li>
<li><b>Labels vs hardware:</b> On the CPU/Python thread, find <code>:data_loading</code>, <code>:data_movement</code>, <code>:forward_step</code>, <code>:backward_step</code>. Note: these NVTX labels sit on the <b>CPU</b> timeline. Then look at the <b>same time windows</b> on CUDA HW — busy kernels, short <code>memcpy</code>, or mostly idle?</li>
<li><b>One iteration time (form Q1):</b> Mark the start of one <code>:data_loading</code> (or the first stage you see) and the end of that step’s <code>:backward_step</code>/<code>:optimizer_step</code>. Use the ruler / timestamps. What is the CPU wall-clock duration of one full iteration?</li>
<li><b>GPU peak memory (form Q2):</b> On CUDA HW / memory views in Nsight, find the GPU memory peak for this process. Which value matches what you read?</li>
<li><b>Longest CPU stage (form Q3):</b> Compare the CPU NVTX bar lengths for data loading, data movement, forward, and backward in one iteration. Which is longest? (Hint: longest CPU bar ≠ necessarily longest GPU work.)</li>
<li><b>Data-loading duration (form Q4):</b> Select one <code>:data_loading</code> range and read its duration. Which form option matches?</li>
<li><b>Sanity check before submitting:</b> Under the longest CPU stage, does CUDA HW show heavy compute or something else? Write one sentence: what is limiting this run?</li>
</ol>
</div>

---

## Single GPU Multiworkers

---

### Single GPU Multiworkers

![](images/profiling/dataloader_workers.webp){style="width: 70%; max-width: 1200px; margin-left: 0; margin-right: auto;"}

Main process send indexes to workers

- index 0 → worker 1  
- index 1 → worker 2  
- index 2 → worker 3  
- index 3 → worker 1  

---

### Single GPU Multiworkers

<div style="font-weight: bold; background-color: #ffcccc;">
```python
train_loader = DataLoader(train_dataset, 
                        batch_size=args.batch_size, 
                        num_workers=4,
                        pin_memory=False)
```
</div>

<div style="font-weight: bold; background-color: #ffcccc;">
```bash
sbatch --disable-dcgm single_gpu_training_profiling.sbatch --profile
```
</div>


### Single GPU Multiworkers

<div style="font-weight: bold; background-color: #ffcccc;; font-size: 0.8em; min-width: 1150px;">
```bash
File -> Open -> Single_GPU/Single_GPU/Report_01_multi_woker_unpined_non_blocking_False/nsys_logs/647_ATA_566_nsys_logs_rank.nsys-rep
```
</div>

Go to this link to answer the questions:

<div style="font-weight: bold; background-color: #f0d965; font-size: 0.6em; min-width: 800px; padding: 2px;">
  <a href="https://forms.gle/oxNaBnjP8cA9kPJw5" target="_blank">
    Single GPU Multiworkers
  </a>
</div>

<div style="padding: 15px 15px 15px 25px; background-color: #e8f5e9; border-left: 5px solid #4caf50; margin: 10px 0; font-size: 0.5em;">
<div style="font-weight: bold; margin-bottom: 8px; margin-left: -850px;">📘 Exercise</div>
<ul style="margin: 5px 0 0 20px;">
<li>Find `pt_data_worker` processes</li>
<li>How many pt_data_worker traces are created by Nsys?</li>
<li>Which part of training takes the most time: data loading, data movement, the forward pass, or the backward pass?</li>
<li>How long does it take until one data loading step is finished?</li>
</ul>
</div>



---

### Single GPU Multiworkers
![](images/profiling/multiworker_vs_zeroworkers.png){style="width: 70%; max-width: 1200px; margin-left: 0; margin-right: auto;"}


<!--
---

### Single GPU Multiworkers (Asyn. Transfer)

![](images/profiling/Asynchronous_Transfer.png){style="width: 45%; max-width: 1200px; margin-left: 0; margin-right: auto;"}

-->
---

## DMA (Direct Memory Access)


![](images/profiling/DMA.png){style="width: 45%; max-width: 1200px; margin-left: 0; margin-right: auto;"}

<div style="padding: 15px 15px 15px 25px; background-color: #e8f5e9; border-left: 5px solid #4caf50; margin: 10px 0; font-size: 0.7em;">
  <div style="font-weight: bold; margin-bottom: 8px;"></div>

  <ul style="margin: 5px 0 0 20px;">
    <li class="fragment">DMA allows some hardware to access memory without involving the CPU</li>
    <li class="fragment"> Copying between CPU & GPU uses DMA</li>
    <li class="fragment">What is the advatage of data transfer by the DMA?</li>
    <li class="fragment">What is the problem of data transfer by the DMA?</li>
  </ul>
</div>


---

## DMA (Direct Memory Access)


<div style="padding: 15px 15px 15px 25px; background-color: #e8f5e9; border-left: 5px solid #4caf50; margin: 10px 0; font-size: 0.7em;">
  <ul style="margin: 5px 0 0 20px;">
    <li class="fragment">
      DMA accesses memory using physical addresses.
      <ul>
        <li class="fragment">Cannot detect if the OS swaps a virtual page with another virtual page at the same physical address.</li>
      </ul>
    </li>
    <li class="fragment">
      How can we prevent data corruption during DMA transfers?
      <ul>
        <li class="fragment"><strong>Pinning Memory!!</strong></li>
      </ul>
    </li>
  </ul>
</div>


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


![](images/profiling/pageable_pinned.svg){style="width: 110%; max-width: 1200px; margin-left: 0; margin-right: auto;"}

---


### Single GPU (Asyn. Transfer)
<div style="font-weight: bold; background-color: #ffcccc;">
```python
train_loader = DataLoader(train_dataset, 
                        batch_size=args.batch_size, 
                        num_workers=4,
                        pin_memory=True)
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

<div style="font-weight: bold; background-color: #ffcccc;; font-size: 0.8em; min-width: 1150px;">
```bash
File -> Open -> Single_GPU/Report_02_multiwoker_pined_non_blocking_True/nsys_logs/107_RKN_402_nsys_logs_rank.nsys-rep
```
</div>

Go to this link to answer the questions:

<div style="font-weight: bold; background-color: #f0d965; font-size: 0.6em; min-width: 800px; padding: 2px;">
  <a href="https://forms.gle/Fi1EVARNaVRZcpsR8" target="_blank">
    Single GPU Multiworkers (Asyn. Transfer)
  </a>
</div>


<div style="padding: 15px 15px 15px 25px; background-color: #e8f5e9; border-left: 5px solid #4caf50; margin: 10px 0; font-size: 0.5em;">
<div style="font-weight: bold; margin-bottom: 8px; margin-left: -850px;">📘 Exercise</div>
<ul style="margin: 5px 0 0 20px;">
<li>Check one iteration of training (PyTorch trace)</li>
<li>Which part of training takes the most time: data loading, data movement, the forward pass, or the backward pass?</li>
<li>Is there any CUDA synchronization in the trace?</li>
</ul>
</div>

<!--

---

### Single GPU Multiworkers (Asyn. Transfer)
![](images/profiling/data_transfer_GPU_mem.png){style="width: 70%; max-width: 1200px; margin-left: 0; margin-right: auto;"}

---
-->

## DDP 

---

### DDP

<div style="font-weight: bold; background-color: #ffcccc;">
```bash
sbatch --disable-dcgm --nodes 1 ddp_training.sbatch --profile
```
</div>




<div style="padding: 15px 15px 15px 25px; background-color: #fff9c4; border-left: 5px solid #fbc02d; margin: 10px 0; font-size: 0.7em;">
<div style="font-weight: bold; margin-bottom: 8px; margin-left: -800px;"> Trace Path:</div>
<ul style="margin: 5px 0 0 20px; font-size: 0.8em">
Multi_GPUs/DDP/Single_node/Report_03_DDP_one_node/nsys_logs
</ul>
</div>



<div style="font-weight: bold; background-color: #ffcccc;; font-size: 0.8em; min-width: 800px;">
```bash
File -> Open -> select all .nsys-rep files --> create a multi-view report -> select all reports
```
</div>

Go to this link to answer the questions:

<div style="font-weight: bold; background-color: #f0d965; font-size: 0.6em; min-width: 800px; padding: 2px;">
  <a href="https://forms.gle/z6VoTiQ1gZjwZTRDA" target="_blank">
    DDP (4 GPUs)
  </a>
</div>




<div style="padding: 15px 15px 15px 25px; background-color: #e8f5e9; border-left: 5px solid #4caf50; margin: 10px 0; font-size: 0.5em;">
<div style="font-weight: bold; margin-bottom: 8px; margin-left: -850px;">📘 Exercise</div>
<ul style="margin: 5px 0 0 20px;">
<li>Which intra-node communication is active (PCIe or NVLink)?</li>
<li>Check NIC metrics (Why is there some traffic on one node?) </li>
<li>Check the NCCL trace inside of CUDA HW </li>
<li>How many times is the all-reduce operation called (check NCCL trace inside of CUDA HW)?  </li>
<li>Do the compute kernels overlap with the NCCL all-reduce operation? </li>
</ul>
</div>



---


### DDP

<div style="font-weight: bold; background-color: #ffcccc;">
```bash
sbatch --disable-dcgm --nodes 2 ddp_training.sbatch --profile
```
</div>




<div style="padding: 15px 15px 15px 25px; background-color: #fff9c4; border-left: 5px solid #fbc02d; margin: 10px 0; font-size: 0.7em;">
<div style="font-weight: bold; margin-bottom: 8px; margin-left: -800px;"> Trace Path:</div>
<ul style="margin: 5px 0 0 20px; font-size: 0.8em">
Multi_GPUs/DDP/Multi_nodes/Report_03_DDP_multi_nodes/GPU_Direct_RDMA_enable/nsys_logs
</ul>
</div>



<div style="font-weight: bold; background-color: #ffcccc;; font-size: 0.8em; min-width: 800px;">
```bash
File -> Open -> select all .nsys-rep files --> create a multi-view report -> select all reports
```
</div>



Go to this link to answer the questions:

<div style="font-weight: bold; background-color: #f0d965; font-size: 0.6em; min-width: 800px; padding: 2px;">
  <a href="https://forms.gle/KvvPEqvEnVLhVNrL8" target="_blank">
    DDP (8 GPUs)
  </a>
</div>




<div style="padding: 15px 15px 15px 25px; background-color: #e8f5e9; border-left: 5px solid #4caf50; margin: 10px 0; font-size: 0.5em;">
<div style="font-weight: bold; margin-bottom: 8px; margin-left: -850px;">📘 Exercise</div>
<ul style="margin: 5px 0 0 20px;">
<li>Which intra-node communication is active (PCIe or NVLink)?</li>
<li>Check the NCCL trace inside of CUDA HW </li>
<li>Do the compute kernels overlap with the NCCL all-reduce operation? </li>
<li>Do you think the training will be scalable?</li>
</ul>
</div>



---


### DDP

<div style="font-weight: bold; background-color: #ffcccc;">
```bash
NCCL_P2P_LEVEL=LOC sbatch --disable-dcgm --nodes 2 ddp_training.sbatch --profile
```
</div>




<div style="padding: 15px 15px 15px 25px; background-color: #fff9c4; border-left: 5px solid #fbc02d; margin: 10px 0; font-size: 0.7em;">
<div style="font-weight: bold; margin-bottom: 8px; margin-left: -800px;"> Trace Path:</div>
<ul style="margin: 5px 0 0 20px; font-size: 0.8em">
Multi_GPUs/DDP/Multi_nodes/Report_03_DDP_multi_nodes/GPU_Direct_RDMA_disable/nsys_logs
</ul>
</div>



<div style="font-weight: bold; background-color: #ffcccc;; font-size: 0.8em; min-width: 800px;">
```bash
File -> Open -> select all .nsys-rep files --> create a multi-view report -> select all reports
```
</div>



Go to this link to answer the questions:

<div style="font-weight: bold; background-color: #f0d965; font-size: 0.6em; min-width: 800px; padding: 2px;">
  <a href="https://forms.gle/XXF5EHMLQkRaM88k8" target="_blank">
    DDP (8 GPUs without GPU_Direct_RDMA)
  </a>
</div>




<div style="padding: 15px 15px 15px 25px; background-color: #e8f5e9; border-left: 5px solid #4caf50; margin: 10px 0; font-size: 0.5em;">
<div style="font-weight: bold; margin-bottom: 8px; margin-left: -850px;">📘 Exercise</div>
<ul style="margin: 5px 0 0 20px;">
<li>Check the NCCL trace inside of CUDA HW </li>
<li>Do the compute kernels overlap with the NCCL all-reduce operation? </li>
<li>Do you think the training will be scalable?</li>
</ul>
</div>



---



### DDP

<div style="font-weight: bold; background-color: #ffcccc;">
```bash
NCCL_IB_DISABLE=1 sbatch --disable-dcgm --nodes 2 ddp_training.sbatch --profile
```
</div>




<div style="padding: 15px 15px 15px 25px; background-color: #fff9c4; border-left: 5px solid #fbc02d; margin: 10px 0; font-size: 0.7em;">
<div style="font-weight: bold; margin-bottom: 8px; margin-left: -800px;"> Trace Path:</div>
<ul style="margin: 5px 0 0 20px; font-size: 0.8em">
Multi_GPUs/DDP/Multi_nodes/Report_03_DDP_multi_nodes/NOT_IB_USAGE
</ul>
</div>



<div style="font-weight: bold; background-color: #ffcccc;; font-size: 0.8em; min-width: 800px;">
```bash
File -> Open -> select all .nsys-rep files --> create a multi-view report -> select all reports
```
</div>

Go to this link to answer the questions:

<div style="font-weight: bold; background-color: #f0d965; font-size: 0.6em; min-width: 800px; padding: 2px;">
  <a href="https://forms.gle/zn58hSUSLZZWPzLX8" target="_blank">
    DDP (8 GPUs without Infiniband usage)
  </a>
</div>




<div style="padding: 15px 15px 15px 25px; background-color: #e8f5e9; border-left: 5px solid #4caf50; margin: 10px 0; font-size: 0.5em;">
<div style="font-weight: bold; margin-bottom: 8px; margin-left: -850px;">📘 Exercise</div>
<ul style="margin: 5px 0 0 20px;">
<li>Do the compute kernels overlap with the NCCL all-reduce operation? </li>
<li>Do you think the training will be scalable?</li>
</ul>
</div>



---


### DDP scaling

![](images/profiling/DDP_scaling.png)

---

