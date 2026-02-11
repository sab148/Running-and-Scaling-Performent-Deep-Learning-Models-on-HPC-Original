---
author: Javad Kasravi
title: AI Profiling
date: February 10, 2026
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
Go to the following link and download the **Nsight System 1.3.2025**:

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
    --python-sampling=true \
    --python-sampling-frequency=1 \
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

### Single GPU without dataloader Worker

<div style="font-weight: bold; background-color: #ffcccc;">
```python
train_loader = DataLoader(train_dataset, 
                        batch_size=args.batch_size, 
                        num_workers=0,
                        pin_memory=False)
```
</div>

<div style="font-weight: bold; background-color: #ffcccc;">
```bash
sbatch --disable-dcgm single_gpu_training.sbatch --profile
```
</div>

Only the main process transfers data to system memory.

---

### Single GPU without dataloader Worker

Move the trace folder to your local machine by running:
<div style="font-weight: bold; background-color: #ffcccc; font-size: 0.8em; min-width: 1050px">
```bash
scp -r -4 <user>@jureca.fz-juelich.de:/p/project1/training2560/AI_profiling/Nsys_trace_update_Jan_2026 .
```
</div>

<div style="font-weight: bold; background-color: #ffcccc;; font-size: 0.8em; min-width: 1050px;">
```bash
File -> Open -> Single_GPU/Report_00_zero_woker_unpined_non_blocking_False/nsys_logs/540_WVB_725_nsys_logs_rank.nsys-rep
```
</div>

Use + and − keys to zoom in and out

<div style="padding: 15px 15px 15px 25px; background-color: #e8f5e9; border-left: 5px solid #4caf50; margin: 10px 0; font-size: 0.7em;">
<div style="font-weight: bold; margin-bottom: 8px; margin-left: -850px;">📘 Exercise</div>
<ul style="margin: 5px 0 0 20px;">
<li>Find the Python process with CUDA HW </li>
<li>Find the Python thread inside the above process</li>
<li>Explore PyTorch & NVTX annotations</li>
<li>How long does it take until one iteration is finished (data transfers, forward, backward, ...)?</li>
<li>Explore CUDA HW thread and what is the GPU peak memory?</li>
<li>Explore PyTorch & NVTX annotations (inside CUDA HW)</li>
</ul>
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
sbatch --disable-dcgm single_gpu_training.sbatch --profile
```
</div>


### Single GPU Multiworkers

<div style="font-weight: bold; background-color: #ffcccc;; font-size: 0.8em; min-width: 1150px;">
```bash
File -> Open -> Single_GPU/Single_GPU/Report_01_multi_woker_unpined_non_blocking_False/nsys_logs/647_ATA_566_nsys_logs_rank.nsys-rep
```
</div>


<div style="padding: 15px 15px 15px 25px; background-color: #e8f5e9; border-left: 5px solid #4caf50; margin: 10px 0; font-size: 0.7em;">
<div style="font-weight: bold; margin-bottom: 8px; margin-left: -850px;">📘 Exercise</div>
<ul style="margin: 5px 0 0 20px;">
<li>Find `pt_data_worker` processes</li>
<li>How many pt_data_worker traces are created by Nsys?</li>
<li>How can we determine that this setup is not I/O-bound?</li>
</ul>
</div>



---

### Single GPU Multiworkers
![](images/profiling/multiworker_vs_zeroworkers.png){style="width: 70%; max-width: 1200px; margin-left: 0; margin-right: auto;"}

---

### Single GPU Multiworkers

<div style="padding: 15px 15px 15px 25px; background-color: #e8f5e9; border-left: 5px solid #4caf50; margin: 10px 0; font-size: 0.7em;">
<div style="font-weight: bold; margin-bottom: 8px; margin-left: -850px;">📘 Exercise</div>
<ul style="margin: 5px 0 0 20px;">
<li>Check one iteration of training (PyTorch trace)</li>
<li>Which operation dominates the training time (per iteration)?</li>
</ul>
</div>

<!--
---

### Single GPU Multiworkers (Asyn. Transfer)

![](images/profiling/Asynchronous_Transfer.png){style="width: 45%; max-width: 1200px; margin-left: 0; margin-right: auto;"}

-->
---

## DMA (Direct Memory Access)


![](images/profiling/DMA.png){style="width: 45%; max-width: 1200px; margin-left: 0; margin-right: auto;"}

<div style="padding: 15px 15px 15px 25px; background-color: #e8f5e9; border-left: 5px solid #4caf50; margin: 10px 0; font-size: 0.7em;">
  <div style="font-weight: bold; margin-bottom: 8px;">📌 Questions</div>

  <ul style="margin: 5px 0 0 20px;">
    <li class="fragment">What does DMA do?</li>
    <li class="fragment">What is the problem with data transfer by the CPU?</li>
    <li class="fragment">What is the problem of data transfer by the DMA?</li>
    <li class="fragment">How can we prevent data corruption during DMA transfers?</li>
  </ul>
</div>


## 

![](images/profiling/cuda_q.png){style="width: 60%; max-width: 1200px; margin-left: 0; margin-right: auto;"}

How does `cudaMemcpy` copy data from host to device?

---

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
sbatch --disable-dcgm single_gpu_training.sbatch --profile
```
</div>

---

### Single GPU Multiworkers (Asyn. Transfer)

<div style="font-weight: bold; background-color: #ffcccc;; font-size: 0.8em; min-width: 1150px;">
```bash
File -> Open -> Single_GPU/Report_02_multiwoker_pined_non_blocking_True/nsys_logs/107_RKN_402_nsys_logs_rank.nsys-rep
```
</div>

<div style="padding: 15px 15px 15px 25px; background-color: #e8f5e9; border-left: 5px solid #4caf50; margin: 10px 0; font-size: 0.7em;">
<div style="font-weight: bold; margin-bottom: 8px; margin-left: -850px;">📘 Exercise</div>
<ul style="margin: 5px 0 0 20px;">
<li>Check one iteration of training (PyTorch trace)</li>
<li>Which part of code dominates the training time (per iteration)?</li>
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




<div style="padding: 15px 15px 15px 25px; background-color: #e8f5e9; border-left: 5px solid #4caf50; margin: 10px 0; font-size: 0.7em;">
<div style="font-weight: bold; margin-bottom: 8px; margin-left: -850px;">📘 Exercise</div>
<ul style="margin: 5px 0 0 20px;">
<li>Which intra-node communication is active (PCIe or NVLink)?</li>
<li>Check NIC metrics (why? There is some traffic on one node) </li>
<li>Check the NCCL trace inside the GPU </li>
<li>Check the number of all-reduce calls </li>
<li>Check the overlap between the all-reduce calls and the compute kernels inside the GPU </li>
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




<div style="padding: 15px 15px 15px 25px; background-color: #e8f5e9; border-left: 5px solid #4caf50; margin: 10px 0; font-size: 0.7em;">
<div style="font-weight: bold; margin-bottom: 8px; margin-left: -850px;">📘 Exercise</div>
<ul style="margin: 5px 0 0 20px;">
<li>Which intra-node communication is active (PCIe or NVLink)?</li>
<li>Check NIC metrics & compre the traffic with previous run</li>
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




<div style="padding: 15px 15px 15px 25px; background-color: #e8f5e9; border-left: 5px solid #4caf50; margin: 10px 0; font-size: 0.7em;">
<div style="font-weight: bold; margin-bottom: 8px; margin-left: -850px;">📘 Exercise</div>
<ul style="margin: 5px 0 0 20px;">
<li>Which intra-node communication is active (PCIe or NVLink)?</li>
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


<div style="padding: 15px 15px 15px 25px; background-color: #e8f5e9; border-left: 5px solid #4caf50; margin: 10px 0; font-size: 0.7em;">
<div style="font-weight: bold; margin-bottom: 8px; margin-left: -850px;">📘 Exercise</div>
<ul style="margin: 5px 0 0 20px;">
<li>Which intra-node communication is active (PCIe or NVLink)?</li>
<li>Check the overlap between the all-reduce calls and the compute kernels inside the GPU </li>
<li>Do you think the training will be scalable?</li>
</ul>
</div>



---


### DDP scaling

![](images/profiling/DDP_scaling.png)

---

