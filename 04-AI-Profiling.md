---
author: Alexandre Strube // Sabrina Benassou // Javad Kasravi
title: Bringing Deep Learning Workloads to JSC supercomputers
subtitle: AI Profiling
date: September 16, 2025
---
<style>
.reveal h3 {
  font-size: 40px;
}

.reveal ul li {
  margin-bottom: 20px;  /* adjust as needed */
}
</style>

---

## Agenda

- Performance Terminology
- Node Communications
- NVIDIA AI Profiling tools
- Single GPU training
- Distributed Data Parallel (DDP)
    - Single node training
    - Multi node training
- DDP scaling

---

### Performance Terminology

- Latency: the time it takes for one GPU or node to start exchanging information with another GPU or node.

- Bandwidth: The maximum amount of data that can be transferred per unit of time between GPUs, CPUs, or nodes.

- Host: CPU + system memory.

- Device: GPU + GPU memory.

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
| Naive communication               | ~16 GB/s              <span style="font-size:2em">🐢</span> |
| PCIe Bus P2P communication       | ~32 GB/s              <span style="font-size:2em">🚗</span> |
| GPUDirect P2P communication             | 600 GB/s total per GPU          <span style="font-size:2em">🏎️</span> |

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
| GPUDirect Without RDMA              | ~16 GB/s              <span style="font-size:2em">🐢</span> |
| GPUDirect With RDMA             | ~ 50 GB/s (2 HDR InfiniBand)          <span style="font-size:2em">🏎️</span> |

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
    srun torchrun_jsc \
        --nnodes=$SLURM_NNODES \
        --rdzv_backend c10d \
        --nproc_per_node=gpu \
        --rdzv_id $RANDOM \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        --no-python ./run_profile.sh train/ddp_training.py --profile
```
</div>

Inside of `run_profile.sh`

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

### Single GPU Single Process

<div style="font-weight: bold; background-color: #ffcccc;">
```python
train_loader = DataLoader(train_dataset, 
                        batch_size=args.batch_size, 
                        shuffle=True,
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

### Single GPU Single Process

Move the trace file to your local machine by running:
<div style="font-weight: bold; background-color: #ffcccc; min-width: 1050px">
```bash
scp -r {user_name}@jureca.fz-juelich.de:/p/project1/atmlaml/HPC_Supporter_Workshop/nsys_traces .
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
<li>Find the main Python process with CUDA HW thread</li>
<li>Explore PyTorch & NVTX annotations (main Python process)</li>
<li>How long does it take until one iteration is finished (data transfers, forward, backward, ...)?</li>
<li>Explore CUDA HW thread and what is the GPU peak memory?</li>
<li>Explore PyTorch & NVTX annotations (inside CUDA HW)</li>
<li>Why are some annotations missing inside CUDA HW?</li>
</ul>
</div>

---

## Single GPU Multiprocesses

---

### Single GPU Multiprocesses

![](images/profiling/dataloader_workers.webp){style="width: 70%; max-width: 1200px; margin-left: 0; margin-right: auto;"}

Main process send indexes to workers

- index 0 → worker 1  
- index 1 → worker 2  
- index 2 → worker 3  
- index 3 → worker 1  

---

### Single GPU Multiprocesses

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

### Single GPU Multiprocesses
![](images/profiling/data_transfer_sys_mem.png){style="width: 70%; max-width: 1200px; margin-left: 0; margin-right: auto;"}

---

### Single GPU Multiprocesses

<div style="padding: 15px 15px 15px 25px; background-color: #e8f5e9; border-left: 5px solid #4caf50; margin: 10px 0; font-size: 0.7em;">
<div style="font-weight: bold; margin-bottom: 8px; margin-left: -850px;">📘 Exercise</div>
<ul style="margin: 5px 0 0 20px;">
<li>Check one iteration of training</li>
<li>Which operation dominates the training time (per iteration)?</li>
</ul>
</div>

---

### Single GPU Multiprocesses (Asyn. Transfer)

![](images/profiling/Asynchronous_Transfer.png){style="width: 45%; max-width: 1200px; margin-left: 0; margin-right: auto;"}

---

### Single GPU (Asyn. Transfer)
<div style="font-weight: bold; background-color: #ffcccc;">
```python
train_loader = DataLoader(train_dataset, 
                        batch_size=args.batch_size, 
                        shuffle=False,
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

### Single GPU Multiprocesses (Asyn. Transfer)

<div style="font-weight: bold; background-color: #ffcccc;; font-size: 0.8em; min-width: 1150px;">
```bash
File -> Open -> Single_GPU/Report_02_multiwoker_pined_non_blocking_True/nsys_logs/107_RKN_402_nsys_logs_rank.nsys-rep
```
</div>

<div style="padding: 15px 15px 15px 25px; background-color: #e8f5e9; border-left: 5px solid #4caf50; margin: 10px 0; font-size: 0.7em;">
<div style="font-weight: bold; margin-bottom: 8px; margin-left: -850px;">📘 Exercise</div>
<ul style="margin: 5px 0 0 20px;">
<li>Check one iteration of training</li>
<li>Which part of code dominates the training time (per iteration)?</li>
</ul>
</div>

---

### Single GPU Multiprocesses (Asyn. Transfer)
![](images/profiling/data_transfer_GPU_mem.png){style="width: 70%; max-width: 1200px; margin-left: 0; margin-right: auto;"}

---

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

