# DDP multi-node without GPUDirect RDMA (`NCCL_P2P_LEVEL=LOC`) — Form questions (with answers)

**Form:** [DDP (8 GPUs without GPU_Direct_RDMA)](https://forms.gle/XXF5EHMLQkRaM88k8)  
**Trace:** `Multi_GPUs/DDP/Multi_nodes/Report_03_DDP_multi_nodes/GPU_Direct_RDMA_disable/nsys_logs/nsys_logs_rank_2.nsys-rep`  
(largest rank file: ~42 MB)  
**Setup:** 2 nodes × 4× A100-SXM4-40GB, NCCL DDP, `NCCL_P2P_LEVEL=LOC` (GPUDirect RDMA / high-level P2P disabled)

## How to read the trace

- `NCCL_P2P_LEVEL=LOC` limits GPU P2P to **PCIe locality** — NVLink P2P is not used.
- Zoom to a **steady middle** iteration.
- Compare **NVLink vs PCIe** GPU metrics to the previous RDMA-enable slide.
- On CUDA HW, inspect `ncclDevKernel_AllReduce_...` length and overlap with compute.

---

## Q1. Intra-node communication path (`NCCL_P2P_LEVEL=LOC`)

With `NCCL_P2P_LEVEL=LOC`, which path carries intra-node GPU communication during collectives?

- A) NVLink (same as the default / RDMA-enable run)
- B) PCIe (NVLink unused / flat)
- C) InfiniBand only (no on-node GPU fabric activity)
- D) No communication visible on GPU metrics

**Answer: B) PCIe (NVLink unused / flat)**

In this trace NVLink metrics stay at **0**; PCIe RX/TX spike hard (peaks ~90%+). That is the point of `NCCL_P2P_LEVEL=LOC`: P2P is restricted so traffic falls back to the slower PCIe / host-local path instead of NVLink.

---

## Q2. Compute vs NCCL AllReduce overlap

In a steady middle iteration, how do non-NCCL compute kernels relate to NCCL AllReduce on **CUDA HW**?

(Same options as the other DDP forms.)

- A) **No overlap** — AllReduce and compute are sequential; compute kernels stop while AllReduce runs
- B) **Partial overlap** — some AllReduce windows overlap compute; others do not
- C) **Strong overlap** — most AllReduce windows clearly overlap compute kernels
- D) **Comm-dominated / little useful overlap** — during AllReduce the GPU is mostly waiting on communication (long AllReduce, sparse or no concurrent compute)

**Answer: B) Partial overlap**

Mid-run sample: ~50/100 AllReduce intervals still overlap some compute (e.g. `layer_norm_grad_...`).  
**But** AllReduce is much slower than with GPUDirect RDMA (~**6.7 ms** avg here vs ~**3.2 ms** with RDMA enable vs ~**0.4 ms** single-node NVLink). So even with partial overlap, communication is more expensive — a hint that this setting does **not** scale as well.
