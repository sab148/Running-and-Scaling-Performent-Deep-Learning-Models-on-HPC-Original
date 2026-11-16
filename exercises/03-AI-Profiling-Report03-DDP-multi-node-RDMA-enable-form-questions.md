# DDP multi-node with GPUDirect RDMA (8 GPUs) — Form questions (with answers)

**Form:** [DDP (8 GPUs)](https://forms.gle/KvvPEqvEnVLhVNrL8)  
**Trace:** `Multi_GPUs/DDP/Multi_nodes/Report_03_DDP_multi_nodes/GPU_Direct_RDMA_enable/nsys_logs/nsys_logs_rank_1.nsys-rep`  
(largest rank file: ~34 MB)  
**Setup:** 2 nodes × 4× A100-SXM4-40GB, NCCL DDP, GPUDirect RDMA enabled

## How to read the trace

- Zoom past startup / NCCL warm-up to a **steady middle** iteration.
- **Intra-node:** GPU metrics → NVLink vs PCIe.
- **Inter-node:** NIC / InfiniBand metrics should show traffic (RDMA path).
- **Overlap:** on CUDA HW, compare non-NCCL compute vs `ncclDevKernel_AllReduce_...` in the same window.

---

## Q1. Intra-node communication

Which intra-node communication is active (PCIe or NVLink)?

- A) PCIe
- B) NVLink
- C) InfiniBand only (no intra-node GPU fabric)
- D) Neither PCIe nor NVLink

**Answer: B) NVLink**

Still A100-SXM4 on each node: on-node collectives use NVLink. NVLink user-data metrics spike during collectives; PCIe stays near baseline. InfiniBand carries **inter-node** traffic (visible on NIC metrics), not the answer to this intra-node question.

---

## Q2. Compute vs NCCL AllReduce overlap

In a steady middle iteration, how do non-NCCL compute kernels relate to NCCL AllReduce on **CUDA HW**?

(Same options as the single-node DDP form — compare across slides.)

- A) **No overlap** — AllReduce and compute are sequential; compute kernels stop while AllReduce runs
- B) **Partial overlap** — some AllReduce windows overlap compute; others do not
- C) **Strong overlap** — most AllReduce windows clearly overlap compute kernels
- D) **Comm-dominated / little useful overlap** — during AllReduce the GPU is mostly waiting on communication (long AllReduce, sparse or no concurrent compute)

**Answer: B) Partial overlap**

Mid-run sample: ~50/100 AllReduce intervals overlap some non-NCCL compute (e.g. `layer_norm_grad_...`).  
Note: AllReduce is **longer** than single-node (~3.2 ms avg here vs ~0.4 ms on 1 node) because of multi-node IB, but overlap category is still partial.
