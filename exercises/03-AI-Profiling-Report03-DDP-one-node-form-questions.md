# DDP single node (4 GPUs) — Form questions (with answers)

**Form:** [DDP (4 GPUs)](https://forms.gle/z6VoTiQ1gZjwZTRDA)  
**Trace:** `Multi_GPUs/DDP/Single_node/Report_03_DDP_one_node/nsys_logs/nsys_logs_rank_2.nsys-rep`  
(largest rank file: ~39 MB; other ranks are ~16 MB)  
**Setup:** 1 node × 4× A100-SXM4-40GB, NCCL DDP

## How to read the trace

- Zoom past startup / NCCL warm-up to a **steady middle** iteration.
- **GPU metrics:** compare **NVLink** vs **PCIe** during collectives.
- **CUDA HW / NCCL:** look for `ncclDevKernel_AllReduce_...` (and NVTX `ncclAllReduce`).
- **Overlap:** in the same window, are non-NCCL compute kernels active while AllReduce runs?

---

## Q1. Intra-node communication

Which intra-node communication is active during training collectives?

- A) PCIe
- B) NVLink
- C) InfiniBand only
- D) Neither PCIe nor NVLink (host staging only)

**Answer: B) NVLink**

On A100-SXM4, NCCL uses NVLink for on-node allreduce. In this trace, NVLink user-data metrics spike (peaks ~70%+) around collectives, while PCIe stays near baseline (~1%).

---

## Q2. All-reduce count

How many times is the all-reduce operation called? Check the NCCL / CUDA HW row (`ncclDevKernel_AllReduce_...`). Choose the closest value.

- A) ~50
- B) ~215
- C) ~430
- D) ~860

**Answer: C) ~430**

- NVTX range `ncclAllReduce`: **431**
- CUDA HW kernel `ncclDevKernel_AllReduce_Sum_f32_RING_LL`: **433**

(Small NVTX vs kernel differences are normal; count on CUDA HW / NCCL as the form asks.)

---

## Q3. Compute vs NCCL AllReduce overlap

In a steady middle iteration, how do non-NCCL compute kernels relate to NCCL AllReduce on **CUDA HW**?

(Use the same options on later DDP slides — the correct letter can change with NVLink / RDMA / no RDMA / no IB.)

- A) **No overlap** — AllReduce and compute are sequential; compute kernels stop while AllReduce runs
- B) **Partial overlap** — some AllReduce windows overlap compute; others do not
- C) **Strong overlap** — most AllReduce windows clearly overlap compute kernels
- D) **Comm-dominated / little useful overlap** — during AllReduce the GPU is mostly waiting on communication (long AllReduce, sparse or no concurrent compute)

**Answer: B) Partial overlap**

Mid-run sample: ~half of AllReduce intervals overlap some non-NCCL compute (e.g. `layer_norm_grad_...`); many do not. Typical of bucketed DDP backward, not full fusion.
