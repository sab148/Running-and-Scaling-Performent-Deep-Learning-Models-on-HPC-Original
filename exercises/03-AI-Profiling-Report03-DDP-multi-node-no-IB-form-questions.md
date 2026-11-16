# DDP multi-node without InfiniBand (`NCCL_IB_DISABLE=1`) — Form questions (with answers)

**Form:** [DDP (8 GPUs without Infiniband usage)](https://forms.gle/zn58hSUSLZZWPzLX8)  
**Trace:** `Multi_GPUs/DDP/Multi_nodes/Report_03_DDP_multi_nodes/NOT_IB_USAGE/nsys_logs/nsys_logs_rank_0.nsys-rep`  
(largest rank file: ~27 MB)  
**Setup:** 2 nodes × 4× A100-SXM4-40GB, NCCL DDP, `NCCL_IB_DISABLE=1` (no InfiniBand for NCCL)

## How to read the trace

- `NCCL_IB_DISABLE=1` makes NCCL fall back to **sockets / Ethernet** instead of IB / GPUDirect RDMA.
- Zoom to a **steady middle** iteration.
- Compare AllReduce bar length and **SMs Active** to the RDMA-enable slide.
- Overlap options are the same as on the other DDP forms — the correct letter can change here.

---

## Q1. Compute vs NCCL AllReduce overlap

In a steady middle iteration, how do non-NCCL compute kernels relate to NCCL AllReduce on **CUDA HW**?

- A) **No overlap** — AllReduce and compute are sequential; compute kernels stop while AllReduce runs
- B) **Partial overlap** — some AllReduce windows overlap compute; others do not
- C) **Strong overlap** — most AllReduce windows clearly overlap compute kernels
- D) **Comm-dominated / little useful overlap** — during AllReduce the GPU is mostly waiting on communication (long AllReduce, sparse or no concurrent compute)

**Answer: D) Comm-dominated / little useful overlap**

AllReduce is very long (~**32 ms** avg, up to ~**68 ms**) vs ~3 ms with GPUDirect RDMA and ~0.4 ms single-node NVLink.  
SMs Active averages only ~**5%**. Some AllReduce windows still touch a bit of compute (~15% of AllReduce duration on average), but the GPU is mostly **waiting on communication** — not useful compute/comm overlap.

---

## Q2. Will this training scale well?

Do you think this multi-node setup (InfiniBand disabled) will scale well as you add nodes/GPUs?

- A) Yes — communication is short and well overlapped with compute
- B) Somewhat — AllReduce is slower but still comparable to single-node
- C) No — communication dominates; adding nodes will make collectives worse over Ethernet/sockets
- D) Scaling only depends on model size, not the network path

**Answer: C) No — communication dominates**

With IB disabled, NCCL uses a slow path. Collectives dominate step time and the GPU sits idle. This is the anti-pattern vs the GPUDirect RDMA slide: **network path matters for DDP scalability**.
