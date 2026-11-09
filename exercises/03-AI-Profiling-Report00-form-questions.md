# Single GPU without dataloader Worker — Form questions (with answers)

**Trace:** `Single_GPU/Report_00_zero_woker_unpined_non_blocking_False/nsys_logs/540_WVB_725_nsys_logs_rank.nsys-rep`  
**Settings:** `num_workers=0`, `pin_memory=False`, `non_blocking=False`

## How to read the trace

- NVTX labels (`:data_loading`, `:data_movement`, `:forward_step`, …) are drawn on the **CPU/Python** timeline.
- Use **CUDA HW** under the same time window to see whether the GPU is busy (kernels), doing a short `memcpy`, or mostly idle.
- For **iteration / stage wall time**, measure on the **CPU** timeline.
- For **where the compute runs**, look at **CUDA HW**.
- Zoom to a **steady middle** iteration (not startup).

---

## Q1. One iteration — CPU wall time

**Using the CPU timeline**, what is the approximate wall-clock duration of one training iteration from the start of `:data_loading` to the end of `:optimizer_step`? Choose the closest value.

- A) 90 ms
- B) 150 ms
- C) 200 ms
- D) 250 ms

**Answer: A) 90 ms**  
(~98 ms in this trace; closest option.)

---

## Q2. GPU peak memory

In Nsight, what **GPU peak memory** do you read for this process?

- A) 20.4 G
- B) 22.4 G
- C) 24.4 G
- D) 25.4 G

**Answer: D) 25.4 G**  
(~25.37 GiB device memory peak.)

---

## Q3. Longest stage on the CPU timeline

Which NVTX stage has the **longest bar on the CPU/Python timeline** (not CUDA HW)?

- A) data loading
- B) data movement
- C) forward pass
- D) backward pass

**Answer: B) data movement**  
(~59 ms CPU wall time per step.)

---

## Q4. Data loading — CPU wall time

What is the duration of one `:data_loading` range on the **CPU** thread?

- A) 1–5 ms
- B) 8–10 ms
- C) 15–25 ms
- D) 30–35 ms

**Answer: B) 8–10 ms**  
(~9.2 ms average.)

---

## Q5. Where do the annotations appear?

The NVTX ranges `:data_loading`, `:data_movement`, `:forward_step`, and `:backward_step` are drawn on:

- A) the CUDA HW row only
- B) the CPU/Python thread only
- C) both rows as identical duplicated bars
- D) only on stages that run on the GPU

**Answer: B) the CPU/Python thread only**

---

## Q6. Interpreting `:data_movement`

Under `:data_movement` / `Tensor.to`, what do you observe?

- A) Long CPU bar and long GPU compute kernels filling most of that window
- B) Long CPU bar, but only a short CUDA `memcpy` / transfer on CUDA HW
- C) No CPU bar; all time appears only on CUDA HW
- D) GPU is fully busy with GEMM/backward kernels during that whole range

**Answer: B) Long CPU bar, but only a short CUDA `memcpy` / transfer on CUDA HW**  
(CPU ~59 ms vs GPU H2D ~0.17 ms — pageable / unpinned transfer wait.)

---

## Q7. What is the “forward time”?

For `:forward_step`, which statement is correct?

- A) The CPU NVTX bar (~10 ms) is the full forward compute time; the GPU does almost no work
- B) Forward compute runs on the GPU (CUDA kernels); the CPU NVTX bar is mainly host launch/phase time and can be shorter than the GPU work
- C) Forward runs only on the CPU because the label appears on the CPU row
- D) CPU and GPU forward times are always exactly equal

**Answer: B) Forward compute runs on the GPU; the CPU NVTX bar is mainly host launch/phase time**  
(CPU `:forward_step` ~9–10 ms; GPU projected forward work ~25–27 ms.)

---

## Q8. Why does backward show CPU and GPU activity?

Why can `:backward_step` show activity on **both** the CPU thread and CUDA HW?

- A) Gradients are computed on the CPU; the GPU only copies results
- B) Nsight duplicates the same work on both rows for visualization
- C) The CPU runs autograd (graph walk / kernel launches); the GPU runs the gradient kernels
- D) Backward is purely CPU when `num_workers=0`

**Answer: C) The CPU runs autograd (graph walk / kernel launches); the GPU runs the gradient kernels**

---

## Q9. Overall bottleneck (conclusion)

In this run (`num_workers=0`, `pin_memory=False`), what mainly limits step time?

- A) GPU forward/backward math
- B) Data loading on the CPU
- C) Host → device data movement / transfer waiting (long CPU `:data_movement`)
- D) Optimizer step

**Answer: C) Host → device data movement / transfer waiting (long CPU `:data_movement`)**

---

## Q10. Short answer (optional)

In one sentence: if an NVTX label appears on the CPU row, does that mean the work ran on the CPU? Explain using forward or data movement.

**Expected answer:**  
No. NVTX marks the host phase on the CPU timeline. Check CUDA HW under the same window: forward shows GPU kernels (GPU compute), while data movement shows a long CPU bar with only a short `memcpy` (CPU/transfer wait).

---

## Quick answer key

| Q  | Answer |
|----|--------|
| Q1 | A — 90 ms |
| Q2 | D — 25.4 G |
| Q3 | B — data movement |
| Q4 | B — 8–10 ms |
| Q5 | B — CPU/Python thread only |
| Q6 | B — long CPU bar, short memcpy |
| Q7 | B — GPU compute; CPU bar is phase/launch time |
| Q8 | C — CPU autograd launches; GPU computes grads |
| Q9 | C — H2D / data-movement waiting |
| Q10 | NVTX ≠ device; always check CUDA HW |
