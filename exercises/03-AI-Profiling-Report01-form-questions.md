# Single GPU Multiworkers — Form questions (with answers)

**Form:** [Single GPU Multiworkers](https://docs.google.com/forms/d/e/1FAIpQLSepMiQ3Dq7Hyx6G7N_yniDwbsBA9mFE0SH0y9yoRZZFDQ4WDA/viewform)  
**Trace:** `Single_GPU/Report_01_multi_woker_unpined_non_blocking_False/nsys_logs/647_ATA_566_nsys_logs_rank.nsys-rep`  
**Settings:** `num_workers=4`, `pin_memory=False`, `non_blocking=False`  
**Compare with:** Report 00 (`num_workers=0`, same pin / non_blocking)

## How to read this trace

- Zoom to a **steady middle** iteration (skip startup).
- NVTX labels (`data_loading`, `data_movement`, `forward_step`, …) are on the **CPU/Python** timeline of the **main** process.
- Look for separate rows named **`pt_data_worker`** (DataLoader worker subprocesses).
- Measure stage wall time on the **CPU** timeline; use **CUDA HW** to see compute vs a short Host→Device `memcpy`.

---

## Five main form questions (replace the existing ones)

### Q1. Worker processes

In Nsight, how many **`pt_data_worker`** worker processes / PyTorch worker annotations do you find for this run?

- A) 0 — same as Report 00  
- B) 1  
- C) 2  
- D) 4  

**Answer: D) 4**  
(`num_workers=4`. Report 00 has none. You may also see several threads under each worker; count **worker processes**, not every thread.)

---

### Q2. What do the workers do on the timeline?

Which statement best matches the multi-worker trace?

- A) Each `pt_data_worker` has its own CUDA HW row with long GEMM kernels  
- B) Workers appear as separate **CPU** processes; GPU compute / H2D stays on the **main** process  
- C) Workers replace the main process; there is no main Python row  
- D) Workers only appear inside the CUDA HW `memcpy` row  

**Answer: B)**  
Workers load/collate on CPU and fill the queue. Only the main process does `.to(device)` and training.

---

### Q3. Main-process `data_loading` time

On the **main** process CPU timeline, what is the approximate duration of one steady `data_loading` range?

- A) ~1–2 ms  
- B) ~8–10 ms  
- C) ~30–35 ms  
- D) ~60–70 ms  

**Answer: A) ~1–2 ms**  
(Median ~1.7 ms here, via `_MultiProcessingDataLoaderIter.__next__`. In Report 00 it was ~9 ms.)

---

### Q4. Longest stage after enabling workers

In a steady iteration of Report 01, which NVTX stage is still the **longest** on the main CPU timeline?

- A) data loading  
- B) data movement  
- C) forward pass  
- D) backward pass  

**Answer: B) data movement**  
(~67 ms median. Forward ~9 ms, backward ~17 ms, data loading ~1.7 ms.)

---

### Q5. Did multi-workers fix the step bottleneck?

Compared with Report 00, what is the fairest conclusion from this trace?

- A) Yes — workers removed the bottleneck; the step is now GPU-bound and much faster  
- B) Partially — `data_loading` got much shorter, but overall step time barely changed because `data_movement` still dominates (~100 ms/iter)  
- C) No — `data_loading` got worse than Report 00  
- D) Workers only help the optimizer step  

**Answer: B)**  
Loading improved (~9 → ~1.7 ms), but `data_movement` remains ~60–70 ms and the full iteration stays ~98–99 ms. Next experiment: `pin_memory=True` (and typically `non_blocking=True` on `.to(device)`).

---

## Quick answer key

| Q  | Answer |
|----|--------|
| Q1 | D — 4 `pt_data_worker` |
| Q2 | B — workers on CPU; GPU on main |
| Q3 | A — ~1–2 ms |
| Q4 | B — data movement |
| Q5 | B — loading fixed; step still movement-bound |

## Measured steady-state summary (for instructors)

| Stage (CPU NVTX) | Report 00 (`workers=0`) | Report 01 (`workers=4`) |
|------------------|-------------------------|-------------------------|
| data_loading     | ~9.2 ms                 | ~1.7 ms                 |
| data_movement    | ~59 ms                  | ~67 ms                  |
| forward_step     | ~9.4 ms                 | ~9.4 ms                 |
| backward_step    | ~16.6 ms                | ~16.7 ms                |
| optimizer_step   | ~3.0 ms                 | ~3.0 ms                 |
| full iteration   | ~98 ms                  | ~99 ms                  |

Under `data_movement`: Host→Device `memcpy` on CUDA HW is only ~0.01 ms; the long bar is CPU/pageable-transfer waiting (same class of problem as Report 00).
