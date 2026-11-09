# Single GPU Multiworkers (Async Transfer) — Form questions (with answers)

**Form:** [Single GPU Multiworkers (Asyn. Transfer)](https://forms.gle/Fi1EVARNaVRZcpsR8)  
**Trace:** `Single_GPU/Report_02_multiwoker_pined_non_blocking_True/nsys_logs/107_RKN_402_nsys_logs_rank.nsys-rep`  
**Settings:** `num_workers=4`, `pin_memory=True`, `non_blocking=True`  
**Compare with:** Report 01 (`num_workers=4`, `pin_memory=False`, `non_blocking=False`)

## How to read this trace

- Zoom to a **steady middle** iteration (skip startup).
- NVTX labels (`data_loading`, `data_movement`, `forward_step`, …) are on the **main** process **CPU** timeline.
- Under the same window, check **CUDA HW** for Host→Device `memcpy` and whether the source is **Pinned** vs **Pageable**.
- Optional: open CUDA API / runtime rows and look for `cudaMemcpyAsync` and `cudaStreamSynchronize`.

## Learning goal

Report 01 fixed **data loading** with workers, but **`data_movement` still dominated** (~67 ms) because pageable H2D forced a slow path (CPU staging + sync).  
This run adds **pinned memory + non-blocking `.to(device)`** so DMA can run from pinned host memory without that long wait.

---

## Four form questions (replace the existing ones)

### Q1. `data_movement` after pinning

On the main CPU timeline, what is the approximate duration of one steady `data_movement` range?

- A) ~60–70 ms (same as Report 01)  
- B) ~8–10 ms  
- C) ~0.3–0.5 ms  
- D) ~25–30 ms  

**Answer: C) ~0.3–0.5 ms**  
(Median ~0.39 ms. Report 01 was ~67 ms.)

---

### Q2. What changed on CUDA HW for H2D?

For the Host→Device copies in this run, which statement is correct?

- A) CUDA HW shows **Pageable** host memory as the H2D source (same as Report 01)  
- B) CUDA HW shows **Pinned** host memory as the H2D source  
- C) There is no Host→Device `memcpy` at all  
- D) All copies are Device→Device only  

**Answer: B)**  
With `pin_memory=True`, H2D uses **pinned** source memory (DMA-friendly). Report 01 used **pageable** source and paid for staging/sync on the CPU timeline.

---

### Q3. Longest stage now

In a steady iteration of Report 02, which NVTX stage is the **longest** on the main CPU timeline?

- A) data loading  
- B) data movement  
- C) forward pass  
- D) backward pass  

**Answer: D) backward pass**  
(~35 ms median. Forward ~25 ms. `data_movement` is now sub-millisecond; `data_loading` ~0.14 ms.)

---

### Q4. Did pin + non_blocking fix the bottleneck?

Compared with Report 01, what is the fairest conclusion?

- A) No change — `data_movement` is still ~67 ms  
- B) Yes for the transfer path — `data_movement` collapsed (tens of ms → &lt;1 ms) because pinned + async H2D avoids the pageable staging/sync wait; the step is now dominated by **forward/backward** compute  
- C) Only `optimizer_step` improved; transfers are unchanged  
- D) Workers disappeared; there are no `pt_data_worker` processes anymore  

**Answer: B)**  
Iteration wall time improves only modestly (~99 → ~94 ms here), but the **profile story** changes: transfer is no longer the long red bar. With `non_blocking=True`, Report 01’s long `cudaStreamSynchronize` waits during the copy path are gone; GPU compute (forward/backward) dominates.

---

## Quick answer key

| Q  | Answer |
|----|--------|
| Q1 | C — ~0.3–0.5 ms `data_movement` |
| Q2 | B — H2D from **Pinned** memory |
| Q3 | D — backward pass longest |
| Q4 | B — transfer fixed; step now compute-bound |

## Measured steady-state summary (for instructors)

| Stage (CPU NVTX) | Report 01 (workers, unpinned) | Report 02 (workers, pinned + non_blocking) |
|------------------|-------------------------------|--------------------------------------------|
| data_loading     | ~1.7 ms                       | ~0.14 ms                                   |
| data_movement    | ~67 ms                        | ~0.39 ms                                   |
| forward_step     | ~9.4 ms                       | ~25 ms                                     |
| backward_step    | ~16.7 ms                      | ~35 ms                                     |
| optimizer_step   | ~3.0 ms                       | ~3.1 ms                                    |
| full iteration   | ~99 ms                        | ~94 ms                                     |

**CUDA details worth showing live**

| Observation | Report 01 | Report 02 |
|-------------|-----------|-----------|
| H2D `srcKind` | Pageable | Pinned |
| `cudaMemcpyAsync` | present | present |
| `cudaStreamSynchronize` | many, ~33 ms mean | not in the copy path (no long sync spikes like Report 01) |

**Teaching note:** Forward/backward CPU NVTX bars look longer in Report 02 because the old transfer wait is gone and the timeline now reflects compute/sync around GPU work. Students should not conclude “forward got slower” without checking CUDA HW under the same window.
