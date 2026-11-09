# Nsight Systems — first-time guide (what to open, what to ignore)

You will see **many tabs and rows**. For this course exercise you only need a few of them.

**Goal:** measure training-step wall time on the CPU timeline, and check GPU activity / memory on CUDA HW.

---

## 0. Open the report

1. Start **Nsight Systems** (`nsys-ui`).
2. **File → Open**
3. Open only this file for the first exercise:

`Single_GPU/Report_00_zero_woker_unpined_non_blocking_False/nsys_logs/540_WVB_725_nsys_logs_rank.nsys-rep`

4. Wait until the report finishes loading.

---

## 1. Which top-level page/tab should you use?

Stay on:

### **Timeline** (sometimes called **Timeline View**)

This is the main horizontal time plot (processes, threads, CUDA HW, NVTX ranges).

### Ignore for now (do not use for the form answers)

| Tab / page | Why ignore now |
|---|---|
| Project / welcome pages | Only for opening files |
| Analysis Summary / Diagnostic Summary (if shown) | High-level overview, not precise step timing |
| Stats / SQL / expert reports | Useful later, not needed for this exercise |
| Other open reports / multi-report views | Use one report only |

**Rule:** if it is not the **Timeline**, you are probably in the wrong place for this exercise.

---

## 2. What should the Timeline look like? (rows that matter)

On the left side of the Timeline you see a tree of rows. Use this checklist:

### Step A — find the right process
1. Expand **Processes** (or the process list).
2. Find the **Python** process that has a child row named **CUDA HW**.
3. That is your training process.
4. Collapse / ignore other processes (shell, `srun`, unrelated ranks, etc.).

### Step B — expand only these rows
Under that Python process, keep visible:

1. **CPU / thread row(s)** for the main Python thread  
   - This is where **NVTX labels** appear:  
     `:data_loading`, `:data_movement`, `:forward_step`, `:backward_step`, `:optimizer_step`
2. **CUDA HW**  
   - Expand it if needed.  
   - Important sub-rows:
     - **Kernels** (GPU compute)
     - **Memory** / **Memcpy** / memory operations (H2D transfers)
     - **Memory usage** (for GPU peak memory)

### Step C — rows you can ignore
You can leave these collapsed:

- Many worker/OS helper threads with no NVTX training labels
- Long `poll` / OS runtime noise (unless asked)
- Other GPUs you are not using
- Network / NVLink / NCCL rows (not needed for Single-GPU Report_00)

---

## 3. How to navigate (self-explanatory controls)

| Action | How |
|---|---|
| Zoom in / out | `+` / `-` keys, or Ctrl + mouse wheel |
| Pan left / right | Scroll horizontally / drag the timeline |
| Jump to a useful region | Skip the very beginning (startup / first iterations) |
| Measure a duration | Click-drag on the timeline to select a range, then read the selected duration; or hover a colored NVTX bar |

**Tip:** start zoomed out, find repeating colored NVTX patterns, then zoom into **one middle iteration**.

---

## 4. Bottom panel: Events View (optional but helpful)

At the bottom you may see **Events View**.

Use it like this:

1. Right-click a Timeline row (for example **CUDA HW** or a thread with NVTX).
2. Choose **Show in Events View**.
3. The table lists events in the selected time window (kernels, memcpys, …) with durations.

Use Events View when:

- you need a more precise duration, or
- you want to confirm that a region is `memcpy` vs a big kernel.

You do **not** need Events View to answer every question; Timeline bars are enough for most.

---

## 5. Which view answers which question?

| Question type | Go here | Look at |
|---|---|---|
| One iteration wall time | **Timeline → CPU/Python thread** | From start of `:data_loading` to end of `:optimizer_step` |
| Longest training stage (CPU time) | **Timeline → CPU/Python thread** | Compare NVTX bar lengths |
| Data-loading duration | **Timeline → CPU/Python thread** | One `:data_loading` bar |
| Is work CPU or GPU? | **Same time window on Timeline → CUDA HW** | Kernels / short memcpy / idle |
| GPU peak memory | **Timeline → CUDA HW → Memory usage** | Read the peak value (hover the memory graph) |

### Important interpretation rule
- **NVTX label on CPU** = “the host entered this phase”  
- **CUDA HW under it** = “what the GPU was doing during that phase”

So yes: you can see `:forward_step` on the CPU row even though forward compute is on the GPU.

---

## 6. Mini walkthrough for Report_00 (do this once)

1. Open the `.nsys-rep` → stay on **Timeline**.
2. Find **Python + CUDA HW**.
3. Zoom to a steady middle iteration.
4. On the CPU thread, identify:
   - `:data_loading`
   - `:data_movement`
   - `:forward_step`
   - `:backward_step`
   - `:optimizer_step`
5. Measure one full iteration on the **CPU** timeline.
6. Compare CPU bar lengths → which is longest?
7. Click under `:data_movement` on **CUDA HW**: is it long kernels or a short memcpy?
8. Click under `:forward_step` on **CUDA HW**: do you see compute kernels?
9. Open **CUDA HW → Memory usage** and read peak GPU memory.
10. Fill the Google Form.

---

## 7. One-sentence student reminder

**Use Timeline only: measure times on the CPU NVTX bars; judge CPU vs GPU by looking at CUDA HW under the same window; read peak memory from CUDA HW → Memory usage.**

---

## Quick “wrong tab” symptoms

| If you see this… | You are probably wrong |
|---|---|
| Only summary tables, no horizontal time bars | Not on Timeline |
| Many processes, cannot find NVTX labels | Did not expand the Python process with CUDA HW |
| Trying to answer iteration time from CUDA kernels alone | Measuring GPU compute, not step wall time |
| Looking at NCCL / multi-GPU rows for Report_00 | Wrong section for this single-GPU exercise |
