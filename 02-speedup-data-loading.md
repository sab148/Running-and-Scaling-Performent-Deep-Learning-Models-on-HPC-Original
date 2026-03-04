---
author: Alexandre Strube // Sabrina Benassou 
title: Bringing Deep Learning Workloads to JSC supercomputers
subtitle: Data loading
date: September 16, 2025
---

### Schedule for day 2

| Time          | Title                |
| ------------- | -----------          |
| 13:00 - 13:10 | Welcome, questions   |
| 13:10 - 14:10 | Data loading |
| 14:10 - 14:25 | Coffee Break (flexible) |
| 14:25 - 17:00 | Parallelize Training |

---

## Let's talk about DATA

![](images/data.jpeg)

--- 

## I/O is separate and shared

- All compute nodes of all supercomputers see the same files
- Performance tradeoff between shared accessibility and speed
- Our I/O server is almost a supercomputer by itself
    ![JSC Supercomputer Stragegy](images/machines.png){height=350pt}

---

## Where do I keep my files?

- Always store your code in the project1 folder (**`$PROJECT_projectname`** ). In our case 

    ```bash
    /p/project1/training2609/$USER
    ```

- Store data in the scratch directory for faster I/O access (**`$SCRATCH_projectname`**). ⚠️**Files in scratch are deleted after 90 days of inactivity.**
    
    ```bash
    /p/scratch/training2609/$USER
    ```

- Store the data in [`$DATA_dataset`](https://judoor.fz-juelich.de/projects/datasets/) for a more permanent location. 

    ```bash
    /p/data1/datasets
    ```

---

## Data loading

- We have CPUs and lots of memory - let's use them
- If your dataset is relatively small (< 500 GB) and can fit into the working memory (RAM) of each compute node (along with the program state), you can store it in **``/dev/shm``**. This is a special filesystem that uses RAM for storage, making it extremely fast for data access. ⚡️
- For bigger datasets (> 500 GB), you have many strategies:
    - Hierarchical Data Format 5 (HDF5)
    - Apache Arrow
    - NVIDIA Data Loading Library (DALI)
    - SquashFS


---

## Inodes 
- Inodes (Index Nodes) are data structures that store metadata about files and directories.
- Unique identification of files and directories within the file system.
- Efficient management and retrieval of file metadata.
- Essential for file operations like opening, reading, and writing.
- **Limitations**:
  - **Fixed Number**: Limited number of inodes; no new files if exhausted, even with free disk space.
  - **Space Consumption**: Inodes consume disk space, balancing is needed for efficiency.
![](images/inodes.png)
