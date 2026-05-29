---
author: Alexandre Strube // Sabrina Benassou // Ismail Khalfaoui
title: Bringing Deep Learning Workloads to JSC supercomputers
subtitle: Recap of Day 2
date: June 02, 2026
---

# Summary of Day 2

---

## Quick recap

The team discussed data loading strategies and storage approaches for supercomputers, including recommendations for code and data organization, along with demonstrations of different data loading methods and formats. Ismail presented on parallelization techniques using Slurm and distributed training with PyTorch, covering topics like transformer models, DDP implementation, and monitoring tools like LLView. The team addressed various technical challenges including PyTorch version compatibility issues, GPU utilization problems, and FSDP implementation, while also discussing different parallelism approaches for training large models.

---

## Next steps

Attendees: Store code in Project 1 of their respective projects
Attendees: Store data in Scratch for faster access and more memory
Attendees: Join DataOne datasets project for permanent data storage using the provided link
Attendees: Clone the provided GitHub repository to access code examples for data loading strategies
Attendees: Run the data loading examples using the provided SBATCH files to test HDF5 and Apache Arrow implementations
Attendees: Change the reservation in SBatch files from day one to day two to run the loader examples successfully
Attendees: Update the data path in their code to fix loading errors
Attendees: Access the pre-created ImageNet dataset files in the Scratch directory to avoid recreating them
Attendees: Check the pre-created HDF5 and Arrow files stored in the scratch/Data folder
Attendees: Use RAM for datasets smaller than 500 gigabytes
Attendees: Consider using HDF5, Apache Arrow, DALI, or SquashFS strategies for datasets larger than 500 gigabytes
Attendees: Consider using HDF5 for large datasets with many small files due to its better documentation and established use in science
Attendees: Review the GitHub repository code examples for data loading strategies
Attendees: Review the HDF5 and Apache Arrow examples provided in the repository under the "Code and Data Loading" folder
Attendees: Review the ImageNet loader examples for different data loading approaches
Summary

---

## Supercomputer Data Loading Best Practices

Sabrina presented on data loading strategies for supercomputers, covering best practices for storing code and data. She recommended storing code in Project 1 and data in Scratch (with a 90-day data retention policy) or DataOne datasets for permanent storage. Sabrina demonstrated different data loading approaches including RAM usage for datasets under 500GB, and explained strategies for larger datasets using HDF5 files and Apache Arrow, highlighting the importance of managing inode limitations in the file system.

---

## Data Formats and Parallelization Discussion

The team discussed data format options for their project, with Sabrina recommending HDF5 files as they are faster and more established than alternatives like PyArrow or NumPy files. Santiago and Scherer encountered issues running the SBatch scripts, with Scherer's problem being resolved by changing the reservation from day one to day two. Ismail was scheduled to present on parallelization using Slurm, building on Alexandre's previous discussion about setting up minimal examples of parallel computing.

---

## Transformer Training and Parallelization

Ismail led a discussion on training a transformer model using data from Wikitext2, focusing on parallelization techniques and the use of PyTorch and Hugging Face's datasets library. He explained the architecture of transformers and outlined the steps for setting up and running the code on the compute nodes, including the use of Slurm for job management and DDP for distributed training. Sabrina provided clarification on commenting out lines related to downloading datasets due to the lack of internet access on compute nodes, and Ismail emphasized the importance of running data-loading scripts on the login node before proceeding with training.

---

## Data Storage and Training Challenges

Ismail explained the different data storage formats like Parquet, row-based, and tree-based storage, noting that Parquet is preferred for deep learning tasks but less efficient for shuffling data. The team discussed issues with running code on the supercomputer, including GPU availability on login nodes and the need to use SBATCH for distributed training. Santiago encountered errors when running the distributed training code, which Ismail attributed to path problems and missing internet access on compute nodes, suggesting they check the error files for more details.

---

## LLView Monitoring Tool Demonstration

Ismail demonstrated the LLView monitoring tool for tracking job performance on the Jureca DC supercomputer. He showed how to access the tool, view real-time metrics including CPU and GPU usage, and generate PDF reports of job statistics including core hours used. The group discussed how to cancel jobs using the "scancel" command and observed that current GPU utilization was only at 23% despite requesting 4 GPUs, which Ismail explained was due to the code not properly distributing across all available GPUs.

---

## Neural Network Parallelization Techniques

Ismail explained the principles of parallelizing code for training neural networks using multiple GPUs, focusing on Distributed Data Parallel (DDP) approach. He discussed key concepts including communication operations like broadcast and reduce, terminology such as world size and global ranks, and the strategy of distributing data while copying the model to each GPU. Ismail emphasized that while DDP provides optimal performance when the model fits GPU memory, it comes with memory inefficiency challenges, which will be addressed in future discussions with an alternative parallelization scheme.

---

## Reproducibility in Distributed Deep Learning

Ismail explained the challenges of reproducibility in computer science and deep learning, particularly when using PyTorch's DDP (Distributed Data Parallel) for multi-GPU training. He detailed how to set up deterministic behavior in PyTorch, including proper seeding and configuration settings, to achieve higher reproducibility levels. The team then walked through the process of setting up distributed training, including configuring the master address, port selection, and using Torch Run with specific backend settings for their supercomputer environment. Ismail guided the implementation of distributed training features in the code, including importing distributed utilities, setting up process groups, creating distributed samplers, and modifying the model to use DDP with FSDP.

---

## Distributed Training Code Implementation

Ismail explained the implementation of print0 and save0 methods in distributed training to prevent repeated outputs and model weight overwrites. He demonstrated how to modify the code to use these methods and discussed the importance of shuffling data for training but not for validation. The team successfully ran the modified distributed training code on the supercomputer, with some members reporting around 80-86% GPU utilization. Ismail advised starting with one node for debugging before scaling up, and clarified that the local batch size of 128 is distributed across 4 GPUs, resulting in a global batch size of 512.

---

## FSDP Training Method Overview

Ismail explained the differences between DDP and FSDP (Fully Sharded Data Parallel) parallelization methods for training neural networks. He demonstrated how FSDP shards model layers across GPUs to save memory, making it suitable for larger models that don't fit in single GPU memory. The team discussed implementing FSDP in their code, including modifications needed for saving and printing model weights. Ismail advised that while FSDP allows training of models with billions of parameters, it requires sufficient GPU memory and high-bandwidth communication networks like InfiniBand to be effective.

---

## PyTorch Version Compatibility Issues

The team discussed issues with PyTorch version compatibility and errors related to FSDP (Fully Sharded Data Parallel) implementation. Ismail and Alexandre identified that the current environment uses PyTorch 2.5, but the exercises require version 2.6, leading to import errors. The team discovered that updating requirements.txt to version 2.6 and reinstalling the environment using setup.sh resolved the initial import issues. Alexandre provided a fix on Slack involving forced installation of NVIDIA CUDNN CU12, which appeared to address the remaining CUDA-related errors. The team confirmed that the updated environment allowed the code to enter the training loop without errors.

---

## Parallel Training Techniques for Large Models

Ismail explained different parallelism techniques for training large models, including model parallelism, pipeline parallelism, and tensor parallelism. He discussed when to use Fully Sharded Data Parallel (FSDP) versus these other approaches, noting that FSDP works well for models up to 512 GPUs but may suffer from communication issues beyond that scale. Tak shared that he needs to train a model that requires four H100 GPUs due to its size, with gradient tensors consuming the majority of memory. The session concluded with Alexandre announcing a survey for course participants and confirming the project deadline as September 30th.