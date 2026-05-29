---
author: Alexandre Strube // Sabrina Benassou // Ismail Khalfaoui
title: Bringing Deep Learning Workloads to JSC supercomputers
subtitle: Recap of Day 1
date: June 01, 2026
---

https://indico3-jsc.fz-juelich.de/event/240/

# Summary of Day 1

---

##  Quick recap

Alexandre led a comprehensive meeting to introduce a course on AI and supercomputing at JSC, covering course objectives, supercomputer architecture, and the Jureca DC supercomputer specifications. The team worked through practical sessions on setting up SSH connections and Visual Studio Code for remote development, with Alexandre providing step-by-step guidance and troubleshooting support for participants. The conversation ended with demonstrations on managing software modules, running code on supercomputers, and submitting batch jobs, along with discussions about accessing and using supercomputing resources through various systems.

---

##  Next steps

All participants: Ensure they can access the supercomputer by the end of the day.
All participants: Open the provided webpage link for accessing course materials.
All participants: Join the course Slack channel for communications and updates.
Hannah: Complete Slack registration process.
All participants: Set up their SSH keys and Visual Studio Code configuration for connecting to the supercomputer.
Peter, Mawaki, and Michelle: Wait for their supercomputer access to be granted.
Narimene: Present the first part of the course tomorrow.
Ismail: Present with Narimene tomorrow about data handling on the supercomputer.
Sabrina and Ismail: Present on data preparation tomorrow.
All participants: Prepare their data for proper use on the supercomputer.
All participants: Learn how to properly upload and organize their data on the supercomputer.
All participants: Prepare for tomorrow's session on making ML distributed across multiple GPUs and nodes.
All participants: Prepare their code to utilize all 4 GPUs on a node to avoid wasting resources.
All participants: Review basic AI concepts if needed, possibly using Fast AI resources.
All participants: Review the documentation link provided for more details about the Jureca DC supercomputer.
All participants: Utilize their allocated project hours within the next 2 weeks.
All participants: Contact Alexandre if they need more project time beyond the 2-week allocation.
All participants: Complete the survey at the end of the course.
Summary

---

##  AI and Supercomputing Course Introduction

Alexandre led a meeting to introduce a course on AI and supercomputing at JSC. He explained the course objectives, including ensuring all participants reach the same level of understanding and using interactive sessions to address questions. The team discussed accessing necessary materials, including a shared webpage with links to training resources, GitLab, and slides. Alexandre provided an overview of supercomputers, highlighting their architecture, including login nodes, compute nodes, and the importance of high-speed networks. He also introduced the Jureca DC supercomputer, detailing its specifications and capabilities. The conversation ended with a brief explanation of batch processing and Jupyter Notebooks, setting the stage for future discussions on using supercomputers for AI tasks.

---

##  Supercomputer Usage Basics Training

Alexandre explained the basics of using a supercomputer, emphasizing that jobs enter a queue and resources are shared among users. He detailed the process of connecting to the login node, submitting jobs, and using the system's Jupyter environment for small-scale work. Alexandre also covered the importance of using exclusive nodes and the 24-hour job limit, and he provided instructions for connecting to the supercomputer using Visual Studio Code with SecureShell.

---

##  SSH Configuration for Supercomputers Setup

Alexandre guided participants through setting up SSH configurations for connecting to supercomputers, addressing specific issues for Windows, Linux, and Mac users. He provided step-by-step instructions for creating and configuring SSH keys, managing IP restrictions, and uploading public keys to the Jureca system. Participants, including Salika and Mathi, received assistance with troubleshooting errors and connecting to the remote systems. The session focused on ensuring everyone could successfully connect to the supercomputers using Visual Studio Code.

---

##  SSH Setup in Jureca Supercomputer

Alexandre guided participants through setting up SSH connections to the Jureca supercomputer using Visual Studio Code. He helped Scherer successfully connect by configuring the SSH extension and accessing remote files. Tak encountered permission issues with private key files stored on a local drive, which Alexandre helped resolve by changing file permissions and copying keys to the user directory. Somayeh was able to connect via SSH but had issues with the SSH extension not displaying properly in Visual Studio Code, which Alexandre addressed by troubleshooting the configuration settings.

---

##  SSH and VS Code Setup

Alexandre guided participants through setting up SSH access and Visual Studio Code for remote development on a supercomputer. He helped Somayeh and Mathi resolve issues with accessing and configuring the SSH keys, including correcting file paths and installing the necessary VS Code extensions. The session concluded with participants successfully accessing the correct working directory and opening terminals in VS Code, with Alexandre confirming that everyone was set up properly for the next steps.

---

##  Supercomputer Software Management Demonstration

Alexandre demonstrated how software is managed on the supercomputer, explaining the module system which allows multiple versions of software to be available without overwhelming the system. He showed how to search for and load specific software modules using the module spider command, and explained the hierarchy of modules including compiler, MPI, and application levels. Alexandre also demonstrated how to copy files to the supercomputer using the file browser interface, allowing users to work directly with files stored on the system.

---

##  Supercomputer Matrix Multiplication Training Session

Alexandre led a session on running matrix multiplication code on a supercomputer, explaining the difference between login nodes and compute nodes, and the importance of using compute nodes with GPUs for machine learning tasks. He demonstrated how to submit a job using Slurm, including creating a batch script and submitting it with sbatch, and showed how to monitor jobs using the squeue command. Alexandre also explained the concept of reservations, which allows a group to reserve specific resources on the supercomputer for a limited time, and in this case, secured 20 nodes for 3 hours to run the matrix multiplication code.

---

##  Supercomputer Code Execution Demonstration

Alexandre demonstrated how to run code on a supercomputer, including setting up files, installing software using a requirements file, and creating a virtual environment. He explained the difference between login nodes (which have internet access) and compute nodes (which don't), and showed how to download a dataset for a simple cats vs dogs classification example. When attempting to run the full training script, they encountered an internet connectivity issue on the compute node, so Alexandre modified the approach to first download the dataset on the login node before running the complete script.

---

##  Supercomputer Batch Processing Demonstration

Alexandre demonstrated how to submit batch files to the supercomputer, explaining the process of using sbatch and monitoring jobs through the command line. He showed how to use TensorBoard to visualize training progress and explained the concept of port forwarding to access web services on the supercomputer. Alexandre also introduced Blahblador, an AI chatbot service available for free to help with coding and questions about the system.

---

##  Supercomputing Resource Access Guidelines

The meeting focused on discussing access and usage of supercomputing resources, particularly the Jülich cluster. Alexandre explained how to submit multiple jobs using job arrays in Slurm and recommended using JUBE for organizing runs. He described the simple process for accessing compute time through helmodes.ai, which doesn't require a lengthy proposal process. The discussion also covered software installation, with Alexandre advising that users can install their own software when practical, though complex installations are handled by the team. Finally, Alexandre demonstrated how to access graphical applications like ParaView and Blender on the cluster using remote desktop functionality.