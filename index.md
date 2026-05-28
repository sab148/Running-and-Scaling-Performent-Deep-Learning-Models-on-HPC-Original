---
author: Sabrina Benassou
title: Deep Learning on Supercomputers
#subtitle: A primer in supercomputers
date: May 27, 2026

---

![](images/nn/dl.png)

Class 0: Dog

Class 1: Cat

---

## Neural Network

![](images/nn/intro_jsc.003.svg){height=1000px}

---

## What is Training 

- Learn patterns by updating neural network parameters using 4 iterative steps:
    1. Forward pass
    2. Calculate Loss
    3. Backward Pass
    4. Parameter Update
    
---

## Forward pass

![](images/nn/intro_jsc.007.svg){height=600px}

---

## Forward pass

![](images/nn/intro_jsc.008.svg){height=600px}

---

## Forward pass

![](images/nn/intro_jsc.009.svg){height=600px}

---

## Forward pass

![](images/nn/intro_jsc.014.svg){height=600px}

---

## Forward pass

![](images/nn/intro_jsc.015.svg){height=600px}

---

## Forward pass

![](images/nn/intro_jsc.016.svg){height=600px}

---

## Forward pass

![](images/nn/intro_jsc.017.svg){height=600px}

---

## forward pass

![](images/nn/intro_jsc.018.svg){height=600px}

---

## forward pass

![](images/nn/intro_jsc.019.svg){height=600px}

---

## Loss function

![](images/nn/intro_jsc.024.svg){height=600px}

---

## backward pass

![](images/nn/intro_jsc.025.svg){height=600px}

---

## backward pass

![](images/nn/intro_jsc.026.svg){height=600px}

---

## backward pass

![](images/nn/intro_jsc.027.svg){height=600px}

---

## backward pass

![](images/nn/intro_jsc.028.svg){height=600px}

---

## update weights

![](images/nn/intro_jsc.030.svg){height=600px}

---

## Iterations & Epochs

- **Iteration**: one full training step using a single subset (batch) of data.
    
    forward pass → loss computation → backward pass → parameter update

- **Epoch**: one complete pass through the entire dataset.

---

## Dataset Concepts

* **Training set:** data used to train the model and update weights.
* **Validation set:** data used during training to tune hyperparameters and monitor performance.
* **Test set:** unseen data used only for the final evaluation of the model.

---

## From Neural Networks to Deep Learning

- Adding multiple hidden layers produces a deep neural network.
- Each layer learns increasingly abstract representations — edges → textures → objects.
- A 2-layer net recognises simple patterns; a 50-layer model recognises complex objects.
- Depth unlocks capabilities that shallow networks can't approximate efficiently.

--- 

## Deep Learning Models Types

<div style="display:flex; gap:20px; justify-content:center; align-items:flex-start;">

<figure style="text-align:center;">
  <img src="images/nn/cnn.svg" height="300px" width="600px">
  <figcaption>Convolutional Neural Network</figcaption>
</figure>

<figure style="text-align:center;">
  <img src="images/nn/attention.svg" height="300px" width="300px">
  <figcaption>Transformer</figcaption>
</figure>

<figure style="text-align:center;">
  <img src="images/nn/sdm.svg" height="300px" width="600px">
  <figcaption>Diffusion Model</figcaption>
</figure>

</div>

--- 

## The Scaling Law

- **Model Capacity**: Larger models with more parameters are required to absorb massive amounts of data without saturating.
- **Data Capacity**: More data allows these larger models to learn richer, more diverse patterns without overfitting.
- **Compute Demand**: Scaling both models and datasets demands significantly more computational power and memory to train.

**Bigger Models + More Data + More Compute = Better Performance**

---

## Deep Learning, High-Performance Computing & Distributed Training
- GPUs enable thousands of parallel operations, making deep learning far faster than CPUs.
- But modern AI models exceed the capacity of a single GPU, requiring training to be distributed across multiple GPUs and nodes, significantly enhancing training speed and model accuracy.
- **High-Performance Computing (HPC)** provides the infrastructure — high-speed communication, large-scale storage — that enables this scaling.

---

## Distributed Data Parallel (DDP)

[DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) is a method in parallel computing used to train deep learning models across multiple GPUs or nodes efficiently.

![](images/ddp/ddp-2.svg){height=400px}

--- 

## DDP

![](images/ddp/ddp-3.svg){height=500px}

--- 

## DDP

![](images/ddp/ddp-4.svg){height=500px}

--- 

## DDP

![](images/ddp/ddp-5.svg){height=500px}

--- 

## DDP

![](images/ddp/ddp-6.svg){height=500px}

--- 

## DDP

![](images/ddp/ddp-7.svg){height=500px}

--- 

## DDP

![](images/ddp/ddp-8.svg){height=500px}

--- 

## DDP

![](images/ddp/ddp-9.svg){height=500px}

--- 

## DDP

If you're scaling DDP to use multiple nodes, the underlying principle remains the same as single-node multi-GPU training.

---

## DDP

![](images/ddp/multi_node.svg){height=500px}

---

## DDP recap

- Each GPU on each node gets its own process.
- Each GPU has a copy of the model.
- Each GPU has visibility into a subset of the overall dataset and will only see that subset.
- Each process performs a full forward and backward pass in parallel and calculates its gradients.
- The gradients are synchronized and averaged across all processes.
- Each process updates its optimizer.

---

## Demo

- Let's train a deep learning model on the supercomputer.

---

## Setup
#### Make sure you are connected to the supercomputer

```bash
# Create a folder for myself
mkdir $PROJECT_training2623/$USER

# Create a shortcut for the project on the home folder
rm -rf ~/course ; ln -s $PROJECT_training2623/$USER ~/course

# Enter course folder and
cd ~/course

# Where am I?
pwd

# We well need those later
mkdir ~/course/.cache
mkdir ~/course/.config
mkdir ~/course/.fastai

rm -rf $HOME/.cache ; ln -s ~/course/.cache $HOME/
rm -rf $HOME/.config ; ln -s ~/course/.config $HOME/
rm -rf $HOME/.fastai ; ln -s ~/course/.fastai $HOME/
```
---

## Setup

```bash
cd $PROJECT_training2623/$USER
git clone -b intro_jsc https://github.com/sab148/Running-and-Scaling-Performent-Deep-Learning-Models-on-HPC.git
cd Running-and-Scaling-Performent-Deep-Learning-Models-on-HPC/
```

---

## setup

```bash
source install_venv.sh
sbatch Lit_training.sbatch
watch squeue --me
```

---

## Libraries

- We will use today:
    - **PyTorch Lightning:** A deep learning framework for building and training models.
    - **datasets** A library from Hugging Face used to access, load, process, and share large datasets.
    
---

## What this code does 

- It trains a [Transformer](https://arxiv.org/pdf/1706.03762) language model on [WikiText-2](https://huggingface.co/datasets/mindchain/wikitext2) dataset to predict the next word in a sequence. 
- **Transformers** is a deep learning model architecture that uses self-attention to process sequences in parallel.
- **WikiText-2** is a word-level language modeling dataset consisting of over 2 million tokens extracted from high-quality Wikipedia articles. 

---

## What this code does 

- If you are not familiar with the model and the dataset, just imagine it as a black box: you provide it with text, and it generates another text.

    ![](images/black_box.svg)

---

## Let's have a look at the code

---

## THAT’S ALL FOLKS!

- Thanks for listening!
- Questions?

---