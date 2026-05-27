---
author: Sabrina Benassou
title: Deep Learning on Supercomputers
#subtitle: A primer in supercomputers
date: May 27, 2026

---

## Neural Network

![](images/nn/intro_jsc.003.svg){height=1000px}

---

## What is Training 

- Learn patterns by updating neural network parameters to minimize the loss function using an **optimizer algorithm**.

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

## Optimizer

- Common optimization strategies:
    - **SGD (Stochastic Gradient Descent)**: Updates parameters opposite to the gradient using a learning rate. Efficient but may converge slowly and oscillate in complex loss landscapes
    - **SGD with Momentum**: Improves SGD by adding part of the previous update to the current one. Helps faster convergence and smoother optimization
    - **RMSProp**: Adapts the learning rate for each parameter using recent gradient magnitudes. Enables faster and more stable learning
    - **Adam (Adaptive Moment Estimation)**: Combines Momentum and RMSProp. Fast, stable, and widely used in deep learning

---

## update weights

![](images/nn/intro_jsc.030.svg){height=600px}

---

## From Neural Networks to Deep Learning

- Adding multiple hidden layers produces a deep neural network.
- Each layer learns increasingly abstract representations — edges → textures → objects.
- A 2-layer net recognises simple patterns; a 50-layer model recognises complex objects.
- Depth unlocks capabilities that shallow networks can't approximate efficiently.

--- 

## Why Deep Learning Succeeded

- **Large datasets** — ImageNet's 1.2M+ labelled images gave models enough examples to generalise.
- **Powerful GPUs** — ~1000× speedup over CPUs for matrix operations made training feasible in days, not years.
- **Better algorithms** — ReLU, Dropout, and Batch Normalisation solved vanishing gradients, overfitting, and slow convergence that blocked earlier progress.

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
  <figcaption>Stable Diffusion Model</figcaption>
</figure>

</div>

--- 

## Applications

![](images/nn/dl_tasks.svg){height=600px}

--- 

## Scaling Laws 
<div style="text-align: left">
-> More data helps models learn richer and more diverse patterns.

-> Larger datasets usually require larger models with more parameters.

-> Larger models require significantly more compute power and memory.
</div>
**Bigger models + more data + more compute = better performance**

---

## Deep Learning & High-Performance Computing
- GPUs allows thousands of operations to run in parallel, making DL dramatically faster than on a CPU.
- However, a single GPU is no longer enough. Training a model with billions of parameters demands far more memory and compute than any one device can provide. To scale efficiently, training is distributed across multiple GPUs, multi-node clusters, and supercomputers.
- High-Performance Computing (HPC) provides the infrastructure — fast interconnects, large storage, and job orchestration — that makes this possible.

---

## Distributed Training

- Parallelize the training across multiple nodes, 
- Significantly enhancing training speed and model accuracy.
- It is particularly beneficial for large models and computationally intensive tasks, such as deep learning.[[1]](https://pytorch.org/tutorials/distributed/home.html)


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

## Let's code it

---

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

```bash
cd $PROJECT_training2623/$USER
git clone -b intro_jsc https://github.com/sab148/Running-and-Scaling-Performent-Deep-Learning-Models-on-HPC.git
cd Running-and-Scaling-Performent-Deep-Learning-Models-on-HPC/
source install_venv.sh
sbatch Lit_training.sbatch
```

---

## Let's have a look at the code

---

## THAT’S ALL FOLKS!

- Thanks for listening!
- Questions?

---