---
author: Sabrina Benassou, Ismael Khalfaoui 
title: Running Deep Learning Models on HPC System
#subtitle: A primer in supercomputers
date: October 14, 2026

---
## Important Links

Links for the skill-up

- [JuDOOR project page invitation](https://judoor.fz-juelich.de/projects/join/training2657)
- [Workshop slides](
https://sab148.github.io/Running-and-Scaling-Performent-Deep-Learning-Models-on-HPC-Original/) 
- [Workshop code on GitHub](https://github.com/sab148/Running-and-Scaling-Performent-Deep-Learning-Models-on-HPC/tree/inm_retreat_2026)
- [Jupyter-JSC](https://jupyter.jsc.fz-juelich.de)


Please open the slides on your device so you can copy and paste from the slides.

![](images/Logo_FZ_Juelich_rgb_Schutzzone_transparent.svg){height=180px}


---

## Goals for this course

- Access our machines in Jülich using Jupyter-JSC 👩‍💻
- Use the file system efficiently so that the supercomputer can access your data quickly 🏃
- Distribute your machine learning workload 💪


![](images/Logo_FZ_Juelich_rgb_Schutzzone_transparent.svg){height=180px}

---

## Team

::: {.container}
:::: {.col}
![Sabrina Benassou](pics/sabrina.jpg){width=240px}
::::
:::: {.col}
![Ismael Khalfaoui](pics/ismail.jpeg){width=240px}
::::
:::

![](images/Logo_FZ_Juelich_rgb_Schutzzone_transparent.svg){height=120px}

---

### Schedule 

| Time          | Title        |
| ------------- | -----------  |
| 11:00 - 11:15 | Introduction and Definitions |
| 11:15 - 11:30 | Use Jupyter-JSC to connect to the Supercomputer |
| 11:30 - 11:45 | How to run code on a compute node  |
| 11:45 - 12:15 | Single GPU training |
| 12:15 - 12:45 | Multiple GPU training using Distributed Data Parallel (DDP) |
| 12:45 - 13:00 | Wrap up and Q&A |


---

### Jülich Supercomputers

![JSC Supercomputer Strategy](images/machines.png){height=520px}

---

### What is a supercomputer?

- Compute cluster: Many computers bound together locally
- Supercomputer: A lot of computers bound together locally 😒
  - with a fancy network 🤯

---

### Anatomy of a supercomputer

- Login nodes: Normal machines for compilation, data transfer, scripting, etc. No GPUs. A limited number of shared CPU cores.
- Compute nodes: Guess what?
  - For compute! With GPUs! 🤩
- High-speed, ultra-low-latency network
- Shared networked file systems

---

### JURECA DC Compute Nodes

- 192 accelerated nodes (with GPUs)
- 2x AMD EPYC Rome 7742 CPUs at 2.25 GHz (128 cores/node)
- 512 GiB memory
- Network: Mellanox HDR InfiniBand (FAST💨 and EXPENSIVE💸)
- 4x NVIDIA A100 with 40 GB 😻
- TL;DR: 24576 cores, 768 GPUs 💪
- Way deeper technical info at [JURECA DC Overview](https://apps.fz-juelich.de/jsc/hps/jureca/configuration.html)

---

<!-- ### JUWELS Booster Compute Nodes

- 936 Nodes
- 2x AMD EPYC Rome 7402 CPUs at 2.7 GHz (48 cores x 2 threads = 96 virtual cores/node)
- 512 GiB memory
- Network: Mellanox HDR InfiniBand (FAST💨 and EXPENSIVE💸)
- 4x NVIDIA A100 with 40 GB 😻
- TL;DR: 89856 cores, 3744 GPUs, 468 TB RAM 💪
- Way deeper technical info at [JUWELS Booster Overview](https://apps.fz-juelich.de/jsc/hps/juwels/booster-overview.html)

--- -->

### You don't use the whole supercomputer

#### You submit jobs to a queue asking for resources

![](images/supercomputer-queue.svg){height=350px}

::: {.container}
:::: {.col}
- Your job(s) enter the queue and wait for their turn
- When there are enough resources for that job, it runs
::::
:::

---

### You don't use the whole supercomputer

#### And get results back

![](images/queue-finished.svg){height=500px}

---

### Supercomputer Usage Model

- Using the supercomputer means submitting a job to a batch system.
- No node-sharing. The smallest allocation for jobs is one compute node (4 GPUs).
- Maximum runtime of a job: 24 h.

---

### Recap

- Login nodes are for submitting jobs, downloading and moving files, compiling code, etc.
- NOT FOR TRAINING DEEP LEARNING MODELS!

---

#### Compute time

- Compute time allocation is based on compute projects. For every compute job, a compute project pays.
- Time is measured in core-hours. One hour of JURECA DC is 128 core-hours.
- Example: A job runs for 8 hours on 64 nodes of JURECA DC:
8 * 64 * 128 = 65536 core-h!

---

### Connecting to JURECA DC

#### Recall that you should have completed the following preparations already:

- Go to the course project on JuDOOR: [training2657](https://judoor.fz-juelich.de/projects/join/training2657).
- Join the workshop's compute project `training2657`.
- Sign the Usage Agreements ([video](https://drive.google.com/file/d/1mEN1GmWyGFp75uMIi4d6Tpek2NC_X8eY/view)).

---

## Connecting to JURECA DC via Jupyter-JSC

Go to https://jupyter.jsc.fz-juelich.de/hub/login and choose **Sign in with JSC account**.

![](images/login_02.png){height=350px}

---

## Connecting to JURECA DC via Jupyter-JSC

Log in with your JuDOOR account.

![](images/login_03.png){width=800px}

- Confirm that you allow the JSC Login service to use the information provided by the identity provider.

---

## Create a new JupyterLab

![](images/jupyter_01.png){width=1100px}

---

## Create a new JupyterLab

Please choose the following configuration. In particular, it is **important** that you choose the **login node**!

![](images/jupyter_02.png){height=380px}

Click **Open** once it is ready.

---

## Open a terminal on the login node

![](images/jupyter_03.png){height=450px}

---

## Open a terminal on the login node

![](images/jupyter_04.png){width=1050px}

---

### Exercise: Create folders for the workshop
Use the terminal in Jupyter-JSC for the following steps:

1. Create your course folder.

    ```bash
      mkdir $PROJECT/$USER
    ```

- If you click on `$PROJECT` on the left, you should see a folder with your name.

---

### Exercise: Create folders for the workshop
2. Create a shortcut (link) to your course folder in your home folder.

    ```bash
    rm -rf ~/course
    mkdir ~/course
    ln -s $PROJECT/$USER ~/course
    ```

3. Enter your course folder.

    ```bash
    cd ~/course
    ```

---

### Exercise: Create folders for the workshop
4. Link certain cache directories to the course folder. They should not be in `$HOME`, as it has limited space.

    ```bash
    mkdir ~/course/.cache
    mkdir ~/course/.config
    mkdir ~/course/.fastai
    mkdir ~/course/.local

    rm -rf $HOME/.cache
    ln -s ~/course/.cache $HOME/
    rm -rf $HOME/.config
    ln -s ~/course/.config $HOME/
    rm -rf $HOME/.fastai
    ln -s ~/course/.fastai $HOME/
    rm -rf $HOME/.local
    ln -s ~/course/.local $HOME/
    ```

- If you click *View > Show Hidden Files*, you should see the cache folders you created.

---

## Working with the supercomputer's software

- We have literally thousands of software packages, compiled specifically for the supercomputer.
- [Full list](https://www.fz-juelich.de/en/ias/jsc/services/user-support/using-systems/software)
- [Detailed documentation](https://apps.fz-juelich.de/jsc/hps/jureca/software-modules.html)

---

### Example: PyTorch

1. Copy and paste these lines:

    ```bash
    # This command fails, as we do not have the correct PyTorch environment.
    python -c "import torch; print(torch.__version__)"
    # Then load the correct modules.
    module load Stages/2025
    module load GCC OpenMPI Python 
    # And we run a small test: import PyTorch and ask for its version.
    python -c "import torch; print(torch.__version__)"
    ```

---


### Demo code
2. Create a new Python file in your course folder.

    ```bash
    touch matrix.py
    ```

- Open it in the editor and paste this into the file:

    ```python
    import torch

    matrix1 = torch.randn(3,3)
    print("The first matrix is:\n", matrix1)

    matrix2 = torch.randn(3,3)
    print("The second matrix is:\n", matrix2)

    result = torch.matmul(matrix1,matrix2)
    print("The result is:\n", result)
    ```

---

### How to run it on the login node
3. Run it on the login node.

    ```bash
      module load Stages/2025
      module load GCC OpenMPI Python PyTorch torchvision
      python matrix.py
    ```

---

### But that's not what we want... 😒

---

### So we send it to the queue!

---

## HOW?🤔

---

### Slurm 🤯

![](images/slurm.jpg){height=480px}

Simple Linux Utility for Resource Management

---

### Slurm submission file

- A simple text file that describes what we want, how much of it we need, how long we need it, and what to do with the results

4. Create a new file:

    ```bash
    touch jureca-matrix.sbatch
    ```

---

### Slurm submission file example

- Paste the following content into the file:

    ```bash
      #!/bin/bash
      #SBATCH --account=training2657          # Who pays?
      #SBATCH --nodes=1                        # How many compute nodes
      #SBATCH --job-name=matrix-multiplication
      #SBATCH --ntasks-per-node=1              # How many MPI processes per node
      #SBATCH --cpus-per-task=1                # How many CPUs per MPI process
      #SBATCH --output=output.%j        # Where to write results
      #SBATCH --error=error.%j
      #SBATCH --time=00:01:00          # For how long can it run?
      #SBATCH --partition=dc-gpu         # Machine partition

      module load Stages/2025
      module load GCC OpenMPI PyTorch torchvision # Load the correct modules on the compute node(s)

      srun python matrix.py            # srun tells the supercomputer how to run it
    ```

---

### Submitting a job: `sbatch`
5. Submit the Slurm job.

    ```bash
      sbatch jureca-matrix.sbatch

      Submitted batch job 412169
    ```

---

### Are we there yet?

![](images/are-we-there-yet.gif){height=500px}

--- 

### Are we there yet? 🐴

6. Check the status of the job using the following command:

    ```bash
    watch squeue --me
      JOBID  PARTITION    NAME      USER    ST       TIME  NODES NODELIST(REASON)
      412169 gpus         matrix-m  benassou1 CF       0:02      1 jrc0203

    ```

- Close it with **Ctrl+C**.

  `ST` is the status:

  - PD (pending)
  - CF (configuring)
  - R (running)
  - CG (completing)

---

### The job is wrong and needs to be cancelled

```bash
  scancel <JOBID>
```

---

### Check logs

7. By now, you should have output and error log files in your directory. Check them! For example:

    ```bash
    output.412169
    error.412169
    ```

- Here, 412169 is the job ID. You can open the files in the editor.

---

### Let's train a 🐈 classifier!

This is a minimal demo to show some quirks of the supercomputer.

We will train:

- ResNet-34, a convolutional neural network pretrained on image data, to classify images as cats or dogs
- Training data: the Oxford-IIIT Pet dataset, containing about 7,400 images of cats and dogs
- using the fastai library (high level training setup)

---

### Some preparations: Set up a virtual environment using uv
1. Load the modules we want to use:

    ```bash
    module load Stages/2025
    module load GCC OpenMPI Python PyTorch torchvision
    module load uv
    ```

2. Create a new virtual environment in your course folder

    ```bash
    uv venv ~/course/.venv --python 3.12 
    source ~/course/.venv/bin/activate
    ```

---

3. We need to install some additional packages. Create a file `touch requirements.txt` with the following content
    
    ```bash
      accelerate==1.1.1
      datasets==3.6.0
      fastai==2.8.12
      fsspec==2025.2.0
      ipykernel==7.3.0
      lightning==2.6.6
      matplotlib==3.9.2
      numba==0.60.0
      numpy==1.26.4
      pandas==2.2.2
      pyarrow==18.1.0
      scipy==1.13.1
      scikit-learn==1.5.2
      sentencepiece==0.2.0
      tensorboard==2.21.0
      torch==2.8
      torchrun-jsc==0.0.19
      transformers==4.46.3
      wandb==0.30.0
    ```

4. Install these by

    ```bash
      uv pip install -r requirements.txt
    ```

---

5. Finally, activate the virtual environment:
    
    ```bash
      source ~/course/.venv/bin/activate
    ```

- We will always use this virtual environment from now on!

---

## The cat classifier
1. Create a file in your course folder:

    ```
      touch cats.py
    ```
with the following content:

    ```python 
      from fastai.vision.all import *
      from fastai.callback.tensorboard import *
      #
      print("Downloading dataset...")
      path = untar_data(URLs.PETS)/'images'
      print("Finished downloading dataset")
      #
      def is_cat(x): return x[0].isupper()
      # Create the data loaders and resize the images
      dls = ImageDataLoaders.from_name_func(
          path, get_image_files(path), valid_pct=0.2, seed=42,
          label_func=is_cat, item_tfms=Resize(224))
      print("On the login node, this will download ResNet-34")
      learn = vision_learner(dls, resnet34, metrics=accuracy)
      cbs=[SaveModelCallback(), TensorBoardCallback('runs', trace_model=True)]
      # Trains the model for 6 epochs with this dataset
      learn.unfreeze()
      learn.fit_one_cycle(6, cbs=cbs)
    ```

---

### Submission file for the classifier

2. Create a Slurm file.
    
    ```bash
      touch fastai.sbatch
    ```

    ```{.bash style="font-size: 0.85em;"}
    #!/bin/bash
    #SBATCH --account=training2657
    #SBATCH --mail-user=MYUSER@fz-juelich.de
    #SBATCH --mail-type=ALL
    #SBATCH --nodes=1
    #SBATCH --job-name=cat-classifier
    #SBATCH --ntasks-per-node=1
    #SBATCH --cpus-per-task=128
    #SBATCH --output=output.%j
    #SBATCH --error=error.%j
    #SBATCH --time=00:20:00
    #SBATCH --partition=dc-gpu
    #SBATCH --reservation=inm_res  # For today only

    module load Stages/2025
    module load GCC OpenMPI PyTorch torchvision # Load the correct modules on the compute node(s)

    source ~/course/.venv/bin/activate # Activate the virtual environment

    srun python cats.py
    ```

--- 

### Submit it

3. Run the cat classifier.
    
    ```bash
      sbatch fastai.sbatch
    ```

---

### Submission time
4. Check the error and output logs and the queue.

---

Check the `error.${JOBID}` file

You will get an error. Why?

---

## 🤔...

---

### What is it doing?

- This downloads the dataset:
   
    ```python
      path = untar_data(URLs.PETS)/'images'
    ```

- This line downloads the pre-trained weights:
    
    ```python
      learn = vision_learner(dls, resnet34, metrics=accuracy)
    ```

---

## Compute nodes have no internet connection!

- But the login nodes do!
- So we download our dataset beforehand...
  - On the login nodes!

---


## On the login node

5. Comment out the line that performs the AI training:
    
    ```python
      # learn.fit_one_cycle(6, cbs=cbs)
    ```
Run the code on the login node!
    
    ```bash
      source .venv/bin/activate # So that we have the fastai library
      python cats.py
    ```

---

## Run the downloader on the login node

```{.bash style="font-size: 0.55em;"}
$ source .venv/bin/activate
$ python cats.py
Downloading dataset...
 |████████-------------------------------| 23.50% [190750720/811706944 00:08<00:26]
 Downloading: "https://download.pytorch.org/models/resnet34-b627a593.pth" to /p/project/ccstao/cstao05/.cache/torch/hub/checkpoints/resnet34-b627a593.pth
100%|█████████████████████████████████████| 83.3M/83.3M [00:00<00:00, 266MB/s]
```

---

## Run it again on the compute nodes!

6. Uncomment the line that performs the training:
    
    ```python
      learn.fit_one_cycle(6, cbs=cbs)
    ```
Submit the job again!
    
    ```bash
      sbatch fastai.sbatch
    ```

---

## 7. Check output files

```{.bash style="font-size: 0.9em;"}
The activation script must be sourced, otherwise the virtual environment will not work.
Setting vars
Downloading dataset...
Finished downloading dataset
epoch     train_loss  valid_loss  error_rate  time    
Epoch 1/1 : |-----------------------------------| 0.00% [0/92 00:00<?]
Epoch 1/1 : |-----------------------------------| 2.17% [2/92 00:14<10:35 1.7452]
Epoch 1/1 : |█----------------------------------| 3.26% [3/92 00:14<07:01 1.6413]
Epoch 1/1 : |██---------------------------------| 5.43% [5/92 00:15<04:36 1.6057]
...
....
Epoch 1/1 :
epoch     train_loss  valid_loss  error_rate  time    
0         0.049855    0.021369    0.007442    00:42     
```

- 🎉
- 🥳

---

### Tools for results analysis

- We already ran the code and have results.
- To analyze them, there is a useful tool called TensorBoard.
- We already have the code for it in our example!
    
    ```python
      cbs=[SaveModelCallback(), TensorBoardCallback('runs', trace_model=True)]
    ```

---

## TensorBoard
- In the terminal enter:
    
    ```bash
      tensorboard   --logdir=runs   --port=6000   --host=127.0.0.1
    ```

- Then open a jupyter notebook (File -> New Notebook), select Python 3 (ipykernel) and execute:
    
    ```bash
      import os
      from IPython.display import IFrame

      url = f"{os.environ['JUPYTERHUB_SERVICE_PREFIX']}proxy/6000/"

      IFrame(url, width="100%", height=800)
    ```

![](images/notebook.png){}



---

### TensorBoard

![](images/tensorboard_jupyter.png){height=500px}

---

## Now, Let's talk about DATA

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
    /p/project1/training2657/$USER
    ```

- Store data in the scratch directory for faster I/O access (**`$SCRATCH_projectname`**). ⚠️**Files in scratch are deleted after 90 days of inactivity.**
    
    ```bash
    /p/scratch/training2657/$USER
    ```

- Store the data in [`$DATA_dataset`](https://judoor.fz-juelich.de/projects/datasets/) for a more permanent location. 

    ```bash
    /p/data1/datasets
    ```


---

## Now lets talk about Parallelization technique

---

## Before Starting

- If you have not done it already, clone the following repo:

    ```bash
    git clone --branch inm_retreat_2026 https://github.com/sab148/Running-and-Scaling-Performent-Deep-Learning-Models-on-HPC.git
    ```

---

## What this code does 

- It trains a [Transformer](https://arxiv.org/pdf/1706.03762) language model on [WikiText-2](https://huggingface.co/datasets/mindchain/wikitext2) dataset to predict the next word in a sequence. 
- **Transformers** is a deep learning model architecture that uses self-attention to process sequences in parallel.
- **WikiText-2** is a word-level language modeling dataset consisting of over 2 million tokens extracted from high-quality Wikipedia articles. 

---

## What this code does 

- Again, this is not a deep learning course.
- If you are not familiar with the model and the dataset, just imagine it as a black box: you provide it with text, and it generates another text.

    ![](images/black_box.svg)

---

## Libraries

- You already downloaded the libraries that we will use today:
    - **PyTorch:** A deep learning framework for building and training models.
    - **datasets** A library from Hugging Face used to access, load, process, and share large datasets.
    
---

Let's have a look at the files **```train/Lit_training.py```** and **```Lit_training.sbatch```** in the repo.

![](images/look.jpg)

---

## Run the Training Script

- There are TODOs in these two files. **Do not modify the TODOs for now**. The code is already working, so you don’t need to make any changes at this point.
- Now run:

    ```bash
    sbatch Lit_training.sbatch
    ```

- Spoiler alert 🚨

- The code won't work.

- Check the output and error files

---

## What is the problem?

- Remember, there is no internet on the compute node.
- Therefore, you should:
    - **Comment out** lines 22 **to** 60.
    - Activate your environment:

        ```bash
        module load Stages/2025
        module load GCC uv
        source $HOME/course/.venv/bin/activate
        ```

    - Run:

        ```bash
        export PYTHONPATH="${PYTHONPATH}:$(pwd)"
        python train/Lit_training.py
        ```

    - **Uncomment back** lines 22-60.
    - Finally, run your job again 🚀:

        ```bash
        sbatch Lit_training.sbatch
        ```

---

## JOB Running

- Congrats, you are training a DL model on the supercomputer using one GPU 🎉

--- 

## llview

- You can monitor your training using [llview](https://go.fzj.de/llview-jureca). 
- Use your Judoor credentials to connect.
- Check the job number that you are intrested in.
    ![](images/llview_job.png){height=400px}

---


## llview

- Go to the right to open the PDF document. **It may take some time to load the job information, so please wait until the icon turns blue**.
    ![](images/llview.png){height=450px}

---

## llview

- You have many information about your job once you open the PDF file.
    ![](images/llview_info.png)

---

## GPU utilization 

- You can see that in fact we are using **1 GPU**

    ![](images/llview_gpu_1.png)

---

## GPU utilization   

- It is a waste of resources.

- The training takes time (1h32m according to llview).

- Then, can we run our model on multiple GPUs ?

---

## llview

- We are still using **1 GPU**

![](images/llview_gpu_2.png)

---

## We need communication

- Without correct setup, the GPUs might not be utilized.

- Furthermore, we don't have an established communication between the GPUs

    ![](images/dist/no_comm.svg){height=400px}

---

## We need communication

![](images/dist/comm1.svg){height=500px}

---

## We need communication

![](images/dist/comm2.svg){height=500px}

---

## collective operations

- The GPUs use collective operations to communicate and share data in parallel computing
- The most common collective operations are: All Reduce, All Gather, and Reduce Scatter

---

## All Reduce 

![](images/dist/all_reduce.svg)

- Other operations, such as **min**, **max**, and **avg**, can also be performed using All-Reduce.

---

## All Gather

![](images/dist/all_gather.svg)

--- 

## Reduce Scatter

![](images/dist/reduce_scatter.svg)

--- 

## Terminologies

- Before going further, we need to learn some terminologies

---

## World Size

![](images/dist/gpus.svg){height=550px}

---

## Rank

![](images/dist/rank.svg){height=550px}

---

## local_rank

![](images/dist/local_rank.svg){height=550px}

---

## Now

That we have understood how the devices communicate and the terminologies used in parallel computing, 
we can move on to distributed training (training on multiple GPUs).

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

## Let's start coding!

- Whenever you see **TODOs**💻📝, follow the instructions to either copy-paste the code at the specified line numbers or type it yourself.

- Depending on how you copy and paste, the line numbers may vary, but always refer to the TODO numbers in the code and slides.

---

## Pass the number of nodes

**TODO 1**💻📝: At line 55 pass to num_nodes the number of nodes 

```python 
  num_nodes=int(os.getenv("SLURM_NNODES", "1"))
```

---

## We are almost there

- Before launching our job, we need to add some lines to **Lit_training.sbatch** file 

---

## Setup communication

In **```Lit_training.sbatch```** file:

- **TODOs 2**💻📝: 
    - At line 3, increase the number of GPUs to 4:

        ```bash
        #SBATCH --gres=gpu:4
        ```

- **TODOs 3**💻📝: 
    - At line 4, set the number of tasks per node to 4.
        
        ```bash
          #SBATCH --ntasks-per-node=4
        ```

- **TODOs 4**💻📝: 
    - At line 20, pass the correct number of devices.

        ```bash
        # Set up four visible GPUs that the job can use 
        export CUDA_VISIBLE_DEVICES=0,1,2,3
        ```



---

## Setup communication

Stay in **```Lit_training.sbatch```** file:

- **TODO 5**💻📝: we need to setup **MASTER_ADDR** and **MASTER_PORT** to allow communication over the system.

    - At line 22, add the following:

        ```bash
          # Extracts the first hostname from the list of allocated nodes to use as the master address.
          export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"

          # Modifies the master address to allow communication over InfiniBand cells.
          if [ "$SYSTEMNAME" = juwelsbooster ] \
                || [ "$SYSTEMNAME" = juwels ] \
                || [ "$SYSTEMNAME" = jurecadc ] \
                || [ "$SYSTEMNAME" = jusuf ]; then
              # Allow communication over InfiniBand cells on JSC machines.
              MASTER_ADDR="$MASTER_ADDR"i
          fi

          export MASTER_PORT=7010
        ```

---

## done ✅

- You can finally run:

    ```bash
    sbatch Lit_training.sbatch
    ```

---

## llview

- Let's have a look at our job using [llview](https://go.fzj.de/llview-jureca) again.

- You can see that now, we are using all the GPUs of the node

- ![](images/llview_gpu_4.png)

--- 

## llview

- And that our job took less time to finish training (25m vs 1h32m with one GPU)

- But what about using more nodes ?

---

## What about using more nodes ?

---

## Multi-node training

- **TODO 6**💻📝: in **```Lit_training.sbatch```** at line 2, you can increase the number of nodes to 2:

    ```bash
    #SBATCH --nodes=2
    ```

- Hence, you will use 8 GPUs for training.

- Run again:

    ```bash
    sbatch Lit_training.sbatch
    ```

--- 

## llview

- Open [llview](https://go.fzj.de/llview-jureca) again.

- You can see that now, we are using 2 nodes and 8 GPUs.

- ![](images/llview_gpu_8.png)

- And the training took less time (14m)

---

## Amazing ✨

---

## RECAP 

By now, I expect you have managed to:

- Stay awake for the most part of this tutorial 😴
- Connect to JURECA using Jupyter-JSC
- Edit files, submit jobs and read results on JURECA
- You know where to store your code and your data. 🗂️📄
- You know what distributed training is. 🧑‍💻
- You can submit training jobs on a single GPU, multiple GPUs, or across multiple nodes. 🎮💻
- You know how to monitor your training using llview. 📊👀

---

## Find Out More

- Here are some useful:

    - Tutorials:
        - [PyTorch at JSC](https://sdlaml.pages.jsc.fz-juelich.de/ai/recipes/pytorch_at_jsc/)
        - [PyTorch tutorials GitHub](https://github.com/pytorch/tutorials/tree/main)
        - [PyTorch documentation](https://pytorch.org/tutorials/distributed/home.html)

    - Links
        - [AI Landing Page](https://sdlaml.pages.jsc.fz-juelich.de/ai/)
        - [Other courses at JSC](https://www.fz-juelich.de/en/ias/jsc/education/training-courses)


---

## ANY QUESTIONS??

#### Feedback is more than welcome!




