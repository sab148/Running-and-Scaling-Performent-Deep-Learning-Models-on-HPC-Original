---
author: Sabrina Benassou, Javad Kasravi, Eske Ewert
title: Running Deep Learning Models on HPC Systems
#subtitle: A primer in supercomputers
date: September 21, 2026

---
## Important Links

Links for the skill-up

- [JuDOOR project page invitation](https://judoor.fz-juelich.de/login?show=/projects/join/training2638)
- [Workshop slides](
https://sab148.github.io/Running-and-Scaling-Performent-Deep-Learning-Models-on-HPC-Original/) 
- [Workshop code on GitHub](https://github.com/sab148/Running-and-Scaling-Performent-Deep-Learning-Models-on-HPC/tree/scicoco)
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
![Eske Ewert](pics/eske.jpg){width=240px}
::::
:::: {.col}
![Javad Kasravi](pics/javad.jpg){width=240px}
::::
:::

![](images/Logo_FZ_Juelich_rgb_Schutzzone_transparent.svg){height=120px}

---

### Schedule for the skill-up (Part 1 & 2)

| Time          | Title        |
| ------------- | -----------  |
| Mon, 14:30 - 15:00 | Use Jupyter-JSC to connect to JURECA  |
| Mon, 15:00 - 15:30 | How to run code on a compute node  |
| Mon, 15:30 - 16:00 | Train a cat classifier   |
| Tue, 10:15 - 11:00 | Single GPU training |
| Tue, 11:00 - 11:45 | Multiple GPU training using data parallel training |


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

![](images/supercomputer-queue.svg){height=400px}

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

- Go to the course project on JuDOOR: [https://judoor.fz-juelich.de/login?show=/projects/join/training2638](https://judoor.fz-juelich.de/login?show=/projects/join/training2638).
- Join the workshop's compute project `training2638`.
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
mkdir $PROJECT/dl-on-hpc-workshop/$USER
```

If you click on `$PROJECT/dl-on-hpc-workshop` on the left, you should see a folder with your name.

---

### Exercise: Create folders for the workshop
2. Create a shortcut (link) to your course folder in your home folder.

```bash
rm -rf ~/course
ln -s $PROJECT/dl-on-hpc-workshop/$USER ~/course
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

rm -rf $HOME/.cache
ln -s ~/course/.cache $HOME/
rm -rf $HOME/.config
ln -s ~/course/.config $HOME/
rm -rf $HOME/.fastai
ln -s ~/course/.fastai $HOME/
```

If you click *View > Show Hidden Files*, you should see the cache folders you created.

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
module load GCC OpenMPI Python PyTorch
# And we run a small test: import PyTorch and ask for its version.
python -c "import torch; print(torch.__version__)"
```

---


### Demo code
2. Create a new Python file in your course folder.

```bash
touch matrix.py
```

Open it in the editor.
Paste this into the file:

``` {.python .number-lines}
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
```
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

Paste the following content into the file:

```{.bash .number-lines style="font-size: 0.9em;"}
#!/bin/bash
#SBATCH --account=training2638          # Who pays?
#SBATCH --nodes=1                        # How many compute nodes
#SBATCH --job-name=matrix-multiplication
#SBATCH --ntasks-per-node=1              # How many MPI processes per node
#SBATCH --cpus-per-task=1                # How many CPUs per MPI process
#SBATCH --output=output.%j        # Where to write results
#SBATCH --error=error.%j
#SBATCH --time=00:01:00          # For how long can it run?
#SBATCH --partition=dc-gpu         # Machine partition
#SBATCH --reservation=scicoco_day1  # Reservation (for today only)

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
   412169 gpus         matrix-m  ewert4 CF       0:02      1 jsfc013

```

Close it with **Ctrl+C**.

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

Here, 412169 is the job ID. You can open the files in the editor.

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
uv venv --python 3.12 
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
source .venv/bin/activate
```

We will always use this virtual environment from now on!

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
#SBATCH --account=training2638
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
#SBATCH --reservation=scicoco_day1  # For today only

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

This downloads the dataset:
```python
path = untar_data(URLs.PETS)/'images'
```

This line downloads the pre-trained weights:
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

We already ran the code and have results.
To analyze them, there is a useful tool called TensorBoard.
We already have the code for it in our example!
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

## Day 1 recap

By now, I expect you have managed to:

- Stay awake for the most part of this afternoon 😴
- Connect to JURECA using Jupyter-JSC
- Edit files on JURECA
- Submit jobs and read results
- Be ready to write great code!

**Tomorrow**: More about training and parallelization.

Thank you for your attention! Any questions?
