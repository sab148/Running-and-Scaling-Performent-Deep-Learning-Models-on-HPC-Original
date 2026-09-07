---
author: Alexandre Strube // Ismail Khalfaoui-Hassani
title: Accessing the machines, intro
#subtitle: A primer in supercomputers
date: September 8, 2026

---
## Communication:

Links for the complimentary parts of this course: 

- [Zoom](https://go.fzj.de/running-and-scaling-deep-learning-models-on-HPC-zoom)
- [Slack](https://go.fzj.de/running-and-scaling-deep-learning-models-on-HPC-slack)
- [JSC Training Page](https://go.fzj.de/running-and-scaling-deep-learning-models-on-HPC-training-page)
- [Judoor project page invite](https://judoor.fz-juelich.de/projects/join/training2643)
- [This document: https://go.fzj.de/running-and-scaling-deep-learning-models-on-HPC](https://go.fzj.de/running-and-scaling-deep-learning-models-on-HPC)
- Our mailing list for [AI news](https://lists.fz-juelich.de/postorius/lists/ai.jsc.lists.fz-juelich.de/)
- [Survey at the end of the course](https://go.fzj.de/bringing-dl-workloads-to-jsc-survey)
- [Virtual Environment template](https://gitlab.jsc.fz-juelich.de/kesselheim1/sc_venv_template)
- [SOURCE of the course/slides on Github](https://github.com/sab148/Running-and-Scaling-Performent-Deep-Learning-Models-on-HPC-Original)

![](images/Logo_FZ_Juelich_rgb_Schutzzone_transparent.svg)


---

## Goals for this course:

- Make sure you know how to access and use our machines 👩‍💻
- Put your data in way that supercomputer can use it fast 🏃
- Distribute your ML workload 💪


![](images/Logo_FZ_Juelich_rgb_Schutzzone_transparent.svg)

---

## Team:

::: {.container}
:::: {.col}
![Alexandre Strube](pics/alex.jpg)
::::
:::: {.col}
![Sabrina Benassou](pics/sabrina.jpg)
::::
:::: {.col}
![Ismail Khalfaoui-Hassani](pics/ismail.jpeg)
::::
:::: {.col}
![Javad Kasravi](pics/javad.jpg)
::::
:::

![](images/Logo_FZ_Juelich_rgb_Schutzzone_transparent.svg)

---

### Schedule for day 1

| Time          | Title        |
| ------------- | -----------  |
| 13:00 - 13:15 | Welcome      |
| 13:15 - 14:00 | Introduction |
| 14:00 - 14:15 | Coffee break |
| 14:16 - 14:30 | Judoor, Keys |
| 14:30 - 15:00 | SSH, VS Code |
| 15:00 - 15:15 | Coffee Break |
| 15:15 - 16:00 | Running services on the login and compute nodes | 
| 16:00 - 16:15 | Coffee Break |
| 16:30 - 17:00 | Sync (everyone should be at the same point) |

---

### Note

Please open this document on your own browser! We will need it for the exercises.
[https://sab148.github.io/Running-and-Scaling-Performent-Deep-Learning-Models-on-HPC-Original/#/title-slide](https://sab148.github.io/Running-and-Scaling-Performent-Deep-Learning-Models-on-HPC-Original/#/title-slide)

![Mobile friendly, but you need it on your computer, really](images/Running-and-Scaling-Performent-Deep-Learning-Models-on-HPC.png)

---

### Jülich Supercomputers

![JSC Supercomputer Strategy](images/machines.png)

---

### What is a supercomputer?

- Compute cluster: Many computers bound together locally 
- Supercomputer: A lot of computers bound together locally 😒
  - with a fancy network 🤯

---

### Anatomy of a supercomputer

-  Login Nodes: Normal machines, for compilation, data transfer, scripting, etc. No GPUs. Limited number of shared CPU cores.
- Compute Nodes: Guess what? 
  - For compute! With GPUs! 🤩
- High-speed, ultra-low-latency network
- Shared networked file systems
- Some numbers we should (more or less) know about them:
    - Nodes
    - Cores, Single-core Performance
    - RAM
    - Network: Bandwidth, Latency
    - Accelerators (e.g. GPUs)
      - GPU MEMORY

---

### JURECA DC Compute Nodes

- 192 Accelerated Nodes (with GPUs)
- 2x AMD EPYC Rome 7742 CPU 2.25 GHz (128 cores/node)
- 512 GiB memory
- Network Mellanox HDR infiniband (FAST💨 and EXPENSIVE💸)
- 4x NVIDIA A100 with 40gb 😻
- TL;DR: 24576 cores, 768 GPUs 💪
- Way deeper technical info at [Jureca DC Overview](https://apps.fz-juelich.de/jsc/hps/jureca/configuration.html)

---

<!-- ### JUWELS Booster Compute Nodes

- 936 Nodes
- 2x AMD EPYC Rome 7402 CPU 2.7 GHz (48 cores x 2 threads = 96 virtual cores/node)
- 512 GiB memory
- Network Mellanox HDR infiniband (FAST💨 and EXPENSIVE💸)
- 4x NVIDIA A100 with 40gb 😻
- TL;DR: 89856 cores, 3744 GPUs, 468 TB RAM 💪
- Way deeper technical info at [Juwels Booster Overview](https://apps.fz-juelich.de/jsc/hps/juwels/booster-overview.html)

--- -->

## How do I use a Supercomputer?

- Batch: For heavy compute, ML training
- Interactively: Jupyter

---

### You don't use the whole supercomputer

#### You submit jobs to a queue asking for resources

![](images/supercomputer-queue.svg)

---

### You don't use the whole supercomputer

#### And get results back

![](images/supercomputer-queue-2.svg)

---

### You don't use the whole supercomputer

#### You are just submitting jobs via the login node

![](images/supercomputer-queue-3.svg)

---

### You don't use the whole supercomputer

#### You are just submitting jobs via the login node

![](images/supercomputer-queue-4.svg)

---

### You don't use the whole supercomputer

#### You are just submitting jobs via the login node

![](images/supercomputer-queue-5.svg)

---

### You don't use the whole supercomputer



::: {.container}
:::: {.col}
- Your job(s) enter the queue, and wait for its turn
- When there are enough resources for that job, it runs
::::
:::: {.col}
![](images/midjourney-queue.png)
::::
:::

![]()

---

### You don't use the whole supercomputer

#### And get results back

![](images/queue-finished.svg)

---

### Supercomputer Usage Model
- Using the the supercomputer means submitting a job to a batch system.
- No node-sharing. The smallest allocation for jobs is one compute node (4 GPUs).
- Maximum runtime of a job: 24h.

---

### Recap:

- Login nodes are for submitting jobs, download and move files, compile, etc
- NOT FOR TRAINING NEURAL NETS!

---

### Recap:

- User submit jobs
- Job enters the queue
- When it can, it runs
- Sends results back to user

---

### Connecting to Jureca DC

#### Getting compute time
- Go to [https://judoor.fz-juelich.de/projects/join/training2643](https://judoor.fz-juelich.de/projects/join/training2643)
- Join the course project `training2643`
- Sign the Usage Agreements ([Video](https://drive.google.com/file/d/1mEN1GmWyGFp75uMIi4d6Tpek2NC_X8eY/view))
- Compute time allocation is based on compute projects. For every compute job, a compute project pays.
- Time is measured in core-hours. One hour of Jureca DC is 128 core-hours.
- Example: Job runs for 8 hours on 64 nodes of Jureca DC: 8 * 64 * 128 = 65536 core-h!

---

<!-- ## Jupyter

[jupyter-jsc.fz-juelich.de](https://jupyter-jsc.fz-juelich.de)

- Jupyter-JSC uses the queue 
- When you are working on it, you are using project time ⌛️
- *Yes, if you are just thinking and looking at the 📺, you are burning project time*🤦‍♂️
- *Yes, if you are just thinking and looking at the 📺, you are slowing down the queue for everyone*🤦
- It's useful for small tests - not for full-fledged development 🙄

--- -->

<!-- ## Jupyter

#### Pay attention to the partition - DON'T RUN IT ON THE LOGIN NODE!!!

![](images/jupyter-partition.png)

---

## Connecting to Jureca DC

---

## VSCode

- [Download VScode: code.visualstudio.com](https://code.visualstudio.com/download)
- Install and run it
  - On the local terminal, type `code`
- Install [Remote Development Tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack)
- Install [Remote: SSH](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh)
- If you have Windows, you need WSL as explained on the email.

--- -->

## VSCode

### Now with the remote explorer tab
![](images/vscode-welcome.png)


---

#### SSH
- SSH is a secure shell (terminal) connection to another computer
- You connect from your computer to the LOGIN NODE
- Security is given by public/private keys
- A connection to the supercomputer needs a 
    1. Key,
    2. Configuration
    3. Key/IP address known to the supercomputer

---

### SSH

#### Create key in VSCode's Terminal (menu View->Terminal)

```bash
mkdir ~/.ssh/
ssh-keygen -a 100 -t ed25519 -f ~/.ssh/id_ed25519-JSC
```

```bash
$ ssh-keygen -a 100 -t ed25519 -f ~/.ssh/id_ed25519-JSC
Generating public/private ed25519 key pair.
Enter passphrase (empty for no passphrase): 
Enter same passphrase again: 
Your identification has been saved in /Users/khalfaoui1/.ssh/id_ed25519-JSC
Your public key has been saved in /Users/khalfaoui1/.ssh/id_ed25519-JSC.pub
The key fingerprint is:
SHA256:EGNNC1NTaN8fHwpfuZRPa50qXHmGcQjxp0JuU0ZA86U khalfaoui1@homepc
The keys randomart image is:
+--[ED25519 256]--+
|      *++oo=o. . |
|     . =+o .= o  |
|      .... o.E..o|
|       .  +.+o+B.|
|        S  =o.o+B|
|          . o*.B+|
|          . . =  |
|           o .   |
|            .    |
+----[SHA256]-----+
```

---

### SSH

#### Configure SSH session

```bash
code $HOME/.ssh/config
```

Windows users, from Ubuntu WSL
(Change username for your user on windows)

```bash
ls -la /mnt/c/Users/
mkdir /mnt/c/Users/USERNAME/.ssh/
cp $HOME/.ssh/* /mnt/c/Users/USERNAME/.ssh/
```


---

### SSH

#### Configure SSH session

```bash
Host jureca
        HostName jureca.fz-juelich.de
        User [MY_USERNAME]   # Here goes your username, not the word MY_USERNAME.
        AddressFamily inet
        IdentityFile ~/.ssh/id_ed25519-JSC
        MACs hmac-sha2-512-etm@openssh.com
```

Copy contents to the config file and save it 

**REPLACE [MY_USERNAME] WITH YOUR USERNAME!**

---

### SSH

####  JSC restricts from where you can login
#### So we need to:
1. Find our ip range
2. Add the range and key to [Judoor](https://judoor.fz-juelich.de)

---

### SSH

#### Find your ip/name range

Open **[https://www.whatismyip.com](https://www.whatismyip.com)**

---

### SSH

#### Find your ip/name range

![](images/whatismyip.png)

- Let's keep this inside vscode: `code key.txt` and paste the number you got

---

### SSH

Did everyone get their **own** ip address?

---

### SSH - EXAMPLE

- I will use the number `93.199.55.163`
- **YOUR NUMBER IS DIFFERENT**


---

### SSH - Example: `93.199.55.163`

- Go to VSCode and make it simpler, replace the 2nd half with `"0.0/16"`:
  - It was `93.199.55.163`
  - Becomes `93.199.0.0/16` (with YOUR number, not with the example)
- Add a `from=""` around it
- So, it looks like this, now: `from="93.199.0.0/16"`
- Add a second magic number, with a comma: `,10.0.0.0/8` 🧙‍♀️
- I promise, the magic is worth it 🧝‍♂️ (If time allows)
- In the end it looks like this: `from="93.199.0.0/16,10.0.0.0/8"` 🎬
- Keep it open, we will use it later
- If you are from FZJ, also add "134.94.0.0/16" with a comma

---

### SSH - Example: `93.199.0.0/16`

#### Copy your ssh key
- Terminal: `code ~/.ssh/id_ed25519-JSC.pub`
- Something like this will open:

- `ssh-ed25519 AAAAC3NzaC1lZDE1NTA4AAAAIHaoOJF3gqXd7CV6wncoob0DL2OJNfvjgnHLKEniHV6F khalfaoui@demonstration.fz-juelich.de`

- Paste this line at the same `key.txt` which you just opened

---

### SSH

#### Example: `93.199.0.0/16`

- Put them together and copy again:
- `from="93.199.0.0/16,10.0.0.0/8" ssh-ed25519 AAAAC3NzaC1lZDE1NTA4AAAAIHaoOJF3gqXd7CV6wncoob0DL2OJNfvjgnHLKEniHV6F khalfaoui@demonstration.fz-juelich.de`

---

### SSH

- Let's add it on [Judoor](https://judoor.fz-juelich.de)
- ![](images/manage-ssh-keys.png)
- Do it for JURECA and JUDAC with the same key

---

### SSH

#### Add new key to [Judoor](https://judoor.fz-juelich.de)

![](images/manage-ssh-keys-from-and-key.png){ width=850px }

This might take some minutes

---

### SSH: Exercise

That's it! Give it a try (and answer yes)

```bash
$ ssh jureca
The authenticity of host 'jrlogin03.fz-juelich.de (134.94.0.185)' cannot be established.
ED25519 key fingerprint is SHA256:ASeu9MJbkFx3kL1FWrysz6+paaznGenChgEkUW8nRQU.
This key is not known by any other names
Are you sure you want to continue connecting (yes/no/[fingerprint])? Yes
**************************************************************************
*                            Welcome to Jureca DC                   *
**************************************************************************
...
...
khalfaoui1@jrlogin03~ $ 
```

---

### SSH: Exercise 
#### Make sure you are connected to the supercomputer

```bash
# Create a folder for myself
mkdir $PROJECT_training2643/$USER

# Create a shortcut for the project on the home folder
rm -rf ~/course ; ln -s $PROJECT_training2643/$USER ~/course

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

## Working with the supercomputer's software

- We have literally thousands of software packages, hand-compiled for the specifics of the supercomputer.
- [Full list](https://www.fz-juelich.de/en/ias/jsc/services/user-support/using-systems/software)
- [Detailed documentation](https://apps.fz-juelich.de/jsc/hps/jureca/software-modules.html)

---

## Software

#### Tool for finding software: `module spider`

```bash
khalfaoui1$ module spider PyTorch
------------------------------------------------------------------------------------
  PyTorch:
------------------------------------------------------------------------------------
    Description:
      Tensors and Dynamic neural networks in Python with strong GPU acceleration. 
      PyTorch is a deep learning framework that puts Python first.

     Versions:
        PyTorch/1.7.0-Python-3.8.5
        PyTorch/1.8.1-Python-3.8.5
        PyTorch/1.11-CUDA-11.5
        PyTorch/1.12.0-CUDA-11.7
     Other possible modules matches:
        PyTorch-Geometric  PyTorch-Lightning
...
```

---

## What do we have?

`module avail` (Inside hierarchy)

---

## Module hierarchy

- Stage (full collection of software of a given year)
- Compiler
- MPI
- Module

- Eg: `module load Stages/2025 GCC OpenMPI PyTorch`

---

#### What do I need to load such software?

`module spider Software/version`

---

## Example: PyTorch

Search for the software itself - it will suggest a version

![](images/module-spider-1.png)

---

## Example: PyTorch

Search with the version - it will suggest the hierarchy

![](images/module-spider-2.png)

---

## Example: PyTorch

(make sure you are still connected to Jureca DC)

```bash
$ python
Python 3.12.3 (main, Sep 7 2026, 00:00:00) 
[GCC 11.5.0 20240719 (Red Hat 11.5.0-11)] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ModuleNotFoundError: No module named 'torch'
```

Oh noes! 🙈

Let's bring Python together with PyTorch!

---

## Example: PyTorch

Copy and paste these lines
```bash
# This command fails, as we have no proper pytorch
python -c "import torch ; print(torch.__version__)" 
# So, we load the correct modules...
module load Stages/2025
module load GCC OpenMPI Python PyTorch
# And we run a small test: import pytorch and ask its version
python -c "import torch ; print(torch.__version__)" 
```

Should look like this:
```bash
$ python -c "import torch ; print(torch.__version__)" 
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'torch'
$ module load Stages/2025
$ module load GCC OpenMPI Python PyTorch
$ python -c "import torch ; print(torch.__version__)" 
2.1.2
```
---

## Python Modules

#### Some of the python softwares are part of Python itself, or of other softwares. Use "`module key`"

```bash
module key toml
The following modules match your search criteria: "toml"
------------------------------------------------------------------------------------

  Jupyter: Jupyter/2020.2.5-Python-3.8.5, Jupyter/2021.3.1-Python-3.8.5,
    Jupyter/2021.3.2-Python-3.8.5, Jupyter/2022.3.3, Jupyter/2022.3.4
    Project Jupyter exists to develop open-source software, open-standards,
    and services for interactive computing across dozens of programming languages.
    

  PyQuil: PyQuil/3.0.1
    PyQuil is a library for generating and executing Quil programs on the Rigetti
    Forest platform.

  Python: Python/3.8.5, Python/3.9.6, Python/3.10.4
    Python is a programming language that lets you work more quickly and integrate 
    your systems more effectively.

------------------------------------------------------------------------------------
```
---

## VSCode
#### Editing files on the supercomputers

![](images/vscode-remotes.png)

---

## VSCode

![](images/vscode-jusuf.png)

---

## VSCode

- You can have a terminal inside VSCode: 
  - Go to the menu View->Terminal

--- 

## VSCode

From the VSCode's terminal, navigate to your "course" folder and to the name you created earlier.

```bash
cd $HOME/course/
pwd
```

- This is out working directory. We do everything here.

---

### Demo code
#### Create a new file "`matrix.py`" on VSCode on Jureca DC

```bash
code matrix.py
```

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

```
module load Stages/2025 
module load GCC OpenMPI Python PyTorch 
python matrix.py
```

---

### But that's not what we want... 😒

---

### So we send it to the queue!

---

## HOW?🤔

---

### SLURM 🤯
![](images/slurm.jpg)

Simple Linux Utility for Resource Management

---

### Slurm submission file

- Simple text file which describes what we want and how much of it, for how long, and what to do with the results

---

### Slurm submission file example

`code jureca-matrix.sbatch`

``` {.bash .number-lines}
#!/bin/bash
#SBATCH --account=training2643           # Who pays?
#SBATCH --nodes=1                        # How many compute nodes
#SBATCH --job-name=matrix-multiplication
#SBATCH --ntasks-per-node=1              # How many mpi processes/node
#SBATCH --cpus-per-task=1                # How many cpus per mpi proc
#SBATCH --output=output.%j        # Where to write results
#SBATCH --error=error.%j
#SBATCH --time=00:01:00          # For how long can it run?
#SBATCH --partition=dc-gpu         # Machine partition
#SBATCH --reservation=RSPDLM_Day1  # For today only

module load Stages/2025
module load GCC OpenMPI PyTorch  # Load the correct modules on the compute node(s)

srun python matrix.py            # srun tells the supercomputer how to run it
```

---

### Submitting a job: SBATCH

```bash
sbatch jureca-matrix.sbatch

Submitted batch job 412169
```

---

### Are we there yet?

![](images/are-we-there-yet.gif)

--- 

### Are we there yet? 🐴

`squeue --me`

```bash
squeue --me
   JOBID  PARTITION    NAME      USER    ST       TIME  NODES NODELIST(REASON)
   412169 gpus         matrix-m  khalfaoui1 CF       0:02      1 jsfc013

```

#### ST is status:

- PD (pending), 
- CF(configuring), 
- R (running),   
- CG (completing)

---

### Reservations

- Some partitions have reservations, which means that only certain users can use them at certain times.
- For this course, it's called `training2643`

--- 

### Job is wrong, need to cancel

```bash
scancel <JOBID>
```

---

### Check logs

#### By now you should have output and error log files on your directory. Check them!

```bash
# Notice that this number is the job id. It's different for every job
cat output.412169 
cat error.412169 
```

Or simply open it on VSCode!

---

## Extra software, modules and kernels

#### You want some extra Python software from PyPI...

We will use [`uv`](https://docs.astral.sh/uv/) to create a virtual environment and install our Python dependencies.

* The environment will live in `$HOME/course/.venv`
* Install packages on the **login node**
* The same environment can then be used from the compute nodes

---

## Install `uv`

Check whether `uv` is already available:

```bash
uv --version
```

If it is not installed, install it into your home directory:

```bash
pip install uv
```

Check that it works:

```bash
uv --version
```

---

## Create a virtual environment

Go to the course directory:

```bash
cd $HOME/course/
```

Load the Python environment we want to use:

```bash
module load Stages/2025
module load GCC OpenMPI Python
```

Now create a virtual environment using that Python:

```bash
uv venv --python 3.12 
```

You should see something similar to:

```text
Using CPython ...
Creating virtual environment at: .venv
Activate with: source .venv/bin/activate
```

---

## Example: Let's install some software!

Even though the supercomputer provides a lot of software, sometimes we need additional Python packages.

For example:

* fast.ai
* Weights & Biases
* Transformers
* Lightning
* 🤗 Datasets

We will install them into our `.venv`.

---

## Create `requirements.txt`

From `$HOME/course/`:

```bash
code requirements.txt
```

Add:

```text
ipykernel
fastai
numba==0.60.0
numpy==1.26.4
scipy==1.13.1
matplotlib==3.9.2
scikit-learn==1.5.2
pandas==2.2.2
accelerate==1.1.1
pyarrow==18.1.0
transformers==4.46.3
sentencepiece==0.2.0
datasets==3.6.0
fsspec==2025.2.0.*
torch==2.8
torchrun_jsc>=0.0.15
wandb
tensorboard
lightning
```

Notice that we do **not** need to add `pip`.

`uv` will install the packages for us.

---

## Install the dependencies with `uv`

Make sure you are still in the course directory:

```bash
cd $HOME/course/
```

Then run:

```bash
uv pip install -r requirements.txt
```

Because `.venv` exists in this directory, `uv` automatically installs the packages into:

```text
$HOME/course/.venv
```

We don't even need to activate the environment to install packages. ✨

Check the installation:

```bash
uv pip check
```

---

## Activating the virtual environment

When working interactively, activate it with:

```bash
cd $HOME/course/
source .venv/bin/activate
```

Check which Python you are using:

```bash
which python
```

It should point to something like:

```text
.../course/.venv/bin/python
```

Now test some packages:

```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import fastai; print('fastai:', fastai.__version__)"
python -c "import wandb; print('wandb:', wandb.__version__)"
```

🎉

To leave the environment:

```bash
deactivate
```

---

## Adding more software later

Want another Python package?

For example:

```bash
cd $HOME/course/
uv pip install rich
```

---

## Create a Jupyter kernel

Our environment already contains `ipykernel`.

Activate it:

```bash
cd $HOME/course/
source .venv/bin/activate
```

Then register it as a Jupyter kernel:

```bash
python -m ipykernel install --user \
    --name training2643-uv \
    --display-name "Python (training2643 / uv)"
```

You can now select:

```text
Python (training2643 / uv)
```

as the kernel in Jupyter.

---

## Using the environment in VS Code

You do not have to create a separate environment for VS Code.

Open the course directory:

```bash
cd $HOME/course/
code .
```

Then select the Python interpreter:

```text
$HOME/course/.venv/bin/python
```

VS Code will use the packages installed in our `uv` environment.

---

## Using the environment in a Slurm job

The virtual environment lives on the shared filesystem, so we do **not** reinstall anything inside the job.

Load the same modules:

```bash
module load Stages/2025
module load GCC OpenMPI Python
```

Then run Python directly from the virtual environment:

```bash
cd $HOME/course/

srun .venv/bin/python cats.py
```

There is no need to activate the environment inside the batch script.

For example:

```bash
#!/bin/bash

#SBATCH --account=training2643
#SBATCH --nodes=1
#SBATCH --job-name=cat-classifier
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --output=output.%j
#SBATCH --error=error.%j
#SBATCH --time=00:20:00
#SBATCH --partition=dc-gpu
#SBATCH --reservation=RSPDLM_Day1

module load Stages/2025
module load GCC OpenMPI Python

cd $HOME/course/

source .venv/bin/activate

srun cats.py
```

That's it: one `.venv`, managed with `uv`, usable interactively, from VS Code/Jupyter, and inside Slurm jobs. 🚀

---

## Important: install packages on the login node

The compute nodes do not have Internet access.

Therefore, commands such as:

```bash
uv pip install -r requirements.txt
```

or:

```bash
uv pip install some-package
```

must be run on the **login node**.

Once the packages are installed in `$HOME/course/.venv`, the same environment can be used from the compute nodes because the course directory is on the shared filesystem.

---

## Recap

Create the environment:

```bash
cd $HOME/course/

module load Stages/2025
module load GCC OpenMPI Python

uv venv --python "3.12" .venv
uv pip install -r requirements.txt
```

Use it interactively:

```bash
source .venv/bin/activate
python cats.py
```

Use it in a Slurm job:

```bash
srun $HOME/course/.venv/bin/python cats.py
```

No custom virtual-environment scripts needed. 🎉

---

### Let's train a 🐈 classifier!

This is a minimal demo, to show some quirks of the supercomputer
```
code cats.py
```

---

```python 
from fastai.vision.all import *
from fastai.callback.tensorboard import *
#
print("Downloading dataset...")
path = untar_data(URLs.PETS)/'images'
print("Finished downloading dataset")
#
def is_cat(x): return x[0].isupper()
# Create the dataloaders and resize the images
dls = ImageDataLoaders.from_name_func(
    path, get_image_files(path), valid_pct=0.2, seed=42,
    label_func=is_cat, item_tfms=Resize(224))
print("On the login node, this will download resnet34")
learn = vision_learner(dls, resnet34, metrics=accuracy)
cbs=[SaveModelCallback(), TensorBoardCallback('runs', trace_model=True)]
# Trains the model for 6 epochs with this dataset
learn.unfreeze()
learn.fit_one_cycle(6, cbs=cbs)
```

---

### Submission file for the classifier

```bash
code fastai.sbatch
```

```bash
#!/bin/bash
#SBATCH --account=training2643
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
#SBATCH --reservation=RSPDLM_Day1  # For today only

cd $HOME/course/
source .venv/bin/activate # Now we finally use the fastai module

srun python cats.py
```

--- 

### Submit it

```bash
sbatch fastai.sbatch
```

---

### Submission time

- Check error and output logs, check queue

---

### Probably not much happening...

```bash
$ cat output.7948496 
The activation script must be sourced, otherwise the virtual environment will not work.
Setting vars
Downloading dataset...
```

---

### But it might fail

```bash
$ cat err.7948496 
The following modules were not unloaded:
  (Use "module --force purge" to unload all):

  1) Stages/2025
```

---

### 💥

---

### What happened?

It might be that it's not enough time for the job to give up

Check the `error.${JOBID}` file

If you run it longer, you will get the actual error:

```python
Traceback (most recent call last):
  File "/p/project/training2643/khalfaoui1/cats.py", line 5, in <module>
    path = untar_data(URLs.PETS)/'images'
    ...
    ...
    raise URLError(err)
urllib.error.URLError: <urlopen error [Errno 110] Connection timed out>
srun: error: jwb0160: task 0: Exited with exit code 1
```

---

## 🤔...

---

### What is it doing?

This downloads the dataset:
```python
path = untar_data(URLs.PETS)/'images'
```

And this one downloads the pre-trained weights:
```python
learn = vision_learner(dls, resnet34, metrics=accuracy)
```

---


## Remember, remember

![](images/queue-finished.svg)

---

## Remember, remember

![](images/compute-nodes-no-net.svg)

---

## Compute nodes have no internet connection

- But the login nodes do!
- So we download our dataset before...
  - On the login nodes!

---


## On the login node:

Comment out the line which does AI training:
```python
# learn.fit_one_cycle(6, cbs=cbs)
```
Call our code on the login node!
```bash
source .venv/bin/activate # So that we have fast.ai library
python cats.py
```

---

## Run the downloader on the login node

```bash
$ source .venv/bin/activate
$ python cats.py 
Downloading dataset...
 |████████-------------------------------| 23.50% [190750720/811706944 00:08<00:26]
 Downloading: "https://download.pytorch.org/models/resnet34-b627a593.pth" to /p/project/ccstao/cstao05/.cache/torch/hub/checkpoints/resnet34-b627a593.pth
100%|█████████████████████████████████████| 83.3M/83.3M [00:00<00:00, 266MB/s]
```

---

## Run it again on the compute nodes!

Un-comment back the line that does training:
```python
learn.fit_one_cycle(6, cbs=cbs)
```
Submit the job!
```bash
sbatch fastai.sbatch
```

---

## Waiting for the job to run?

```bash
watch squeue --me
```
(To exit, type CTRL-C)

---

## Check output files

You can see them within VSCode
```bash
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

We already ran the code and have results
To analyze them, there's a neat tool called Tensorboard
And we already have the code for it on our example!
```python
cbs=[SaveModelCallback(), TensorBoardCallback('runs', trace_model=True)]
```

---

## Example: Tensorboard

The command 
```bash
tensorboard --logdir=runs  --port=9999 serve
```
- Opens a connection on port 9999... *OF THE SUPERCOMPUTER*.
- This port is behind the firewall. You can't access it directly... 
- We need to bypass the firewall 🏴‍☠️
  - SSH PORT FORWARDING

---

## Example: Tensorboard

![](images/supercomputer-firewall.svg)

---

## Port Forwarding

![A tunnel which exposes the supercomputer's port 3000 as port 1234 locally](images/port-forwarding.svg)


---

## Port forwarding demo:

On VSCode's terminal:
```bash
cd $HOME/course/
source .venv/bin/activate
tensorboard --logdir=runs  --port=12345 serve
```
- Note the tab `PORTS` next to the terminal 
- On the browser: [http://localhost:12345](http://localhost:12345)

---

### Tensorboard on Jureca DC

![](images/tensorboard-cats.png)


---

## Day 1 recap

As of now, I expect you managed to: 

- Stay awake for the most part of this morning 😴
- Have your own ssh keys 🗝️🔐
- A working ssh connection to the supercomputers 🖥️
- Can edit and transfer files via VSCode 📝
- Submit jobs and read results 📫
- Access web services on the login nodes 🧙‍♀️
- Is ready to make great code! 💪

---

## ANY QUESTIONS??

#### Feedback is more than welcome!

---

## Backup slides

---

## There's more!

- Remember the magic? 🧙‍♂️
- Let's use it now to access the compute nodes directly!

---

## Proxy Jump

#### Accessing compute nodes directly

- If we need to access some ports on the compute nodes
- ![](images/proxyjump-magic.svg)

---

## Proxy Jump - SSH Configuration

Type on your machine "`code $HOME/.ssh/config`" and paste this at the end:

```ssh

# -- Compute Nodes --
Host *.jureca
        User [ADD YOUR USERNAME HERE]
        StrictHostKeyChecking no
        IdentityFile ~/.ssh/id_ed25519-JSC
        ProxyJump jureca
```        

---

## Proxy Jump: Connecting to a node

Example: A service provides web interface on port 9999

On the supercomputer:

```bash
srun --time=00:05:00 \
     --nodes=1 --ntasks=1 \
     --partition=dc-gpu \
     --account training2643 \
     --cpu_bind=none \
     --pty /bin/bash -i

bash-4.4$ hostname # This is running on a compute node of the supercomputer
jwb0002

bash-4.4$ cd $HOME/course/
bash-4.4$ source sc_venv_template/activate.sh
bash-4.4$ tensorboard --logdir=runs  --port=9999 serve
```

---

## Proxy Jump 

On your machine:

```bash
ssh -L :3334:localhost:9999 jrc002i.jureca
```

- Mind the `i` letter I added at the end of the hostname

- Now you can access the service on your local browser at [http://localhost:3334](http://localhost:3334)

---

### Now that's really the end! 😓

