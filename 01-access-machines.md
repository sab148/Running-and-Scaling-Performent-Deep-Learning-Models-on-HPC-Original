---
author: Sabrina Benassou // Eske Ewert // Javad Kasravi
title: Accessing the machines, intro 
#subtitle: A primer in supercomputers
date: November 17, 2026

---
## Communication:

Links for the complimentary parts of this course: 

- [JSC Training Page](https://go.fzj.de/running-and-scaling-deep-learning-models-on-HPC-training-page)
- [Judoor project page invite](https://judoor.fz-juelich.de/projects/join/training2658)
- [GitHub Code](https://github.com/sab148/Running-and-Scaling-Performent-Deep-Learning-Models-on-HPC/tree/ai_training_hpc)
- Our mailing list for [AI news](https://lists.fz-juelich.de/postorius/lists/ai.jsc.lists.fz-juelich.de/)
- [Survey at the end of the course](https://go.fzj.de/bringing-dl-workloads-to-jsc-survey)
- [Virtual Environment template](https://gitlab.jsc.fz-juelich.de/kesselheim1/sc_venv_template)
- [SOURCE of the course/slides on Github](https://github.com/sab148/Running-and-Scaling-Performent-Deep-Learning-Models-on-HPC-Original)

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

### JURECA DC Compute Nodes

- 192 Accelerated Nodes (with GPUs)
- 2x AMD EPYC Rome 7742 CPU 2.25 GHz (128 cores/node)
- 512 GiB memory
- Network Mellanox HDR infiniband (FAST💨 and EXPENSIVE💸)
- 4x NVIDIA A100 with 40gb 😻
- TL;DR: 24576 cores, 768 GPUs 💪
- Way deeper technical info at [Jureca DC Overview](https://apps.fz-juelich.de/jsc/hps/jureca/configuration.html)

---

### Connecting to Jureca DC

#### Getting compute time
- Go to [https://judoor.fz-juelich.de/projects/join/training2658](https://judoor.fz-juelich.de/projects/join/training2658)
- Join the course project `training2658`
- Sign the Usage Agreements ([Video](https://drive.google.com/file/d/1mEN1GmWyGFp75uMIi4d6Tpek2NC_X8eY/view))
- Compute time allocation is based on compute projects. For every compute job, a compute project pays.
- Time is measured in core-hours. One hour of Jureca DC is 128 core-hours.
- Example: Job runs for 8 hours on 64 nodes of Jureca DC: 8 * 64 * 128 = 65536 core-h!

---

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
mkdir $PROJECT_training2658/$USER

# Create a shortcut for the project on the home folder
rm -rf ~/course ; ln -s $PROJECT_training2658/$USER ~/course

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

### Extra software, modules and kernels

#### You want some extra Python software from PyPI...

We will use [`uv`](https://docs.astral.sh/uv/) to create a virtual environment and install our Python dependencies.

* The environment will live in `$HOME/course/.venv`
* Install packages on the **login node**
* The same environment can then be used from the compute nodes

---

### Install `uv`

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

### Create a virtual environment

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

### Example: Let's install some software!

Even though the supercomputer provides a lot of software, sometimes we need additional Python packages.

For example:

* fast.ai
* Weights & Biases
* Transformers
* Lightning
* 🤗 Datasets
* We will install them into our `.venv`.

---

### Create `requirements.txt`

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

---

### Install the dependencies with `uv`

Make sure you are still in the course directory:

```bash
cd $HOME/course/
```

Then run:

```bash
uv pip install -r requirements.txt
```

`uv` automatically installs the packages into:

```text
$HOME/course/.venv
```


Check the installation:

```bash
uv pip check
```

---

### Activating the virtual environment

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

---

To leave the environment:

```bash
deactivate
```

---

### Adding more software later

Want another Python package?

For example:

```bash
cd $HOME/course/
uv pip install rich
```

---

### Create a Jupyter kernel

Our environment already contains `ipykernel`.

Activate it:

```bash
cd $HOME/course/
source .venv/bin/activate
```

Then register it as a Jupyter kernel:

```bash
python -m ipykernel install --user \
    --name training2658-uv \
    --display-name "Python (training2658 / uv)"
```

You can now select it as the kernel in Jupyter.

```text
Python (training2658 / uv)
```

---

### Using the environment in VS Code

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

---
