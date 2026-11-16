---
author: Sabrina Benassou // Eske Ewert // Javad Kasravi
title: Accessing the machines, intro 
#subtitle: A primer in supercomputers
date: November 17, 2026

---
## Communication:

Links for the complimentary parts of this course: 

- [Course Info](https://gitlab.jsc.fz-juelich.de/hedgedoc/s/-oXsVYvsJ)
- [Judoor project page invite](https://judoor.fz-juelich.de/projects/join/training2658)
- [GitHub Code](https://github.com/sab148/Running-and-Scaling-Performent-Deep-Learning-Models-on-HPC/tree/ai_training_hpc)
- Our mailing list for [AI news](https://lists.fz-juelich.de/postorius/lists/ai.jsc.lists.fz-juelich.de/)

![](images/Logo_FZ_Juelich_rgb_Schutzzone_transparent.svg)

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

# SSH: Connecting to JURECA

- SSH provides secure remote access to a supercomputer through a login node.

- **To connect, you need:**
1. An SSH key pair
2. An SSH configuration
3. Your public key and permitted IP address registered in JUDOOR

---

# Generate Your SSH Key

- Open the VS Code terminal (**View → Terminal**).

  ```bash
  mkdir -p ~/.ssh
  ssh-keygen -a 100 -t ed25519 -f ~/.ssh/id_ed25519-JSC
  ```

- **Private key:** `id_ed25519-JSC` — keep secret
- **Public key:** `id_ed25519-JSC.pub` — register in JUDOOR

---

# Configure SSH

- Open the configuration file:

  ```bash
  code ~/.ssh/config
  ```

- Add:

  ```sshconfig
  Host jureca
      HostName jureca.fz-juelich.de
      User YOUR_USERNAME
      AddressFamily inet
      IdentityFile ~/.ssh/id_ed25519-JSC
      MACs hmac-sha2-512-etm@openssh.com
  ```

**Replace `YOUR_USERNAME` with your JSC username.**

> **Windows/WSL:** If VS Code uses Windows SSH, copy your key and configuration into the Windows `.ssh` directory and set appropriate key permissions.

---

# Find Your IP Address

JSC restricts SSH access by IP address.

1. Visit [whatismyip.com](https://www.whatismyip.com).
2. Copy your public **IPv4** address.
3. Prepare the permitted IP range for JUDOOR.

**Example:**

- Public IP: `93.199.55.163`
- Exact-IP range: `93.199.55.163/32`

> Use a wider range such as `/16` only if required and approved for your network.

---

# Register Your SSH Key in JUDOOR

- Open your public key:

  ```bash
  code ~/.ssh/id_ed25519-JSC.pub
  ```

--- 

# Register Your SSH Key in JUDOOR

- Add your permitted IP range before the public key (keep the **full** public key on one line):

  ```text
  from="93.199.55.163/32,10.0.0.0/8" ssh-ed25519 AAAA... yourname@host
  ```

1. Open [JUDOOR](https://judoor.fz-juelich.de).
2. Go to **Manage SSH Keys**.
3. Add the complete line for **JURECA**.
4. Allow a few minutes for activation.

> Replace the example IP and placeholder key with your own. Include `10.0.0.0/8` or FZJ's `134.94.0.0/16` only where appropriate for your access setup.

---

# Test Your Connection

- In the VS Code terminal, run:

```bash
ssh jureca
```

- On your first connection, verify the host fingerprint against an official JSC source before accepting it.

**Expected result:**

```text
Welcome to JURECA DC
...
YOUR_USERNAME@jrlogin03 ~ $
```

**Exercise:** Connect successfully to the JURECA login node.

---

# Set Up Your Working Directory

- After connecting to JURECA:

**1. Create your workspace**

```bash
mkdir -p "$PROJECT_training2658/$USER"
ln -s "$PROJECT_training2658/$USER" ~/course
cd ~/course
pwd
```

---

# Set Up Your Working Directory

**2. Create application directories**

```bash
mkdir -p ~/course/{.cache,.config,.fastai}
```

---

# Set Up Your Working Directory

**3. Link application directories**

```bash
ln -s ~/course/.cache ~/.cache
ln -s ~/course/.config ~/.config
ln -s ~/course/.fastai ~/.fastai
```

> Create links only if the corresponding home-directory paths do not already exist. Back up or migrate existing data rather than deleting it.

**Goal:** A working SSH connection and a prepared course directory.

---

## Working with the supercomputer's software

- We have literally thousands of software packages, hand-compiled for the specifics of the supercomputer.
- [Full list](https://www.fz-juelich.de/en/ias/jsc/services/user-support/using-systems/software)
- [Detailed documentation](https://apps.fz-juelich.de/jsc/hps/jureca/software-modules.html)

---

# Python Software, Modules & Kernels

# 1. Check `uv`

In the terminal:

```bash
uv --version
```

If unavailable, install it in your user environment:

```bash
python -m pip install --user uv
```

Then check ``uv --version`` again. If the command is not found, add your user-level Python scripts directory to `PATH` or use the JSC-provided `uv` module, if available.

---

# 2. Create the Virtual Environment

```bash
cd ~/course
module load Stages/2025
module load GCC OpenMPI Python
uv venv --python 3.12 .venv
```

**Result:** `~/course/.venv`

> Use the same module stack when running Python later.

---

# 3. Define Python Dependencies

Create `~/course/requirements.txt` with:

```text
ipykernel
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

# 4. Install & Check

From `~/course`:

```bash
uv pip install --python .venv/bin/python -r requirements.txt
uv pip check --python .venv/bin/python
```

**All packages go into `.venv`.**

> Package versions and availability can depend on the JSC module stack and platform.

---

# 5. Activate & Test

```bash
cd ~/course
source .venv/bin/activate
which python
python -c "import torch; print(torch.__version__)"
python -c "import fastai; print(fastai.__version__)"
python -c "import wandb; print(wandb.__version__)"
```

Expected Python path: `~/course/.venv/bin/python`

Exit with:

```bash
deactivate
```
