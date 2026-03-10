---
author: Alexandre Strube // Sabrina Benassou // Ismail Khalfaoui-Hassani // Javad Kasravi
title: Running and Scaling Performent Deep Learning Models on HPC
#subtitle: A primer in supercomputers
date: March 10, 2025
---

Dear students,

the next "Bringing Deep Learning Workloads to JSC supercomputers" course is approaching! Thank you all very much for your participation.

The course is online, over zoom. It might be recorded. This is the link:
https://go.fzj.de/bringing-dl-workloads-to-jsc-zoom


*********
IMPORTANT - Please check all steps! Some things need to be done some days BEFORE the course!!!
*********

Checklist for BEFORE the course:

- If you don't have one, make an account on JuDOOR, our portal: https://judoor.fz-juelich.de/register
Instruction video: https://drive.google.com/file/d/1-DfiNBP4Gta0av4lQmubkXIXzr2FW4a-/view

- Joining the course's project: https://go.fzj.de/bringing-dl-workloads-to-jsc-project-join

- Sign the usage agreements, as shown in this video: https://drive.google.com/file/d/1mEN1GmWyGFp75uMIi4d6Tpek2NC_X8eY/view

- Install software (see below). On windows you DO need administrator rights. We can't support other softwares during the course.

- We will use Slack for communication. Please log in BEFORE the course: https://go.fzj.de/bringing-dl-workloads-to-jsc-slack


---

What software is necessary for this course?

The course is platform-independent. It can even be followed by a Windows user, but if possible, avoid it. In general. Forever.

- Visual Studio Code: it's a free editor which we will demo on this course. Get it from https://code.visualstudio.com/download

- Visual Studio Code Remote Development: https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack

- Visual Studio: Remote - SSH: https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh

- A terminal. On Linux and Mac, it's just called "Terminal". Little familiarity with it is required. On windows, the WSL installs it. PowerShell is also an option, also STRONGLY discouraged.

- The `ssh` command. It's installed by default on Mac and Linux, and should be on Windows after the aforementioned steps.

- Some knowledge of the Python language.

- (WINDOWS ONLY): You can use WSL to have all Linux tools under windows, but this is optional.. This installs the WSL support for Visual Studio Code, which will install WSL itself (And Ubuntu). https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-wsl - This is a long install, take your time.
  PLEASE MAKE SURE WSL IS ACTUALLY INSTALLED - Try running it. Check this example: https://pureinfotech.com/install-windows-subsystem-linux-2-windows-10/

---

***********************
NOTE FOR ADVANCED USERS
***********************

The first day of the course is intended for people to get acquainted to the environment. If you are already familiar with the tools, you can skip the first day. On it, we will cover:
- Using SSH
- Creating SSH keys
- Setting up a stable connection to JSC supercomputers
- Using Visual Studio Code
- Using the JSC batch system
- Running a simple job
- Running a simple deep learning job
- Installing software on your own virtual environment



The course material is available at https://go.fzj.de/bringing-dl-workloads-to-jsc - I will be making some final commits to it, so make sure you reload it every now and then.

See you soon, 

Alexandre, Sabrina, Ismail and Javad





