# Getting this working using an AMD RX 7900 XT or XTX

## Install Ubuntu and pre-requisites

Install Ubuntu Desktop 22.04.3 - you can always dual-boot this if you're doing training on your gaming PC. This section of the documentation is based on https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/install-radeon.html

### Install AMD unified driver package repositories and installer script

Download and install the amdgpu-install script on the system.

Enter the following commands to install the installer script for Ubuntu® version 22.04.3:

```bash
sudo apt update
wget https://repo.radeon.com/amdgpu-install/23.20.00.48/ubuntu/jammy/amdgpu-install_5.7.00.48.50700-1_all.deb
sudo apt install ./amdgpu-install_5.7.00.48.50700-1_all.deb
```

### Install the Graphics usecase

Run the following command to install open source graphics and ROCm.

```bash
amdgpu-install -y --usecase=graphics,rocm
```

    Watch for output warning or errors indicating an unsuccessful driver installation.

    NOTE: The -y option installs non-interactively. This step may take several minutes, depending on internet connection and system speed.

Reboot the system.

```bash
sudo reboot
```

### Set permissions for Groups to allow access to GPU hardware resources

Once the driver is installed, add any current user to the render and video groups to access GPU resources.

    Reboot in order for group changes to take effect.

Add user to render and video groups

Enter the following command to check groups in the system:

```bash
groups
```

Add user to the render and video group using the command:

```bash
sudo usermod -a -G render,video $LOGNAME
```

Reboot the system.

```bash
sudo reboot
```

### Post-install verification checks

Run these post-installation checks to verify that the installation is complete:

Verify that the current user is added to the render and video groups.

```bash
groups
```

Expected result:

`<username>` adm cdrom sudo dip video plugdev render lpadmin lxd sambashare

`<username>` indicates the current user, and this result will vary in your environment.

Check if amdgpu kernel driver is installed.

```bash
dkms status
```

Expected result:

amdgpu/x.x.x-xxxxxxx.xx.xx, x.x.x-xx-generic, x86_64: installed

Check if the GPU is listed as an agent.

```bash
rocminfo
```

Expected result:

[...]
*******
Agent 2
*******
  Name:                    gfx1100
  Uuid:                    GPU-5ecee39292e80c37
  Marketing Name:          Radeon RX 7900 XTX
  Vendor Name:             AMD
  [...]
[...]

Check if the GPU is listed.

```bash
clinfo
```

Expected result:

[...]
  Platform Name:                 AMD Accelerated Parallel Processing
Number of devices:               1
  Device Type:                   CL_DEVICE_TYPE_GPU
  Vendor ID:                     1002h
  Board name:                    Radeon RX 7900 XTX
[...]

### PyTorch via PIP installation method

AMD recommends the PIP install method to create a PyTorch environment when working with ROCm for machine learning development.

Check Pytorch.org for latest PIP install instructions and availability. See Compatibility matrices for support information.

To install PyTorch,

Enter the following command to unpack and begin set up.

```bash
sudo apt install python3-pip -y
```

Enter this command to update the pip wheel.

```bash
pip3 install --upgrade pip wheel
```

Enter this command to install Torch and Torchvision for ROCm AMD GPU support.

```bash
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-5.7/torch-2.0.1%2Brocm5.7-cp310-cp310-linux_x86_64.whl
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-5.7/torchvision-0.15.2%2Brocm5.7-cp310-cp310-linux_x86_64.whl
pip3 install --force-reinstall torch-2.0.1+rocm5.7-cp310-cp310-linux_x86_64.whl torchvision-0.15.2+rocm5.7-cp310-cp310-linux_x86_64.whl 
```

This may take several minutes.

Important! AMD recommends proceeding with ROCm WHLs available at repo.radeon.com.
The ROCm WHLs available at PyTorch.org are not tested extensively by AMD as the WHLs change regularly when the nightly builds are updated.

### Verify PyTorch installation

Confirm if PyTorch is correctly installed.

Verify if Pytorch is installed and detecting the GPU compute device.

```bash
python3 -c 'import torch' 2> /dev/null && echo 'Success' || echo 'Failure'
```

Expected result:

Success

Enter command to test if the GPU is available.

```bash
python3 -c 'import torch; print(torch.cuda.is_available())'
```

Expected result:

True

Enter command to display installed GPU device name.

```bash
python3 -c "import torch; print(f'device name [0]:', torch.cuda.get_device_name(0))"
```

Expected result: Example: device name [0]: Radeon RX 7900 XTX

device name [0]: <Supported AMD GPU>

Enter command to display component information within the current PyTorch environment.

```bash
python3 -m torch.utils.collect_env
```

Expected result:

PyTorch version
 
ROCM used to build PyTorch
 
OS
 
Is CUDA available
 
GPU model and configuration
 
HIP runtime version
 
MIOpen runtime version

Environment set-up is complete, and the system is ready for use with PyTorch to work with machine learning models, and algorithms.

## Update the code

Identify any parts written specifically for NVIDIA GPUs and modify them to be compatible with AMD GPUs. 

In PyTorch this typically means replacing `.cuda()` calls with `.to('hip')` to leverage ROCm's HIP platform.