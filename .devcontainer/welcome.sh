#!/bin/bash
PS1='✨ \u@\h:\w\$ '
echo -e "\n\033[1;35m✨ CleanDiffuser Dev Container\033[0m"
echo -e "\033[0;36m  PyTorch: $(python -c 'import torch; print(torch.__version__)')\033[0m"
echo -e "\033[0;36m  CUDA:    $(python -c 'import torch; print(torch.version.cuda)')\033[0m"
echo -e "\033[0;32m  Ready to diffuse. ✨\033[0m\n"
