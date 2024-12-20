#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q gpuv100
#BSUB -R "select[gpu32gb]"
### -- set the job Name -- 
#BSUB -J train_DiT
### -- ask for number of cores (default: 4) -- 
#BSUB -n 8
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify amount of memory per core/slot -- 
#BSUB -R "rusage[mem=2GB]"
### -- set walltime limit: hh:mm -- 
#BSUB -W 2:00
### -- send notification at start -- 
#BSUB -B
### -- send notification at completion -- 
#BSUB -N
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o Logs/%J.out 
#BSUB -e Logs/%J.err 

nvidia-smi

module load python3/3.12.4
source .venv/bin/activate

# here follow the commands you want to execute
accelerate launch ProtDiffusion/train_ProtDiffusion_v3.py