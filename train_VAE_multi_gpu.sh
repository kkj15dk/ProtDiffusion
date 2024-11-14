#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q gpuv100
###BSUB -R "select[gpu32gb]"
### -- set the job Name -- 
#BSUB -J train_VAE_multi
### -- ask for number of cores (default: 4) -- 
#BSUB -n 8
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=2:mode=exclusive_process"
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify amount of memory per core/slot -- 
#BSUB -R "rusage[mem=2GB]"
### -- set walltime limit: hh:mm -- 
#BSUB -W 00:10
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
accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=2 ProtDiffusion/train_VAE.py