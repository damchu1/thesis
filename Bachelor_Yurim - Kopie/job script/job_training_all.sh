#!/bin/bash

#SBATCH -J trainingecoli
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem-per-cpu=3800
#SBATCH -t 72:00:00
#SBATCH -A project02465
#SBATCH -e /work/scratch/yj90zihi/logs/log.err.%j
#SBATCH -o /work/scratch/yj90zihi/logs/log.out.%j
#SBATCH --mail-user=yurim.jeong@stud.tu-darmstadt.de
#SBATCH --mail-type=NONE
# -------------------------------
module purge

source /work/home/yj90zihi/vnv/bin/activate

module load gcc python cuda cuDNN

python local_training.py

deactivate
