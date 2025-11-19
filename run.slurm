#!/bin/bash
#SBATCH --job-name=DAL_train
#SBATCH --account=mg61
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:A100:1
#SBATCH --partition=fit
#SBATCH --qos=fitq
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=cchi0115@student.monash.edu
#SBATCH --output=log/%j.out
#SBATCH --error=log/%j.err

source ~/mg61_scratch2/ch_env/DALBenchmark/bin/activate
export HF_HOME=~/mg61_scratch2/chunheng/huggingface
export HUGGINGFACE_HUB_CACHE=~/mg61_scratch2/chunheng/huggingface/hub
#module load cuda/12.2.0
#pip install -r requirements.txt
#pip install torchlars==0.1.2
#git clone https://github.com/decile-team/submodlib.git
#cd submodlib
#pip install .
#cd ..
#rm -rf submodlib

python main.py --method Random --dataset AGNEWS --textset --model LlamaCausal --causal-lm \
    --trial 1 --cycle 2 --n-initial 500 --n-query 100 --n-class 4 --epochs 4 --workers 4 --batch-size 4 --test-batch-size 32 \
    --optimizer AdamW --scheduler linear --lr 2e-5 --scheduler StepLR --step_size 9999 --gamma 0.5
