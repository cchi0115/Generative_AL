#!/bin/bash
#SBATCH --job-name=STAR_train
#SBATCH --account=mg61
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:A100:1
#SBATCH --partition=fit
#SBATCH --qos=fitq
#SBATCH --time=08:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=cchi0115@student.monash.edu
#SBATCH --output=log/%x-%j.out
#SBATCH --error=log/%x-%j.err

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

python main.py --method Uncertainty --uncertainty Entropy --dataset AGNEWS --textset --model Llama \
    --trial 1 --cycle 5 --n-initial 10000 --n-query 3500 --n-class 4 --epochs 5 --workers 2 --batch-size 4 --test-batch-size 16 \
    --optimizer AdamW --scheduler linear --lr 2e-5 --scheduler StepLR --step_size 9999 --gamma 0.5
