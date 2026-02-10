#!/bin/bash
#SBATCH --job-name=gsm8k-LL-PPL-exp
#SBATCH --account=mg61
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:A100:1
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --constraint="A100-80G"
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=cchi0115@student.monash.edu
#SBATCH --output=log/%j.out
#SBATCH --error=log/%j.err

#source ~/mg61_scratch2/ch_env/DALBenchmark/bin/activate
module load miniforge3
conda activate DAL
export HF_HOME=~/mg61_scratch2/chunheng/huggingface
export HUGGINGFACE_HUB_CACHE=~/mg61_scratch2/chunheng/huggingface/hub
module load cuda/12.2.0
#nvidia-smi
#pip install -r requirements.txt
#pip install torchlars==0.1.2 --no-build-isolation
#git clone https://github.com/decile-team/submodlib.git
#cd submodlib
#pip install .
#cd ..
#rm -rf submodlib

echo "job started!"

python main.py --method Hybrid --hybrid-strategy diversity_prior --uncertainty Perplexity --diversity LLMLabel --consistent-k 10 --dataset GSM8K \
    --textset --model Llama3 --causal-lm --free-form \
    --trial 3 --cycle 15 --n-initial 100  --n-query 300 --epochs 4 --workers 0 --batch-size 2 --test-batch-size 16 \
    --optimizer AdamW --scheduler StepLR --step_size 2 --gamma 0.5 --lr 1e-4
