#!/bin/bash

#SBATCH --cpus-per-task=8
#SBATCH --mem=48000M
#SBATCH --time=0-1:00
#SBATCH --account=
#SBATCH --gres=gpu:h100:1
#SBATCH --output=logs/benchmark/slice-gpt-llama-2-7b-hf-%j.out
#SBATCH --error=logs/benchmark/slice-gpt-llama-2-7b-hf-%j.err
#SBATCH --mail-user=
#SBATCH --mail-type=END

nvidia-smi

module load python/3.11
module load cuda/12.6
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
module load gcc arrow/21.0.0

pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt
pip install lm-eval==0.4.1
pip install -e .[experiment]

python experiments/run_benchmark.py \
           --model meta-llama/Llama-2-7b-hf \
           --sliced-model-path models/llama_2_7b_hf \
           --eval-dataset wikitext2 \
           --ntokens 128 \
           --sparsity 0.5 \
           --device cuda:0 \
           --no-wandb \
           --hf-token $HF_TOKEN