Prerequisite libraries:
-torch
-torchaudio
-torchvision
-matplotlib
-numpy
-cv2
-transformers
-Levenshtein
-IPython
-transformers
-einops
-mambapy

Commands to setup CARC environment:
'''
salloc --partition=gpu --ntasks=1 --gpus-per-task=2 --mem=16G --time=16:00:00
module purge
module load gcc/13.3.0
module load cuda/12.6.3
module load cudnn/8.9.7.29-12-cuda
module load python/3.12.8
module load conda/25.3.0
module load nano
'''

Command to start training:
-python train.py
