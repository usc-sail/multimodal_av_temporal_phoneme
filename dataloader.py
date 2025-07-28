#General libraries needed for model training/evaluation
import torch
import torch.nn.functional as F
# import torchvision
import torchaudio
# from torchaudio.utils import download_asset
from torch.utils.data import Dataset, DataLoader
# import IPython
# import matplotlib.pyplot as plt
import os
import random
# import sys
import numpy as np
import cv2
# from typing import Optional, Tuple
from itertools import groupby
# import Levenshtein
from einops import rearrange
import glob
import torchvision.models as models

class VideoAudioPhonemeDataset(Dataset):
    def __init__(self, root_dir, transform=None, training=True, modality="f"):
        """
        Args:
            root_dir (str): Path to the directory containing video, audio and text files.
            transform (callable, optional): Transform for video frames.
        """
        self.root_dir = root_dir
        self.modality = modality
        all_of_files = glob.glob(root_dir + "/five_second_of_1/*.npy")
        all_video_files = glob.glob(root_dir + "/avi/five_second_clips_1/*.avi") 
        all_audio_files = glob.glob(root_dir + "/five_second_audio_1/*.wav") 
        all_token_files = glob.glob(root_dir + "/five_second_tokens_1/*.txt") 
        sample_quantity = 0
        if training:
            all_of_files = all_of_files[0:800]
            all_video_files = all_video_files[0:800]
            all_audio_files = all_audio_files[0:800]
            all_token_files = all_token_files[0:800]
            sample_quantity = 800
        else:
            all_of_files = all_of_files[800:1001]
            all_video_files = all_video_files[800:1001]
            all_audio_files = all_audio_files[800:1001]
            all_token_files = all_token_files[800:1001]
            sample_quantity = 200
        indices = random.sample(range(len(all_video_files)), sample_quantity)


        #Since the dataset of 5 second sequences is too large, randomly choose 10000 of them.
        self.flows_files = []
        self.video_files = []
        self.audio_files = []
        self.token_files = []
        # self.flows_files = all_of_files[indices]
        for i in indices:
            self.flows_files.append(all_of_files[i])
            self.video_files.append(all_video_files[i])
            self.audio_files.append(all_audio_files[i])
            self.token_files.append(all_token_files[i])

        self.transform = transform
        self.phonemes = ['', 'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW', 'B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N', 'NG', 'P', 'R', 'S', 'SH', 
                    'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH']

    def __len__(self):
        return len(self.video_files)

    def get_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(cv2.resize(frame, (128, 128)))
        video = torch.tensor(np.array(frames), dtype=torch.float32)
        video = rearrange(video, 't h w c -> t c h w')  # Rearrange to (T, C, H, W)
        return video

    def get_audio(self, audio_path):
        #Fetch audio
        audio, sr = torchaudio.load(audio_path)
        #if sr != 16000:
        #    audio = torchaudio.transforms.Resample(sr, 16000)(audio)
        zeroPaddedAudio = torch.zeros(1, 80320)
        zeroPaddedAudio[:,0:80000] = audio[:,0:80000]
        return zeroPaddedAudio
    
    def get_optical_flow(self, of_path):
        flows = np.load(of_path)
        return torch.Tensor(flows).float()

    def __getitem__(self, idx):
        #Fetch video
        video, audio, flows = 0, 0, 0
        if('i' in self.modality):
            video = self.get_video(self.video_files[idx])
        if('a' in self.modality):
            audio = self.get_audio(self.audio_files[idx])
        if('f' in self.modality):
            flows = self.get_optical_flow(self.flows_files[idx])

        # Fetch label tokens, The tokens are in the form of phonemes, so we need to convert them to indices
        token_path = os.path.join(self.root_dir, "five_second_tokens_1", self.token_files[idx])
        tokens = []

        with open(token_path, 'r') as file:
            for line in file:
                if ''.join(char for char in line.strip() if char.isalpha()) == "H":
                    tokens.append(self.phonemes.index('HH'))
                else:
                    tokens.append(self.phonemes.index(''.join(char for char in line.strip() if char.isalpha())))

        return {
            'video': video,
            'audio': audio,
            'flows': flows,
            'phonemes': F.pad(torch.tensor([key for key, _ in groupby(tokens)]), (0, 250-torch.tensor([key for key, _ in groupby(tokens)]).shape[0])),
            'phoneme_lengths': torch.tensor(len([key for key, _ in groupby(tokens)]))
        }
