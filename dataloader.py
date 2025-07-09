#General libraries needed for model training/evaluation
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
import os
import random
import numpy as np
import cv2
from itertools import groupby
from einops import rearrange

class VideoAudioPhonemeDataset(Dataset):
    def __init__(self, root_dir, transform=None, training=True):
        """
        Args:
            root_dir (str): Path to the directory containing video, audio and text files.
            transform (callable, optional): Transform for video frames.
        """
        self.root_dir = root_dir
        #all_flow_files = [f for f in sorted(os.listdir(os.path.join(root_dir, "five_second_flows"))) if f.endswith('.npy')]
        all_video_files = [f for f in sorted(os.listdir(os.path.join(root_dir, "avi/five_second_clips"))) if f.endswith('.avi')]
        all_audio_files = [f for f in sorted(os.listdir(os.path.join(root_dir, "five_second_audio"))) if f.endswith('.wav')]
        all_token_files = [f for f in sorted(os.listdir(os.path.join(root_dir, "five_second_tokens"))) if f.endswith('.txt')]
        sample_quantity = 0
        if training:
            #For CARC:
            #all_flow_files = all_flow_files[0:26097]
            #all_video_files = all_video_files[0:26097]
            #all_audio_files = all_audio_files[0:26097]
            #all_token_files = all_token_files[0:26097]

            #For Redondo:
            all_video_files = all_video_files[0:137583]
            all_audio_files = all_audio_files[0:137583]
            all_token_files = all_token_files[0:137583]
            sample_quantity = 8000
        else:
            #For CARC:
            #all_flow_files = all_flow_files[26097:]
            #all_video_files = all_video_files[26097:]
            #all_audio_files = all_audio_files[26097:]
            #all_token_files = all_token_files[26097:]

            #For Redondo:
            all_video_files = all_video_files[137583:]
            all_audio_files = all_audio_files[137583:]
            all_token_files = all_token_files[137583:]
            sample_quantity = 1000
        indices = random.sample(range(len(all_video_files)), sample_quantity)
        
        #Since the dataset of 5 second sequences is too large, randomly choose 10000 of them.
        #self.flow_files = []
        self.video_files = []
        self.audio_files = []
        self.token_files = []
        for i in indices:
            #self.flow_files.append(all_flow_files[i])
            self.video_files.append(all_video_files[i])
            self.audio_files.append(all_audio_files[i])
            self.token_files.append(all_token_files[i])

        self.transform = transform
        self.phonemes = ['', 'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW', 'B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N', 'NG', 'P', 'R', 'S', 'SH', 
                    'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH']

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        #Fetch flows
        #flow_path = os.path.join(self.root_dir, "five_second_flows", self.flow_files[idx])
        #flow = np.load(flow_path)
        
        #Fetch video
        video_path = os.path.join(self.root_dir, "avi/five_second_clips", self.video_files[idx])
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(cv2.resize(frame, (128, 128)))
        video = torch.tensor(np.array(frames), dtype=torch.float32)
        video = rearrange(video, 't h w c -> t c h w')
        cap.release()
        
        #Fetch audio
        audio_path = os.path.join(self.root_dir, "five_second_audio", self.audio_files[idx])
        audio, sr = torchaudio.load(audio_path)
        #if sr != 16000:
        #    audio = torchaudio.transforms.Resample(sr, 16000)(audio)
        zeroPaddedAudio = torch.zeros(1, 80320)
        zeroPaddedAudio[:,0:80000] = audio[:,0:80000]

        # Fetch tokens
        token_path = os.path.join(self.root_dir, "five_second_tokens", self.token_files[idx])
        tokens = []
        with open(token_path, 'r') as file:
            for line in file:
                if ''.join(char for char in line.strip() if char.isalpha()) == "H":
                    tokens.append(self.phonemes.index('HH'))
                else:
                    tokens.append(self.phonemes.index(''.join(char for char in line.strip() if char.isalpha())))

        return {
            #'flows': torch.tensor(flow, dtype=torch.float32),
            'video': video,
            'audio': zeroPaddedAudio,
            'phonemes': F.pad(torch.tensor([key for key, _ in groupby(tokens)]), (0, 250-torch.tensor([key for key, _ in groupby(tokens)]).shape[0])),
            'phoneme_lengths': torch.tensor(len([key for key, _ in groupby(tokens)]))
        }
