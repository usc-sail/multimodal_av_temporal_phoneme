#General libraries needed for model training/evaluation
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchaudio
from torchaudio.utils import download_asset
from torch.utils.data import Dataset, DataLoader
import IPython
import matplotlib.pyplot as plt
import os
import random
import sys
import numpy as np
import cv2

#Libraries needed for wav2vec2-lv-60-espeak-cv-ft
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset

class AV_Conformer(nn.Module):
    def __init__(self, modality='multimodal', feats='base', num_heads=4, num_layers=3):
        super().__init__()

        # Modality type: 'audio', 'video', or 'multimodal'
        self.audio_dim = 1024
        self.vid_dim = 1024
        self.modality = modality

        if modality == 'audio':
            input_dim = self.audio_dim
        elif modality == 'video':
            input_dim = self.vid_dim
        else:
            input_dim = self.audio_dim + self.vid_dim
    
        #self.conformer = Conformer(
        #    input_dim=input_dim,
        #    num_heads=num_heads,
        #    ffn_dim=256,
        #    num_layers=num_layers,
        #    depthwise_conv_kernel_size=31,
        #    dropout=0.3
        #)

        # LSTM decoder
        self.decoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
        )

        # Final classification layer
        self.classifier = nn.Linear(128, 40)

    def forward(self, audio_features, video_features, feat_lengths):
    
        if self.modality == 'audio':
            x = audio_features #Shape: [batch_size, 250, 1024]
        elif self.modality == 'video':
            x = video_features
        else:  # multimodal
            x = torch.cat([audio_features, video_features], dim=-1)
        batch_size = x.shape[0]

        #lengths = torch.full((batch_size,), 512, dtype=torch.long)
        lengths = feat_lengths

        # Pass through Conformer
        #x, reps, att, _ = self.conformer(x, lengths)  # Shape: [batch_size, seq_len, conformer_dim]
        #print(reps.shape)

        # Decode with LSTM
        x, _ = self.decoder(x)  # Shape: [batch_size, seq_len, lstm_dim]

        # Classification
        logits = self.classifier(x)  # Shape: [batch_size, seq_len, 40]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        return log_probs

class VideoAudioPhonemeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Path to the directory containing video, audio and text files.
            transform (callable, optional): Transform for video frames.
        """
        self.root_dir = root_dir
        all_video_files = [f for f in sorted(os.listdir(os.path.join(root_dir, "avi/five_second_clips"))) if f.endswith('.avi')]
        all_audio_files = [f for f in sorted(os.listdir(os.path.join(root_dir, "five_second_audio"))) if f.endswith('.wav')]
        all_token_files = [f for f in sorted(os.listdir(os.path.join(root_dir, "five_second_tokens"))) if f.endswith('.txt')]
        indices = random.sample(range(len(all_video_files)), 10000)
        
        #Since the dataset of 5 second sequences is too large, randomly choose 10000 of them.
        self.video_files = []
        self.audio_files = []
        self.token_files = []
        for i in indices:
            self.video_files.append(all_video_files[i])
            self.audio_files.append(all_audio_files[i])
            self.token_files.append(all_token_files[i])

        self.transform = transform
        self.phonemes = ['', 'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW', 'B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N', 'NG', 'P', 'R', 'S', 'SH', 
                    'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH']

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        #Fetch video
        video_path = os.path.join(self.root_dir, "avi/five_second_clips", self.video_files[idx])
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        video = torch.tensor(np.array(frames), dtype=torch.float32)
        
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
            'video': video,
            'audio': zeroPaddedAudio,
            'tokens': torch.tensor(tokens)
        }

# Define video directory path
video_directory = "/data1/jaypark/single_spk_corpus"

#Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Define batch length
batch_length = 8

# Define the DataLoader
whole_dataset = VideoAudioPhonemeDataset(video_directory)
train_len = int(len(whole_dataset)*0.7)
train_set, test_set = torch.utils.data.random_split(whole_dataset, [train_len, len(whole_dataset)-train_len])

train_loader = DataLoader(train_set, batch_size=batch_length, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_length, shuffle=False)

#Prepare models for audio feature acquisition
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
audio_features_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft").to(device)

#Prepare models for video feature acquisition

#Initialize our training parameters
finalModel = AV_Conformer(modality="audio", num_heads=4, num_layers=3).to(device)
optimizer = torch.optim.Adam(finalModel.parameters(), lr=1e-3, weight_decay=1e-4)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=20, gamma=0.9)
loss_function = nn.CTCLoss(blank=0, zero_infinity=True)

epochs = 1000
for t in range(epochs):
    epoch_loss = 0.0
    num_batches = 0
    print(f"Epoch {t+1}\n-------------------------------")
    for index, batch in enumerate(train_loader):
        finalModel.train()
        optimizer.zero_grad()
        videos = batch["video"].to(device) #Shape: [batch_size, num_frames, H, W, 3]
        audio = batch["audio"].to(device) #Shape: [batch_size, num_channels, num_samples]
        tokens = batch["tokens"].to(device) #Shape: [batch_size, time_steps]

        audio_values = processor(audio[:,0,:], sampling_rate = 16000, return_tensors="pt").input_values.to(device)
        with torch.no_grad():
            audio_features = audio_features_model(audio_values[0], output_hidden_states=True).hidden_states[-1] #Shape: [batch_size, time_steps, 1024]
        log_probs = finalModel(audio_features, videos, 1024) #Shape: [batch_size, 250, 40]
        # Prepare input and target lengths (all sequences are length 250 in your case)
        input_lengths = torch.full(size=(batch_length,), fill_value=250, dtype=torch.long)
        target_lengths = torch.randint(low=1, high=250, size=(batch_length,), dtype=torch.long)

        loss = loss_function(log_probs.transpose(0, 1),  # CTC expects [seq_len, batch, num_classes]
                tokens, 
                input_lengths, 
                target_lengths)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(finalModel.parameters(), max_norm=0.5)
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1
        
        if index % 12 == 0:
            loss, current, size = loss.item(), index * batch_length + len(audio), len(train_loader.dataset)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    avg_epoch_loss = epoch_loss / num_batches
    print(f"Average epoch loss: {avg_epoch_loss:>7f}")
    print("Calculating test loss...")
    lr_scheduler.step()

    finalModel.eval()
    total_loss = 0.0
    num_batches = 0
    incorrect_tokens = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in test_loader:
            videos = batch["video"].to(device)
            audio = batch["audio"].to(device)
            tokens = batch["tokens"].to(device)
        
            audio_values = processor(audio[:,0,:], sampling_rate = 16000, return_tensors="pt").input_values.to(device)
            audio_features = audio_features_model(audio_values[0], output_hidden_states=True).hidden_states[-1] #Shape: [batch_size, time_steps, 1024]
            log_probs = finalModel(audio_features, videos, 1024) #Shape: [batch_size, 250, 40]
            
            # Prepare input and target lengths (all sequences are length 250 in your case)
            input_lengths = torch.full(size=(batch_length,), fill_value=250, dtype=torch.long)
            target_lengths = torch.randint(low=1, high=250, size=(batch_length,), dtype=torch.long)

            loss = loss_function(log_probs.transpose(0, 1),  # CTC expects [seq_len, batch, num_classes]
                tokens, 
                input_lengths, 
                target_lengths)
            total_loss += loss.item()
            num_batches += 1
            token_matches = torch.eq(torch.argmax(log_probs, dim=-1).flatten(), tokens.flatten())
            for i in range(batch_length * 250):
                if not token_matches[i].item():
                    incorrect_tokens = incorrect_tokens + 1
            total_tokens = total_tokens + batch_length * 250

    avg_loss = total_loss / num_batches
    print(f"Average test loss: {avg_loss:>7f}")
    token_error_rate = float(incorrect_tokens)/float(total_tokens)
    print(f"Token error rate on testing dataset: {token_error_rate:>7f}\n")
