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
from typing import Optional, Tuple
from itertools import groupby
import Levenshtein
from dataloader import VideoAudioPhonemeDataset
from models import Articulator_Encoder
from utils import get_dataset, PhonemeErrorRate

#Libraries needed for wav2vec2-lv-60-espeak-cv-ft
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# Define video directory path for CARC
video_directory = "/project/shrikann_35/jaypark/data/single_spk_corpus"

#Define device for CARC
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Define batch length
batch_length = 1
phoneme_dictionary = ['', 'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW', 'B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N', 'NG', 'P', 'R', 'S', 'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH']
epochs = 800

#Define dataloaders
train_loader, test_loader = get_dataset(video_directory, batch_length)

#Prepare models for audio feature acquisition
audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft", do_phonemize=False)
audio_features_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft").to(device)

#Initialize our training parameters
finalModel = Articulator_Encoder(modality="articulator", num_heads=4, patch_size=1).to(device)
optimizer = torch.optim.Adam(finalModel.parameters(), lr=1e-3, weight_decay=1e-4)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=20, gamma=0.9)
loss_function = nn.CTCLoss(blank=0, zero_infinity=True)
train_per = PhonemeErrorRate(phoneme_dictionary, blank_id=0)
val_per = PhonemeErrorRate(phoneme_dictionary, blank_id=0)
x_min = torch.tensor(np.array([0.22690388071665196, 0.11955651562837223, 0.17564414476030232, 0.24438938867013488, 0.473685007464924, 0.26225585780008864]), dtype=torch.float).to(device)
x_max = torch.tensor(np.array([3.5341736387660894, 4.626061521150712, 2.986669440527201, 2.153838267333077, 2.6318475522713562, 2.706900564995662]), dtype=torch.float).to(device)

#Setup for model evaluation
def eval_step(engine, batch):
    return batch
best_val_per = float('inf')

for t in range(epochs):
    epoch_loss = 0.0
    num_batches = 0
    print(f"Epoch {t+1}\n-------------------------------")
    with open('training_output.txt', 'a') as file:
        file.write(f"Epoch {t+1}\n-------------------------------\n")
    for index, batch in enumerate(train_loader):
        finalModel.train()
        optimizer.zero_grad()
        
        articulators = batch["articulator"].to(device) #Shape: [batch_size, num_frames, H, W, 3]
        #videos = batch["video"] #Shape: [batch_size, num_frames, H, W, 3]
        audio = batch["audio"] #Shape: [batch_size, num_channels, num_samples]
        targets = batch["phonemes"].to(device) #Shape: [batch_size, time_steps]
        target_lengths = batch["phoneme_lengths"].to(device)

        #Acquire audio features
        audio_values = audio_processor(audio[:,0,:].to(device), sampling_rate = 16000, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            #audio_features = audio_features_model(audio_inputs[0])
            audio_features = audio_features_model(audio_values['input_values'][0,:,:]).last_hidden_state #Shape: [batch_size, time_steps, 1024]

        log_probs = finalModel(audio_features, articulators, 1024, x_min, x_max) #Shape: [batch_size, 250, 40]
        # Prepare input and target lengths (all sequences are length 250 in your case)
        input_lengths = torch.full(size=(batch_length,), fill_value=250, dtype=torch.long)
        #target_lengths = torch.randint(low=1, high=250, size=(batch_length,), dtype=torch.long)
        
        loss = loss_function(log_probs.transpose(0, 1),  # CTC expects [seq_len, batch, num_classes]
                targets, 
                input_lengths, 
                target_lengths)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(finalModel.parameters(), max_norm=0.5)
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

        train_per.add_batch(log_probs, targets)
        
        if index % 12 == 0:
            loss, current, size = loss.item(), index * batch_length + len(audio), len(train_loader.dataset)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    PER, edits, total = train_per.compute()
    train_per.save('train_pred.json')
    train_per.reset()
    avg_epoch_loss = epoch_loss / num_batches
    print(f"Average epoch loss: {avg_epoch_loss:>7f}")
    print(f"PER on training data: {PER:>7f}")
    with open('training_output.txt', 'a') as file:
        file.write(f"Average epoch loss: {avg_epoch_loss:>7f}\n")
        file.write(f"PER on training data: {PER:>7f}\n")
    lr_scheduler.step()
    print("Calculating test loss...")
    
    finalModel.eval()
    total_loss = 0.0
    num_batches = 0
    per_numerator = 0
    per_denominator = 0
    
    with torch.no_grad():
        for batch in test_loader:
            articulators = batch["articulator"].to(device)
            #videos = batch["video"]
            audio = batch["audio"]
            targets = batch["phonemes"].to(device)
            target_lengths = batch["phoneme_lengths"].to(device)

            #Acquire audio features
            audio_values = audio_processor(audio[:,0,:].to(device), sampling_rate = 16000, return_tensors="pt", padding=True).to(device)
            audio_features = audio_features_model(audio_values['input_values'][0,:,:]).last_hidden_state #Shape: [batch_size, time_steps, 1024]

            log_probs = finalModel(audio_features, articulators, 1024, x_min, x_max) #Shape: [batch_size, 250, 40]
            
            # Prepare input and target lengths (all sequences are length 250 in your case)
            input_lengths = torch.full(size=(batch_length,), fill_value=250, dtype=torch.long)
            #target_lengths = torch.randint(low=1, high=250, size=(batch_length,), dtype=torch.long)
            #target_lengths = torch.full((8,), 100, dtype=torch.long)

            loss = loss_function(log_probs.transpose(0, 1),  # CTC expects [seq_len, batch, num_classes]
                targets, 
                input_lengths, 
                target_lengths)
            
            total_loss += loss.item()
            num_batches += 1

            val_per.add_batch(log_probs, targets)

    avg_loss = total_loss / num_batches
    print(f"Average loss of testing dataset: {avg_loss:>7f}")
    with open('training_output.txt', 'a') as file:
        file.write(f"Average loss of testing dataset: {avg_loss:>7f}\n")

    PER, edits, total = val_per.compute()
    if PER < best_val_per:
        val_per.save('val_pred.json')
        best_val_per = PER
        torch.save(finalModel.state_dict(), "best_model.pth")
        print(f"Saved best model at epoch {t+1} with val_loss: {avg_loss:.4f}")
    val_per.reset()
    print(f"PER on testing data: {PER:>7f}")
    with open('training_output.txt', 'a') as file:
        file.write(f"PER on testing data: {PER:>7f}\n")
