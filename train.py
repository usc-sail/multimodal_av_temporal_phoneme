#General libraries needed for model training/evaluation
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
# from torch.utils.data import Dataset, DataLoader
from itertools import groupby
import Levenshtein
from dataloader import VideoAudioPhonemeDataset
from models import AV_Conformer, rtMRI_Encoder
from utils import get_dataset, PhonemeErrorRate
from transformers import Wav2Vec2Processor, Wav2Vec2Model
#Libraries needed for video encoding
from torchvision.models.optical_flow import raft_large

# MAIN SETTINGS
video_directory = "/data1/open_data/single_spk_corpus" # Define video directory path
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") #Define device
batch_length = 1 #Define batch length
phoneme_dictionary = ['', 'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW', 'B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N', 'NG', 'P', 'R', 'S', 'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH']
epochs = 100

# Define the DataLoader
train_per = PhonemeErrorRate(phoneme_dictionary, blank_id=0)
val_per = PhonemeErrorRate(phoneme_dictionary, blank_id=0)
train_loader, test_loader = get_dataset(video_directory, batch_length, modality="f") #Use only optical flow modality for now

#Initialize our training parameters
finalModel = rtMRI_Encoder(modality="f").cuda()

optimizer = torch.optim.Adam(finalModel.parameters(), lr=1e-3, weight_decay=1e-4)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=20, gamma=0.9)
loss_function = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

#Setup for model evaluation
def eval_step(engine, batch):
    return batch
best_val_per = float('inf')

for t in range(epochs):
    epoch_loss = 0.0
    num_batches = 0
    print(f"Epoch {t+1}\n-------------------------------")
    finalModel.train()

    for index, batch in enumerate(train_loader):
        optimizer.zero_grad()

        targets = batch["phonemes"].cuda() #Shape: [batch_size, time_steps]
        target_lengths = batch["phoneme_lengths"].cuda() #Shape: [batch_size]

        log_probs = finalModel(batch, 1024) #Shape: [batch_size, 250, 40]
        # Prepare input and target lengths (all sequences are length 250 in your case)
        input_lengths = torch.full(size=(batch_length,), fill_value=250, dtype=torch.long)
        loss = loss_function(log_probs.transpose(0, 1),  # CTC expects [seq_len, batch, num_classes]
                targets, 
                input_lengths, 
                target_lengths)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(finalModel.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

        train_per.add_batch(log_probs, targets)
        if index % 100 == 0:
            loss, current, size = loss.item(), index * batch_length, len(train_loader.dataset)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    PER, edits, total = train_per.compute()
    train_per.save('train_pred.json')
    train_per.reset()
    avg_epoch_loss = epoch_loss / num_batches
    print(f"PER on Training data: {PER:>7f} and Average epoch loss: {avg_epoch_loss:>7f}")

    lr_scheduler.step()
    
    #Evaluate the model on the EVAL set
    finalModel.eval()
    total_loss = 0.0
    num_batches = 0
    per_numerator = 0
    per_denominator = 0
    
    with torch.no_grad():
        for batch in test_loader:
            targets, target_lengths = batch["phonemes"].cuda(), batch["phoneme_lengths"].cuda()#Shape: [batch_size, num_frames, H, W, 3], [batch_size, num_channels, num_samples], [batch_size, time_steps], [batch_size], [batch_size, num_frames, H, W, 2]

            # log_probs = finalModel(audio_features, list_of_flows, 1024) #Shape: [batch_size, 250, 40]
            log_probs = finalModel(batch,  1024) #Shape: [batch_size, 250, 40]

            # Prepare input and target lengths (all sequences are length 250 in your case)
            input_lengths = torch.full(size=(batch_length,), fill_value=250, dtype=torch.long)
            
            loss = loss_function(log_probs.transpose(0, 1),  # CTC expects [seq_len, batch, num_classes]
                targets, 
                input_lengths, 
                target_lengths)
            total_loss += loss.item()
            num_batches += 1

            val_per.add_batch(log_probs, targets)

    avg_loss = total_loss / num_batches
    PER, edits, total = val_per.compute()
    val_per.save('val_pred.json')
    if val_per < best_val_per:
        best_val_per = val_per
        torch.save(finalModel.state_dict(), "best_model.pth")
        print(f"✅ Saved best model at epoch {t} with val_loss: {avg_loss:.4f}")
    val_per.reset()
    print(f"PER on testing data: {PER:>7f} and Average loss of testing dataset: {avg_loss:>7f}")

