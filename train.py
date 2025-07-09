#General libraries needed for model training/evaluation
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from itertools import groupby
import Levenshtein
from dataloader import VideoAudioPhonemeDataset
from models import AV_Conformer, rtMRI_Encoder
from utils import get_dataset
from transformers import Wav2Vec2Processor, Wav2Vec2Model

#Libraries needed for video encoding
from torchvision.models.optical_flow import raft_large


video_directory = "/data1/open_data/single_spk_corpus" # Define video directory path
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu") #Define device
batch_length = 1 #Define batch length
phoneme_dictionary = ['', 'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW', 'B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N', 'NG', 'P', 'R', 'S', 'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH']
epochs = 100

# Define the DataLoader
train_loader, test_loader = get_dataset(video_directory, batch_length)

#Prepare models for audio feature acquisition
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
audio_features_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft").to(device)

#Prepare models for video feature acquisition
video_model = raft_large(pretrained=True, progress=False).to(device)
video_model = video_model.eval()

#Initialize our training parameters
finalModel = rtMRI_Encoder(modality="ai").cuda()
optimizer = torch.optim.Adam(finalModel.parameters(), lr=1e-3, weight_decay=1e-4)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=20, gamma=0.9)
loss_function = nn.CTCLoss(blank=0, zero_infinity=True)

#Setup for model evaluation
def eval_step(engine, batch):
    return batch

for t in range(epochs):
    epoch_loss = 0.0
    num_batches = 0
    print(f"Epoch {t+1}\n-------------------------------")
    with open('training_output.txt', 'a') as file:
        file.write(f"Epoch {t+1}\n-------------------------------\n")
    for index, batch in enumerate(train_loader):
        finalModel.train()
        optimizer.zero_grad()
        videos = batch["video"].cuda() #Shape: [batch_size, num_frames, H, W, 3]
        audio = batch["audio"].cuda() #Shape: [batch_size, num_channels, num_samples]
        # assert False, f"batch phonemes.shape {batch[/'phonemes/'].shape}"
        targets = batch["phonemes"].cuda() #Shape: [batch_size, time_steps]
        # assert False, f"batch phonemes.shape {targets.shape}"
        target_lengths = batch["phoneme_lengths"].to(device)
        # assert False, f"target_lengths.shape {target_lengths.shape} is : {target_lengths}"
        #Acquire audio features
        audio_values = processor(audio[:,0,:].to(device), sampling_rate = 16000, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            #audio_features = audio_features_model(audio_inputs[0])
            audio_features = audio_features_model(audio_values['input_values'][0,:,:]).last_hidden_state #Shape: [batch_size, time_steps, 1024]
        
        #Acquire video features
        # list_of_flows = torch.rand(batch_length, 250, 2, 128, 128).to(device)
        # for i in range(batch_length):
        #     with torch.no_grad():
        #         list_of_flows[i,:,:,:,:] = video_model(videos[i,::2, :, :, :].permute(0, 3, 1, 2).to(device), videos[i,1::2, :, :, :].permute(0, 3, 1, 2).to(device))[-1]
        # list_of_flows = torch.flatten(list_of_flows, start_dim=2)

        log_probs = finalModel(audio_features, videos[:,::2,:], 1024) #Shape: [batch_size, 250, 40]
        # assert False, f"videos shape {videos.shape}"
        # Prepare input and target lengths (all sequences are length 250 in your case)
        input_lengths = torch.full(size=(batch_length,), fill_value=250, dtype=torch.long)
        #target_lengths = torch.randint(low=1, high=250, size=(batch_length,), dtype=torch.long)
        # assert False, f"targets.shape {targets.shape} and log_probs.shape {log_probs.shape} and target_lengths.shape {target_lengths}"
        loss = loss_function(log_probs.transpose(0, 1),  # CTC expects [seq_len, batch, num_classes]
                targets, 
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
    with open('training_output.txt', 'a') as file:
        file.write(f"Average epoch loss: {avg_epoch_loss:>7f}\n")
    print("Calculating test loss...")
    lr_scheduler.step()
    
    finalModel.eval()
    total_loss = 0.0
    num_batches = 0
    per_numerator = 0
    per_denominator = 0
    
    with torch.no_grad():
        for batch in test_loader:
            videos = batch["video"]
            audio = batch["audio"]
            targets = batch["phonemes"].to(device)
            target_lengths = batch["phoneme_lengths"].to(device)

            #Acquire audio features
            audio_values = processor(audio[:,0,:].to(device), sampling_rate = 16000, return_tensors="pt", padding=True).to(device)
            audio_features = audio_features_model(audio_values['input_values'][0,:,:]).last_hidden_state #Shape: [batch_size, time_steps, 1024]

            #Acquire video features
            list_of_flows = torch.rand(batch_length, 250, 2, 128, 128).to(device)
            for i in range(batch_length):
                with torch.no_grad():
                    list_of_flows[i,:,:,:,:] = video_model(videos[i,::2, :, :, :].permute(0, 3, 1, 2).to(device), videos[i,1::2, :, :, :].permute(0, 3, 1, 2).to(device))[-1]
            list_of_flows = torch.flatten(list_of_flows, start_dim=2)

            log_probs = finalModel(audio_features, list_of_flows, 1024) #Shape: [batch_size, 250, 40]
            
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

            #Calculation of phoneme error rate (PER)
            for i in range(batch_length):
                ref = []
                hyp = []
                for j in range(250):
                    if torch.argmax(log_probs, dim=-1)[i,j].item() != 0:
                        hyp.append(torch.argmax(log_probs, dim=-1)[i,j].item())
                    if targets[i,j].item() != 0:
                        ref.append(targets[i,j].item())
                hyp = [key for key, _ in groupby(hyp)]
                per_numerator = per_numerator + Levenshtein.distance(ref, hyp)
                per_denominator = per_denominator + len(ref)
                

    avg_loss = total_loss / num_batches
    print(f"Average loss of testing dataset: {avg_loss:>7f}")
    with open('training_output.txt', 'a') as file:
        file.write(f"Average loss of testing dataset: {avg_loss:>7f}\n")

    PER = per_numerator/per_denominator
    print(f"PER on testing data: {PER:>7f}")
    with open('training_output.txt', 'a') as file:
        file.write(f"PER on testing data: {PER:>7f}\n")
    with torch.no_grad():
        audio, sr = torchaudio.load("/data1/jaypark/single_spk_corpus/five_second_audio/usc_s1_66_0554.wav")
        paddedAudio = torch.zeros(1, 80320)
        paddedAudio[:,0:80000] = audio
        paddedAudio = paddedAudio.to(device)
        audio_values = processor(paddedAudio, sampling_rate = 16000, return_tensors="pt", padding=True).to(device)
        audio_features = audio_features_model(audio_values['input_values'][0,:,:]).last_hidden_state
        log_probs = finalModel(audio_features, torch.zeros(1, 1), 1024)
        predictedtokens = ""
        for i in range(250):
            if phoneme_dictionary[torch.argmax(log_probs, dim=-1)[0,i]] == "":
                predictedtokens = predictedtokens + "  "
            else:
                predictedtokens = predictedtokens + phoneme_dictionary[torch.argmax(log_probs, dim=-1)[0,i]] + " "
        print(predictedtokens)
        with open('training_output.txt', 'a') as file:
            file.write(predictedtokens+'\n')
