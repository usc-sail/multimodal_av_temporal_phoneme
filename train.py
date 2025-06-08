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

#Libraries needed for wav2vec2-lv-60-espeak-cv-ft
from transformers import Wav2Vec2Processor, Wav2Vec2Model

def _lengths_to_padding_mask(lengths: torch.Tensor) -> torch.Tensor:
    batch_size = lengths.shape[0]
    max_length = int(torch.max(lengths).item())
    padding_mask = torch.arange(max_length, device=lengths.device, dtype=lengths.dtype).expand(
        batch_size, max_length
    ) >= lengths.unsqueeze(1)
    return padding_mask


class _ConvolutionModule(torch.nn.Module):
    r"""Conformer convolution module.

    Args:
        input_dim (int): input dimension.
        num_channels (int): number of depthwise convolution layer input channels.
        depthwise_kernel_size (int): kernel size of depthwise convolution layer.
        dropout (float, optional): dropout probability. (Default: 0.0)
        bias (bool, optional): indicates whether to add bias term to each convolution layer. (Default: ``False``)
        use_group_norm (bool, optional): use GroupNorm rather than BatchNorm. (Default: ``False``)
    """

    def __init__(
        self,
        input_dim: int,
        num_channels: int,
        depthwise_kernel_size: int,
        dropout: float = 0.0,
        bias: bool = False,
        use_group_norm: bool = False,
    ) -> None:
        super().__init__()
        if (depthwise_kernel_size - 1) % 2 != 0:
            raise ValueError("depthwise_kernel_size must be odd to achieve 'SAME' padding.")
        self.layer_norm = torch.nn.LayerNorm(input_dim)
        self.sequential = torch.nn.Sequential(
            torch.nn.Conv1d(
                input_dim,
                2 * num_channels,
                1,
                stride=1,
                padding=0,
                bias=bias,
            ),
            torch.nn.GLU(dim=1),
            torch.nn.Conv1d(
                num_channels,
                num_channels,
                depthwise_kernel_size,
                stride=1,
                padding=(depthwise_kernel_size - 1) // 2,
                groups=num_channels,
                bias=bias,
            ),
            torch.nn.GroupNorm(num_groups=1, num_channels=num_channels)
            if use_group_norm
            else torch.nn.BatchNorm1d(num_channels),
            torch.nn.SiLU(),
            torch.nn.Conv1d(
                num_channels,
                input_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
            ),
            torch.nn.Dropout(dropout),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            input (torch.Tensor): with shape `(B, T, D)`.

        Returns:
            torch.Tensor: output, with shape `(B, T, D)`.
        """
        x = self.layer_norm(input)
        x = x.transpose(1, 2)
        x = self.sequential(x)
        return x.transpose(1, 2)


class _FeedForwardModule(torch.nn.Module):
    r"""Positionwise feed forward layer.

    Args:
        input_dim (int): input dimension.
        hidden_dim (int): hidden dimension.
        dropout (float, optional): dropout probability. (Default: 0.0)
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.sequential = torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim),
            torch.nn.Linear(input_dim, hidden_dim, bias=True),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, input_dim, bias=True),
            torch.nn.Dropout(dropout),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            input (torch.Tensor): with shape `(*, D)`.

        Returns:
            torch.Tensor: output, with shape `(*, D)`.
        """
        return self.sequential(input)

class ConformerLayer(torch.nn.Module):
    r"""Conformer layer that constitutes Conformer.

    Args:
        input_dim (int): input dimension.
        ffn_dim (int): hidden layer dimension of feedforward network.
        num_attention_heads (int): number of attention heads.
        depthwise_conv_kernel_size (int): kernel size of depthwise convolution layer.
        dropout (float, optional): dropout probability. (Default: 0.0)
        use_group_norm (bool, optional): use ``GroupNorm`` rather than ``BatchNorm1d``
            in the convolution module. (Default: ``False``)
        convolution_first (bool, optional): apply the convolution module ahead of
            the attention module. (Default: ``False``)
    """

    def __init__(
        self,
        input_dim: int,
        ffn_dim: int,
        num_attention_heads: int,
        depthwise_conv_kernel_size: int,
        dropout: float = 0.0,
        use_group_norm: bool = False,
        convolution_first: bool = False,
    ) -> None:
        super().__init__()

        self.ffn1 = _FeedForwardModule(input_dim, ffn_dim, dropout=dropout)

        self.self_attn_layer_norm = torch.nn.LayerNorm(input_dim)
        self.self_attn = torch.nn.MultiheadAttention(input_dim, num_attention_heads, dropout=dropout)
        self.self_attn_dropout = torch.nn.Dropout(dropout)

        self.conv_module = _ConvolutionModule(
            input_dim=input_dim,
            num_channels=input_dim,
            depthwise_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
            bias=True,
            use_group_norm=use_group_norm,
        )

        self.ffn2 = _FeedForwardModule(input_dim, ffn_dim, dropout=dropout)
        self.final_layer_norm = torch.nn.LayerNorm(input_dim)
        self.convolution_first = convolution_first

    def _apply_convolution(self, input: torch.Tensor) -> torch.Tensor:
        residual = input
        input = input.transpose(0, 1)
        input = self.conv_module(input)
        input = input.transpose(0, 1)
        input = residual + input
        return input

    def forward(self, input: torch.Tensor, key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        r"""
        Args:
            input (torch.Tensor): input, with shape `(T, B, D)`.
            key_padding_mask (torch.Tensor or None): key padding mask to use in self attention layer.

        Returns:
            torch.Tensor: output, with shape `(T, B, D)`.
        """
        residual = input
        x = self.ffn1(input)
        x = x * 0.5 + residual

        if self.convolution_first:
            x = self._apply_convolution(x)

        residual = x
        x = self.self_attn_layer_norm(x)
        x, att = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=True,
        )
        x = self.self_attn_dropout(x)
        x = x + residual

        if not self.convolution_first:
            x = self._apply_convolution(x)

        residual = x
        x = self.ffn2(x)
        x = x * 0.5 + residual
        inter_rep = x.clone()

        x = self.final_layer_norm(x)
        return x, inter_rep, att

class Conformer(torch.nn.Module):
    r"""Conformer architecture introduced in
    *Conformer: Convolution-augmented Transformer for Speech Recognition*
    :cite:`gulati2020conformer`.

    Args:
        input_dim (int): input dimension.
        num_heads (int): number of attention heads in each Conformer layer.
        ffn_dim (int): hidden layer dimension of feedforward networks.
        num_layers (int): number of Conformer layers to instantiate.
        depthwise_conv_kernel_size (int): kernel size of each Conformer layer's depthwise convolution layer.
        dropout (float, optional): dropout probability. (Default: 0.0)
        use_group_norm (bool, optional): use ``GroupNorm`` rather than ``BatchNorm1d``
            in the convolution module. (Default: ``False``)
        convolution_first (bool, optional): apply the convolution module ahead of
            the attention module. (Default: ``False``)

    Examples:
        >>> conformer = Conformer(
        >>>     input_dim=80,
        >>>     num_heads=4,
        >>>     ffn_dim=128,
        >>>     num_layers=4,
        >>>     depthwise_conv_kernel_size=31,
        >>> )
        >>> lengths = torch.randint(1, 400, (10,))  # (batch,)
        >>> input = torch.rand(10, int(lengths.max()), input_dim)  # (batch, num_frames, input_dim)
        >>> output = conformer(input, lengths)
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int,
        depthwise_conv_kernel_size: int,
        dropout: float = 0.0,
        use_group_norm: bool = False,
        convolution_first: bool = False,
    ):
        super().__init__()

        self.conformer_layers = torch.nn.ModuleList(
            [
                ConformerLayer(
                    input_dim,
                    ffn_dim,
                    num_heads,
                    depthwise_conv_kernel_size,
                    dropout=dropout,
                    use_group_norm=use_group_norm,
                    convolution_first=convolution_first,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, input: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Args:
            input (torch.Tensor): with shape `(B, T, input_dim)`.
            lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``input``.

        Returns:
            (torch.Tensor, torch.Tensor)
                torch.Tensor
                    output frames, with shape `(B, T, input_dim)`
                torch.Tensor
                    output lengths, with shape `(B,)` and i-th element representing
                    number of valid frames for i-th batch element in output frames.
        """
        encoder_padding_mask = _lengths_to_padding_mask(lengths)

        x = input.transpose(0, 1)
        for layer in self.conformer_layers:
            x, inter_rep, att = layer(x, encoder_padding_mask)
        return x.transpose(0, 1), inter_rep.transpose(0, 1), att, lengths

class AV_Conformer(nn.Module):
    def __init__(self, device, modality='multimodal', feats='base', num_heads=4, num_layers=3):
        super().__init__()

        # Modality type: 'audio', 'video', or 'multimodal'
        self.audio_dim = 1024
        self.vid_dim = 1024
        self.modality = modality
        self.device = device

        if modality == 'audio':
            input_dim = self.audio_dim
        elif modality == 'video':
            input_dim = self.vid_dim
        else:
            input_dim = self.audio_dim + self.vid_dim
    
        self.conformer = Conformer(
            input_dim=input_dim,
            num_heads=num_heads,
            ffn_dim=256,
            num_layers=num_layers,
            depthwise_conv_kernel_size=31,
            dropout=0.3
        ).to(device)

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

        lengths = torch.full((batch_size,), 250, dtype=torch.long).to(self.device)
        #lengths = feat_lengths

        # Pass through Conformer
        x, reps, att, _ = self.conformer(x, lengths)  # Shape: [batch_size, seq_len, conformer_dim]
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
audio_features_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft").to(device)

#Prepare models for video feature acquisition

#Initialize our training parameters
finalModel = AV_Conformer(device=device, modality="audio", num_heads=4, num_layers=3).to(device)
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

        #Acquire audio features
        audio_values = processor(audio[:,0,:], sampling_rate = 16000, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            #audio_features = audio_features_model(audio_inputs[0])
            audio_features = audio_features_model(audio_values['input_values'][0,:,:]).last_hidden_state #Shape: [batch_size, time_steps, 1024]
        #Acquire video features
        
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

            #Acquire audio features
            audio_values = processor(audio[:,0,:], sampling_rate = 16000, return_tensors="pt", padding=True).to(device)
            audio_features = audio_features_model(audio_values['input_values'][0,:,:]).last_hidden_state #Shape: [batch_size, time_steps, 1024]

            #Acquire video features

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
            for i in range(batch_length * 250):
                if tokens.flatten()[i] != 0:
                    total_tokens += 1
                    if torch.argmax(log_probs, dim=-1).flatten()[i] != tokens.flatten()[i]:
                        incorrect_tokens += 1

    avg_loss = total_loss / num_batches
    print(f"Average loss of testing dataset: {avg_loss:>7f}")
    token_error_rate = float(incorrect_tokens)/float(total_tokens)
    print(f"Token error rate on testing dataset: {token_error_rate:>7f}\n")
