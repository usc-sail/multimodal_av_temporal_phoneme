#General libraries needed for model training/evaluation
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.utils import download_asset
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple
from vision_transformer import VisionTransformer, partial
from einops import rearrange

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
        self.vid_dim = 32768
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

        #Define relu activation function
        self.relu = nn.ReLU()

        # Final classification layer
        self.classifier = nn.Linear(128, 40)

    def forward(self, audio_features, video_features, feat_lengths):
    
        if self.modality == 'audio':
            x = audio_features #Shape: [batch_size, 250, 1024]
        elif self.modality == 'video':
            x = video_features #Shape: [batch_size, 250, 19200]
        else:  # multimodal
            x = torch.cat([audio_features, video_features], dim=-1)
        batch_size = x.shape[0]

        lengths = torch.full((batch_size,), 250, dtype=torch.long).to(self.device)
        #lengths = feat_lengths

        # Pass through Conformer
        #x, reps, att, _ = self.conformer(x, lengths)  # Shape: [batch_size, seq_len, conformer_dim]
        #print(reps.shape)

        # Decode with LSTM
        x, _ = self.decoder(x)  # Shape: [batch_size, seq_len, lstm_dim]

        # Classification
        logits = self.classifier(x)  # Shape: [batch_size, 250, 40]

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        return log_probs
    
class rtMRI_Encoder(nn.Module):
    def __init__(self, 
                 modality='a', 
                 feats='base', 
                 patch_size=8, 
                 depth=12, 
                 num_heads=6, 
                 mlp_ratio=4,
                 sequence_model='lstm'):
        
        super().__init__()

        # Modality type: 'audio', 'video', or 'multimodal'
        self.audio_dim = 1024
        self.vid_dim = 384
        self.modality = modality

        if modality == 'a': # audio only
            input_dim = self.audio_dim 
        elif modality == 'ai': # audio + image
            input_dim = self.vid_dim
            # self.motion_model = VisionTransformer(in_chans=2, patch_size=patch_size, embed_dim=384, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
            self.visual_model = VisionTransformer(in_chans=3, patch_size=patch_size, embed_dim=self.vid_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        else:
            input_dim = self.audio_dim + self.vid_dim
    
        # LSTM decoder
        self.sequence_model = nn.LSTM(
            input_size=input_dim,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
        ) if sequence_model == 'lstm' else None

        #Define relu activation function
        self.relu = nn.ReLU()

        # Final classification layer
        self.classifier = nn.Linear(128, 40)

    def forward(self, audio_features, video_features, feat_lengths):
        """
        args:
            audio_features: Tensor of shape [batch_size, 250, 1024] 
            video_features: Tensor of shape [batch_size, 250, 3, 128, 128]
            feat_lengths: Tensor of shape [batch_size] containing lengths of each sequence
            returns:
                log_probs: Tensor of shape [batch_size, 250, 40] containing log probabilities for each class
        """
        if self.modality == 'a':
            x = audio_features #Shape: [batch_size, 250, 1024]
        elif self.modality == 'ai':
            b = video_features.shape[0]
            t = video_features.shape[1]
            ai_feas = rearrange(video_features, 'b t c h w -> (b t) c h w')  # Rearrange to (B*T, C, H, W)
            x = self.visual_model(ai_feas) #Shape: [batch_size* 250, 1024]
            x = rearrange(x, '(b t) e -> b t e', b=b, t=t)  #Shape: [batch_size, 250, 1024]
        else:  # multimodal
            x = torch.cat([audio_features, video_features], dim=-1)

        batch_size = x.shape[0]

        # lengths = torch.full((batch_size,), 250, dtype=torch.long).to(self.device)
        #lengths = feat_lengths

        # Pass through Conformer
        #x, reps, att, _ = self.conformer(x, lengths)  # Shape: [batch_size, seq_len, conformer_dim]
        #print(reps.shape)

        # Decode with LSTM
        x, _ = self.sequence_model(x)  # Shape: [batch_size, seq_len, lstm_dim]

        # Classification
        logits = self.classifier(x)  # Shape: [batch_size, 250, 40]

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        return log_probs
    
class Articulator_Encoder(nn.Module):
    def __init__(self, 
                 modality='audio', 
                 feats='base', 
                 patch_size=8, 
                 depth=12, 
                 num_heads=6, 
                 mlp_ratio=4):
        
        super().__init__()

        # Modality type: 'audio', 'video', or 'multimodal'
        self.audio_dim = 1024
        self.vid_dim = 64
        self.modality = modality

        if modality == 'audio': # audio only
            input_dim = self.audio_dim 
        elif modality == 'articulator': # articulator only
            input_dim = self.vid_dim
            # self.motion_model = VisionTransformer(in_chans=2, patch_size=patch_size, embed_dim=384, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        else:
            input_dim = self.audio_dim + self.vid_dim

        # Convolution layers
        self.conv1 = nn.Conv1d(6, 64, kernel_size=17, stride=1, padding=8)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, self.vid_dim, kernel_size=17, stride=1, padding=8)
        self.batchnorm2 = nn.BatchNorm1d(64)
        #self.conv3 = nn.Conv1d(64, self.vid_dim, kernel_size=19, stride=1, padding=9)

        # LSTM decoder
        self.sequence_model = nn.LSTM(
            input_size=input_dim,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
        )

        # Define relu activation function
        self.relu = nn.ReLU()
        
        # Final classification layer
        self.classifier = nn.Linear(256, 40)

    def forward(self, audio_features, video_features, feat_lengths, x_min, x_max):
        """
        args:
            audio_features: Tensor of shape [batch_size, 250, 1024] 
            video_features: Tensor of shape [batch_size, 250, 3, 128, 128]
            feat_lengths: Tensor of shape [batch_size] containing lengths of each sequence
            returns:
                log_probs: Tensor of shape [batch_size, 250, 40] containing log probabilities for each class
        """
        if self.modality == 'audio':
            x = audio_features #Shape: [batch_size, 250, 1024]
        elif self.modality == 'articulator':
            x = video_features # [B, 500, 6]
            #Min and max of channels:
            #0: 0.22690388071665196, 3.5341736387660894
            #1: 0.11955651562837223, 4.626061521150712
            #2: 0.17564414476030232, 2.986669440527201
            #3: 0.24438938867013488, 2.153838267333077
            #4: 0.473685007464924, 2.6318475522713562
            #5: 0.26225585780008864, 2.706900564995662
            x = (x-x_min)/(x_max-x_min) # [B, 500, 6]
            x = rearrange(x, 'b t d -> b d t') #[B, 6, 500]
            x = self.conv1(x) #[B, 64, 500]
            x = self.batchnorm1(x)
            x = self.relu(x)
            x = self.pool1(x) #[B, 64, 250]
            x = self.conv2(x) #[B, 64, 250]
            x = self.batchnorm2(x)
            x = self.relu(x)
            #x = self.conv3(x)
            #x = self.relu(x)
            x = rearrange(x, 'b d t -> b t d') # [B, 250, 64]
            #ai_feas = rearrange(x, 'b t c h w -> (b t) c h w')  # Rearrange to (B*T, C, H, W)
            #x = self.visual_model(ai_feas)
        else:  # multimodal
            x = video_features
            #x_min, x_max = x.min(), x.max()
            #x_min, x_max = 0.11955651562837223, 4.626061521150712
            x = (x-x_min)/(x_max-x_min)
            x = rearrange(x, 'b t d -> b d t')
            x = self.relu(self.conv1(x))
            x = rearrange(x, 'b d t -> b t d')
            x = self.mambaModel(x)
            x = torch.cat([audio_features, x], dim=-1)

        batch_size = x.shape[0]

        # lengths = torch.full((batch_size,), 250, dtype=torch.long).to(self.device)
        #lengths = feat_lengths

        # Pass through Conformer
        #x, reps, att, _ = self.conformer(x, lengths)  # Shape: [batch_size, seq_len, conformer_dim]
        #print(reps.shape)

        # Decode with LSTM
        x, _ = self.sequence_model(x)  # [B, 250, 256]

        # Final Classification
        logits = self.classifier(x)  # [B, 250, 40]

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        return log_probs
