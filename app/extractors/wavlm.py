"""Official torchaudio WAVLM_BASE, final encoder layer, full fine-tuning.

Matches torchaudio 2.4.1 WAVLM_BASE bundle settings (including dropout).
Base does not require additional waveform layer normalization.
"""
import torch
from torch import nn
from torchaudio.models import wavlm_base


class WavLMBase(nn.Module):
    output_dim = 768

    def __init__(self, weights):
        super().__init__()
        self.encoder = wavlm_base(
            encoder_projection_dropout=0.1,
            encoder_attention_dropout=0.1,
            encoder_ff_interm_dropout=0.0,
            encoder_dropout=0.1,
            encoder_layer_drop=0.05,
            aux_num_out=None,
        )
        state = torch.load(weights, map_location='cpu', weights_only=True)
        self.encoder.load_state_dict(state, strict=True)

    def forward(self, wave):
        features, _ = self.encoder(wave)
        return features.transpose(1, 2)
