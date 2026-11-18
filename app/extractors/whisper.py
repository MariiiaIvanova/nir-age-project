import torch
from torch import nn
import whisper


class WhisperTiny(nn.Module):
    output_dim = 384

    def __init__(self, weights):
        super().__init__()
        model = whisper.load_model(str(weights), device="cpu")
        if model.dims.n_audio_state != 384 or model.dims.n_audio_layer != 4:
            raise ValueError("Expected a Whisper tiny checkpoint")
        self.encoder = model.encoder

    def forward(self, wave):
        # Whisper clips the dynamic range relative to each recording's maximum.
        mel = torch.stack([whisper.log_mel_spectrogram(item) for item in wave])
        encoder = self.encoder
        x = nn.functional.gelu(encoder.conv1(mel))
        x = nn.functional.gelu(encoder.conv2(x)).permute(0, 2, 1)
        x = (x + encoder.positional_embedding[:x.shape[1]]).to(x.dtype)
        for block in encoder.blocks:
            x = block(x)
        return encoder.ln_post(x).transpose(1, 2)
