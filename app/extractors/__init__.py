"""Extractor selection. Each adapter accepts [B,T] and returns [B,D,T']."""


def build_extractor(name, weights):
    if name == 'whisper_tiny':
        from .whisper import WhisperTiny
        return WhisperTiny(weights)
    if name == 'wavlm_base':
        from .wavlm import WavLMBase
        return WavLMBase(weights)
    raise NotImplementedError(f'Extractor {name!r} has no implementation yet')
