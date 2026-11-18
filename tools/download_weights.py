"""Download official pretrained weights explicitly; training stays offline."""
import argparse
import time
import urllib.request
from pathlib import Path


def download_wavlm(directory):
    import torch
    from torchaudio.models import wavlm_base
    url = 'https://download.pytorch.org/torchaudio/models/wavlm_base.pth'
    expected = 377603342  # official WAVLM_BASE file, verified by HTTP HEAD
    target = directory / 'wavlm_base.pth'
    partial = directory / 'wavlm_base.pth.resume'
    if target.exists() and target.stat().st_size != expected:
        if partial.exists():
            raise RuntimeError('Both an incomplete target and resume file exist; inspect them first')
        target.rename(partial)
    if not target.exists():
        failures = 0
        while not partial.exists() or partial.stat().st_size < expected:
            start = partial.stat().st_size if partial.exists() else 0
            end = min(start + 4 * 1024 * 1024, expected) - 1
            request = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}'})
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{expected}':
                        raise RuntimeError('Server did not return the requested byte range')
                    remaining = end - start + 1
                    with partial.open('ab') as out:
                        while remaining:
                            data = response.read(min(65536, remaining))
                            if not data:
                                raise IOError('Download interrupted before end of range')
                            out.write(data)
                            remaining -= len(data)
                failures = 0
            except (OSError, RuntimeError) as exc:
                current = partial.stat().st_size if partial.exists() else 0
                failures = failures + 1 if current == start else 0
                print(f'Retrying from byte {current}: {exc}', flush=True)
                if failures >= 5:
                    raise
                time.sleep(5)
            size = partial.stat().st_size if partial.exists() else 0
            print(f'WavLM download: {size}/{expected} bytes ({100*size/expected:.1f}%)', flush=True)
    candidate = target if target.exists() else partial
    if candidate.stat().st_size != expected:
        raise RuntimeError('Wrong checkpoint size')
    # Publish only after archive and parameter compatibility checks.
    state = torch.load(candidate, map_location='cpu', weights_only=True)
    model = wavlm_base(encoder_ff_interm_dropout=0.0, encoder_layer_drop=0.05)
    model.load_state_dict(state, strict=True)
    if candidate != target:
        candidate.replace(target)
    print('Official WavLM checkpoint validated', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--extractor', choices=['whisper_tiny', 'wavlm_base'], required=True)
    p.add_argument('--directory', type=Path, default=Path('weights'))
    args = p.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    if args.extractor == 'whisper_tiny':
        import whisper
        whisper.load_model('tiny', device='cpu', download_root=str(args.directory))
    else:
        download_wavlm(args.directory)
    print(f'Official {args.extractor} weights ready in {args.directory}')


if __name__ == '__main__':
    main()
