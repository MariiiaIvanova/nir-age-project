"""Real-audio forward/backward and memory check for a selected extractor.

Does not create an experiment or modify pretrained weights. Example:
python tests/check_extractor.py --extractor wavlm_base --weights weights/wavlm_base.pth
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from audio_data import AgeDataset
from model import AgeRegressor
from train import update


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--extractor', required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--splits', type=Path, default=Path('output/native/validation_seed42'))
    args = p.parse_args()
    random.seed(42)
    torch.manual_seed(42)
    ds = AgeDataset(args.splits / 'train.txt', training=True, seconds=4)
    indices = torch.randperm(len(ds))[:args.batch_size].tolist()
    batch = torch.utils.data.default_collate([ds[i] for i in indices])
    assert batch[0].shape == (args.batch_size, 64000)
    assert batch[1].unique().numel() > 1
    model = AgeRegressor(args.extractor, args.weights, sum(r[-1] for r in ds.rows)/len(ds)).cuda()
    assert all(p.requires_grad for p in model.encoder.parameters())
    shape = []
    def capture(module, inputs, output):
        shape[:] = list(output.shape)
    hook = model.encoder.register_forward_hook(capture)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    torch.cuda.reset_peak_memory_stats()
    for step in range(2):
        model.train()
        loss, norm = update(model, optimizer, batch, 'cuda', 'ccc')
        assert norm > 0
        encoder_norm = sum(p.grad.detach().square().sum().item() for p in model.encoder.parameters() if p.grad is not None)**.5
        assert encoder_norm > 0
        assert shape[:2] == [args.batch_size, model.encoder.output_dim]
        print(json.dumps(dict(step=step, loss=loss, gradient_norm=norm, encoder_gradient_norm=encoder_norm, features_shape=shape)), flush=True)
    model.eval()
    with torch.inference_mode():
        first=model(batch[0][:2].cuda()); second=model(batch[0][:2].cuda())
        assert torch.equal(first, second)
        assert torch.isfinite(first).all()
    hook.remove()
    print(json.dumps(dict(status='PASS', extractor=args.extractor, batch_size=args.batch_size,
                          peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                          peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2)), flush=True)


if __name__ == '__main__':
    main()
