"""Create a fixed speaker-disjoint validation split without decoding audio."""
import argparse
import hashlib
import json
import random
import shutil
import statistics
from collections import defaultdict
from pathlib import Path

CORPORA = ('timit', 'nnces', 'agevoxceleb')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create(source, dest, seed=42):
    audit = json.loads((source / 'audit.json').read_text())
    for name, expected in audit['prepared_sha256'].items():
        if digest(source / f'{name}.txt') != expected:
            raise ValueError(f'Changed source: {name}')
    lines = (source / 'all_train.txt').read_text().splitlines(keepends=True)
    speakers = defaultdict(list)
    for line in lines:
        corpus, speaker, rest = line.split(maxsplit=2)
        if corpus not in CORPORA:
            raise ValueError(corpus)
        speakers[corpus, speaker].append(float(rest.rsplit(maxsplit=1)[1]))
    test_ids = set()
    for name in audit['prepared_sha256']:
        if name != 'all_train':
            for line in (source / f'{name}.txt').read_text().splitlines():
                corpus, speaker, _ = line.split(maxsplit=2)
                test_ids.add((corpus, speaker))
    if test_ids & speakers.keys():
        raise ValueError('Source train/test speaker overlap')
    rng = random.Random(seed)
    selected = set()
    for corpus in CORPORA:
        ids = sorted(k for k in speakers if k[0] == corpus)
        rng.shuffle(ids)  # deterministic tie breaking for equal median ages
        ids.sort(key=lambda k: statistics.median(speakers[k]))
        n = max(1, round(len(ids) * .1))
        if n >= len(ids):
            raise ValueError(f'Too few speakers: {corpus}')
        # One random speaker from each equal-sized age-ordered group.
        for i in range(n):
            selected.add(rng.choice(ids[i*len(ids)//n:(i+1)*len(ids)//n]))
    dest.mkdir(parents=True, exist_ok=False)
    for name in audit['prepared_sha256']:
        shutil.copyfile(source / f'{name}.txt', dest / f'{name}.txt')
    shutil.copyfile(source / 'audit.json', dest / 'audit.json')
    outputs = {'train': [], **{f'{c}_val': [] for c in CORPORA}}
    counts = {c: dict(train_records=0, val_records=0,
                     train_speakers=sum(k[0] == c and k not in selected for k in speakers),
                     val_speakers=sum(k[0] == c for k in selected)) for c in CORPORA}
    for line in lines:
        c, sp, _ = line.split(maxsplit=2)
        validation = (c, sp) in selected
        outputs[f'{c}_val' if validation else 'train'].append(line)
        counts[c]['val_records' if validation else 'train_records'] += 1
    for name, rows in outputs.items():
        (dest / f'{name}.txt').write_text(''.join(rows))
    manifest = dict(version=1, seed=seed, fraction_speakers=.1,
                    method='one random speaker per equal-sized median-age-ordered group within each corpus',
                    source_sha256=audit['prepared_sha256'], counts=counts,
                    files_sha256={n: digest(dest / f'{n}.txt') for n in outputs})
    (dest / 'partition.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(counts, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--dest', type=Path, required=True)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    create(a.source, a.dest, a.seed)
