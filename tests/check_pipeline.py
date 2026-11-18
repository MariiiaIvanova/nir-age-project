"""Integration check: requires prepared validation_seed42, real audio, tiny weights and GPU 0.
Creates temporary fixtures and exercises one epoch plus resume; does not run full training.
"""
import hashlib,json,sys,tempfile,subprocess,os
from pathlib import Path
from collections import Counter
sys.path.insert(0,str(Path('app').resolve()))
from split_validation import create
src=Path('output/native/validation_seed42')
tmp=Path(tempfile.mkdtemp(prefix='nir-validation-check-'))
create(Path('output/native/current'),tmp/'repeat')
for n in ['train','timit_val','nnces_val','agevoxceleb_val']:
 assert (src/f'{n}.txt').read_bytes()==(tmp/'repeat'/f'{n}.txt').read_bytes()
train=(src/'train.txt').read_text().splitlines(); vals=sum([(src/f'{c}_val.txt').read_text().splitlines() for c in ['timit','nnces','agevoxceleb']],[])
assert Counter(train+vals)==Counter((src/'all_train.txt').read_text().splitlines())
ids=lambda lines:{tuple(l.split()[:2]) for l in lines}
assert not ids(train)&ids(vals)
for f in src.glob('*_test.txt'):
 assert not ids(train+vals)&ids(f.read_text().splitlines())
small=tmp/'small'; small.mkdir()
audit=json.loads((src/'audit.json').read_text())
for name in audit['prepared_sha256']:
 lines=(src/f'{name}.txt').read_text().splitlines(keepends=True)
 (small/f'{name}.txt').write_text(''.join(lines[:2]))
 audit['prepared_sha256'][name]=hashlib.sha256((small/f'{name}.txt').read_bytes()).hexdigest()
(small/'audit.json').write_text(json.dumps(audit))
p=json.loads((src/'partition.json').read_text()); p['source_sha256']=audit['prepared_sha256']
for name in p['files_sha256']:
 lines=(src/f'{name}.txt').read_text().splitlines(keepends=True)
 (small/f'{name}.txt').write_text(''.join(lines[:2]))
 p['files_sha256'][name]=hashlib.sha256((small/f'{name}.txt').read_bytes()).hexdigest()
(small/'partition.json').write_text(json.dumps(p))
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTORCH_KERNEL_CACHE_PATH='/tmp/nir-torch-kernels')
Path(env['PYTORCH_KERNEL_CACHE_PATH']).mkdir(exist_ok=True)
cmd=[sys.executable,'-u','app/train.py','--splits',str(small),'--weights','weights/tiny.pt','--run',str(tmp/'run'),'--epochs','1','--workers','0','--batch-size','2']
subprocess.run(cmd,check=True,env=env)
assert len(json.loads((tmp/'run/metrics.json').read_text()))==5
before=(tmp/'run/metrics.json').read_bytes()
subprocess.run(cmd+['--resume'],check=True,env=env)
assert before==(tmp/'run/metrics.json').read_bytes()
print('PASS: reproducible partition, conservation, speaker isolation, real GPU training/validation/five tests/checkpoints/resume',tmp)

timing=json.loads((tmp/'run/timing.json').read_text())
assert len(timing['sessions']) == 2
assert all(s['status']=='complete' and s['elapsed_seconds'] > 0 for s in timing['sessions'])
assert [p['stage'] for p in timing['sessions'][0]['phases']] == ['train','validation','test']
assert [p['stage'] for p in timing['sessions'][1]['phases']] == ['test']
assert all(p['seconds'] >= 0 and p['status']=='complete' for s in timing['sessions'] for p in s['phases'])
print('PASS: per-phase timing and separate resume sessions')
