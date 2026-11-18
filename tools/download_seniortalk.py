"""Optional host-side acquisition. Invoke with the destination corpus root."""
import argparse
import shutil
from pathlib import Path

import soundfile as sf
from datasets import load_dataset
from huggingface_hub import hf_hub_download


def main(root):
    (root / "wav").mkdir(parents=True, exist_ok=True)
    repo = "evan0617/seniortalk"
    meta = hf_hub_download(repo, "SPKINFO.txt", repo_type="dataset")
    shutil.copyfile(meta, root / "SPKINFO.txt")
    for split in ("train", "dev", "test"):
        for row in load_dataset(repo, "sentence_data", split=split, streaming=True):
            audio = row["path"]
            path = root / "wav" / Path(audio["path"]).name
            tmp = path.with_suffix(".part")
            sf.write(tmp, audio["array"], audio["sampling_rate"], format="WAV")
            tmp.replace(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    main(parser.parse_args().destination)
