import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import random
import re
import tempfile
import uuid
from collections import Counter, defaultdict
from pathlib import Path

from benchmark import LISTS, check_key


def read_scp(path):
    result = {}
    with path.open() as f:
        for line in f:
            key, value = line.rstrip().split(maxsplit=1)
            if key in result:
                raise ValueError(f"Duplicate ID {key} in {path}")
            result[key] = value
    return result


def timit_index(root, split):
    scp = root / "scp" / split
    ages = read_scp(scp / "utt2age_reg")
    result = {}
    for uid, host_path in read_scp(scp / "wav.scp").items():
        key = re.sub(r"^DR[0-9]+_", "", uid.upper())
        if key in result:
            raise ValueError(f"Ambiguous TIMIT ID: {split}/{key}")
        marker = f"/wav16k/{split}/"
        if marker not in host_path:
            raise ValueError(f"Unexpected TIMIT layout: {host_path}")
        relative = host_path.rsplit(marker, 1)[1]
        check_key(relative)
        result[key] = (root / "wav16k" / split / relative, float(ages[uid]))
    return result


def sample_senior(rows, limit, seed):
    if not limit:
        return rows
    groups = defaultdict(list)
    for row in rows:
        groups[row[1]].append(row)
    rng = random.Random(seed)
    return [row for speaker in sorted(groups)
            for row in sorted(rng.sample(sorted(groups[speaker]), min(limit, len(groups[speaker]))))]


def prepare(args):
    fixed, data, work = args.fixed, args.data, args.work
    train = fixed / "all_train.tsv.gz"
    if not train.is_file():
        raise SystemExit("Нет переносимого all_train.tsv.gz. Выполни make fixed.")

    env_keys = {"timit": "TIMIT_ROOT", "nnces": "NNCES_AUDIO_ROOT",
                "nnces_scp": "NNCES_SCP_ROOT", "agevoxceleb": "AGEVOX_AUDIO_ROOT",
                "voicebio": "VOICEBIO_ROOT", "seniortalk": "SENIORTALK_ROOT"}
    roots = {name: Path(os.environ[key]).resolve(strict=True) if args.host_roots else data / name
             for name, key in env_keys.items()}
    indices = {split: timit_index(roots["timit"], split) for split in ("train", "test")}
    nnces_ages = read_scp(roots["nnces_scp"] / "utt2age_reg")
    nnces_index = {}
    for uid, path in read_scp(roots["nnces_scp"] / "wav.scp").items():
        if "/Read_Speech_Data/" not in path:
            if "Spontaneous_Speech_Data" in str(path).split("/"):
                continue
            raise ValueError(f"Unexpected NNCES layout: {path}")
        key = path.rsplit("/Read_Speech_Data/", 1)[1]
        if key in nnces_index:
            raise ValueError(f"Ambiguous NNCES recording: {key}")
        nnces_index[key] = float(nnces_ages[uid])
    work.mkdir(parents=True, exist_ok=True)
    seen = set()
    train_speakers = set()
    audit = {"lists": {}, "source_sha256": {}, "prepared_sha256": {},
             "decode": not args.paths_only, "seed": args.seed,
             "senior_per_speaker": args.senior_per_speaker}
    with tempfile.TemporaryDirectory(dir=work) as tmp:
        stage = Path(tmp) / "splits"
        stage.mkdir()
        with (stage / "age_differences.tsv").open("w") as log, (stage / "excluded.tsv").open("w") as excluded:
            differences = csv.writer(log, delimiter="\t")
            differences.writerow(["list", "speaker", "key", "benchmark_age", "scp_age"])
            rejects = csv.writer(excluded, delimiter="\t")
            rejects.writerow(["list", "path", "error"])
            for name in LISTS:
                print(f"Проверка {name}", flush=True)
                source = fixed / f"{name}.tsv.gz"
                audit["source_sha256"][name] = hashlib.sha256(source.read_bytes()).hexdigest()
                with gzip.open(source, "rt") as f:
                    rows = list(csv.reader(f, delimiter="\t"))
                counts = Counter(source=len(rows))
                if name == "seniortalk_test":
                    rows = sample_senior(rows, args.senior_per_speaker, args.seed)
                counts["selected"] = len(rows)
                output = stage / f"{name}.txt"
                with output.open("w") as out:
                    for corpus, speaker, key, age in rows:
                        check_key(key)
                        split = "train" if name == "all_train" else "test"
                        if split == "train":
                            if corpus not in ("timit", "nnces", "agevoxceleb"):
                                raise ValueError(f"Unexpected train corpus: {corpus}")
                            train_speakers.add((corpus, speaker))
                        elif name != f"{corpus}_test" or (corpus, speaker) in train_speakers:
                            raise ValueError(f"Split mismatch: {name}/{corpus}/{speaker}")
                        label = float(age)
                        if not math.isfinite(label) or not 0 <= label <= 100:
                            raise ValueError(f"Invalid age {age}: {key}")
                        path = roots[corpus] / key
                        scp_age = None
                        if corpus == "timit":
                            path, scp_age = indices[split][key]
                        elif corpus == "nnces":
                            if (
                                split == "train"
                                and key == "8Y/Boys_8/M5/M5_10/M5_10_01.wav" ##fg
                                and key not in nnces_index
                                and not path.exists()
                            ):
                                rejects.writerow([name, str(path), "missing_read_speech_file"])
                                counts["excluded"] += 1
                                continue
                            scp_age = nnces_index[key]
                        if scp_age is not None and scp_age != label:
                            differences.writerow([name, speaker, key, age, scp_age])
                            counts["age_differences"] += 1
                        if path in seen:
                            raise ValueError(f"Repeated recording: {path}")
                        seen.add(path)
                        if not path.is_file() or not os.access(path, os.R_OK):
                            raise FileNotFoundError(f"{name}: {path}")
                        if not args.paths_only:
                            from audio_data import read_audio
                            try:
                                read_audio(path)
                            except Exception as exc:
                                if not args.exclude_decode_errors:
                                    raise RuntimeError(f"{name}: {path}") from exc
                                rejects.writerow([name, str(path), str(exc)])
                                counts["excluded"] += 1
                                continue
                        out.write(f"{corpus} {speaker} {path} {age}\n")
                        counts["kept"] += 1
                if not counts["kept"]:
                    raise ValueError(f"Empty list: {name}")
                audit["lists"][name] = dict(counts)
                audit["prepared_sha256"][name] = hashlib.sha256(output.read_bytes()).hexdigest()
        (stage / "audit.json").write_text(json.dumps(audit, indent=2))
        token = uuid.uuid4().hex
        dest = work / f"splits-{token}"
        stage.rename(dest)
        link = work / f".current-{token}"
        link.symlink_to(dest.name, target_is_directory=True)
        link.replace(work / "current")
        print(json.dumps(audit["lists"], indent=2), flush=True)
        print(f"Готово: {dest}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fixed", type=Path, default=Path("/fixed"))
    p.add_argument("--data", type=Path, default=Path("/data"))
    p.add_argument("--work", type=Path, default=Path("/output"))
    p.add_argument("--host-roots", action="store_true")
    p.add_argument("--paths-only", action="store_true")
    p.add_argument("--exclude-decode-errors", action="store_true")
    p.add_argument("--senior-per-speaker", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    if a.senior_per_speaker < 0 or (a.paths_only and a.exclude_decode_errors):
        p.error("Invalid sampling or decode options")
    prepare(a)
