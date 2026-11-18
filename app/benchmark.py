"""Portable benchmark format: corpus, speaker, recording key, reference age."""
import gzip
import hashlib
import io
import os
import tempfile
from pathlib import Path, PurePosixPath

TESTS = ("timit", "nnces", "agevoxceleb", "seniortalk", "voicebio")
LISTS = ("all_train", *(f"{c}_test" for c in TESTS))
TRAIN_BLOB = "9fabf1c721d10b6734dd2f39d70af6cf895805af"
MARKERS = {
    "nnces": "/Read_Speech_Data/",
    "agevoxceleb": "/voxceleb2/",
    "voicebio": "/voice_biometrics_age/voice_biometrics_age/",
    "seniortalk": "/seniortalk_full/",
}


def portable_row(line):
    corpus, speaker, old, age = line.split()
    if corpus == "timit":
        key = Path(old).stem.upper()
    else:
        prefix, marker, key = old.rpartition(MARKERS[corpus])
        if not marker:
            raise ValueError(f"Unrecognized {corpus} path: {old}")
    check_key(key)
    return corpus, speaker, key, age


def check_key(key):
    path = PurePosixPath(key)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Invalid recording key: {key}")


def convert(source, dest, expected_blob=None):
    data = source.read_bytes()
    if expected_blob:
        digest = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if digest != expected_blob:
            raise ValueError(f"Unexpected source version: {source} (git blob {digest})")
    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp:
        temp = Path(tmp.name)
    try:
        with temp.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            with io.TextIOWrapper(gz, encoding="utf-8", newline="\n") as out:
                for line in data.decode().splitlines():
                    if line.strip():
                        out.write("\t".join(portable_row(line)) + "\n")
        os.chmod(temp, 0o644)
        temp.replace(dest)
    finally:
        temp.unlink(missing_ok=True)
