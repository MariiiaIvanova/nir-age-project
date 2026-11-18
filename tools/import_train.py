"""One-time conversion of the existing split; standard library only."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from benchmark import TRAIN_BLOB, convert

p = argparse.ArgumentParser()
p.add_argument("source", type=Path)
p.add_argument("destination", type=Path)
args = p.parse_args()
convert(args.source, args.destination, TRAIN_BLOB)
