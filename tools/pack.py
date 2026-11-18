"""Package a handoff with all portable benchmark lists and without machine data."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

root = Path(__file__).resolve().parents[1]
files = ["README.md", "Makefile", "Dockerfile", "compose.yaml", "env.example",
         ".gitignore", ".dockerignore", "requirements.txt", "requirements-download.txt"]
for pattern in ("benchmark/*.tsv.gz", "app/**/*.py", "tools/*.py", "tools/*.sh", "tests/*.py"):
    files.extend(str(p.relative_to(root)) for p in sorted(root.glob(pattern)))
if len(list((root / "benchmark").glob("*.tsv.gz"))) != 6:
    raise SystemExit("Run make fixed before packaging.")
dest = root / "dist/nir-age.zip"
dest.parent.mkdir(exist_ok=True)
with ZipFile(dest.with_suffix(".tmp"), "w", ZIP_DEFLATED) as archive:
    for name in files:
        archive.write(root / name, f"nir-age/{name}")
dest.with_suffix(".tmp").replace(dest)
print(dest)
