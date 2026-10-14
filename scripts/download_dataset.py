"""Download the versioned dataset and verify every original file."""
import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify(folder, records):
    failures = [r["path"] for r in records if not (folder / r["path"]).is_file() or sha256(folder / r["path"]) != r["sha256"]]
    if failures:
        raise ValueError(f"Dataset integrity mismatch ({len(failures)} files): {failures[:5]}")
    actual = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
    expected = {r["path"] for r in records}
    if actual != expected:
        raise ValueError(f"Unexpected raw files: {sorted(actual - expected)[:5]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    descriptor = json.loads((ROOT / "dataset.json").read_text(encoding="utf-8"))
    records = json.loads((ROOT / descriptor["raw_sha256_manifest"]).read_text(encoding="utf-8"))
    destination = ROOT / "Multimodal data for prognostics"
    if destination.exists():
        verify(destination, records)
        print(f"Existing dataset verified: {len(records)} files. No download or overwrite.")
        return
    if args.verify_only:
        raise FileNotFoundError(destination)
    url = f"https://huggingface.co/datasets/{descriptor['repo_id']}/resolve/{descriptor['revision']}/{descriptor['filename']}"
    with tempfile.TemporaryDirectory(prefix="mbdnn-data-", dir=ROOT) as temporary:
        staging = Path(temporary)
        archive = staging / "dataset.zip"
        request = urllib.request.Request(url, headers={"User-Agent": "multi-branch-multimodal-prognostics"})
        with urllib.request.urlopen(request, timeout=120) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output)
        if sha256(archive) != descriptor["sha256"]:
            raise ValueError("Archive SHA-256 mismatch; extraction cancelled")
        extracted = staging / "extracted"
        extracted.mkdir()
        with zipfile.ZipFile(archive) as stream:
            for info in stream.infolist():
                resolved = (extracted / info.filename).resolve()
                if not resolved.is_relative_to(extracted.resolve()):
                    raise ValueError(f"Unsafe ZIP path: {info.filename}")
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("Archive symlinks are not permitted")
            stream.extractall(extracted)
        verified = extracted / destination.name
        verify(verified, records)
        if destination.exists():
            raise FileExistsError("Dataset destination appeared during download; existing files preserved")
        verified.rename(destination)
    print(f"Downloaded and verified {len(records)} raw files. No training executed.")


if __name__ == "__main__":
    main()
