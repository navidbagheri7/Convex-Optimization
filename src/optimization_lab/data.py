"""Explicit cache downloads; raw datasets never enter the public repository."""

import gzip
import json
import urllib.request
import zipfile
from pathlib import Path

from .artifacts import sha256, write_json
from .resources import config_path
from .routing import load_tsplib


def download(url, destination):
    temporary = destination.with_suffix(destination.suffix + ".partial")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as target:
            while chunk := response.read(1024 * 1024):
                target.write(chunk)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def prepare(dataset, data_root):
    root = Path(data_root).expanduser().resolve() / "optimization"
    metadata = json.loads(config_path("data_manifest.json").read_text())
    if dataset == "movielens":
        root /= "movielens"
        root.mkdir(parents=True, exist_ok=True)
        archive = root / "ml-100k.zip"
        spec = metadata[dataset]
        if not archive.exists():
            download(spec["url"], archive)
        if sha256(archive) != spec["sha256"]:
            raise ValueError("MovieLens archive SHA256 disagrees with recorded official download")
        with zipfile.ZipFile(archive) as source:
            for member in source.infolist():
                destination = (root / member.filename).resolve()
                if not destination.is_relative_to(root):
                    raise ValueError("Unsafe archive path")
            source.extractall(root)
        write_json(
            root / "provenance.json",
            spec | dict(policy="Raw ratings private; see downloaded README for dataset terms."),
        )
    else:
        root /= "routing"
        root.mkdir(parents=True, exist_ok=True)
        provenance = {}
        for name, spec in metadata["routing"].items():
            path = root / (name + ".tsp.gz")
            if not path.exists():
                download(spec["url"], path)
            actual, _, _ = load_tsplib(path)
            if actual != name:
                raise ValueError("Unexpected instance name")
            raw = path.read_bytes()
            decompressed = gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
            from hashlib import sha256 as hash_bytes

            provenance[name] = spec | dict(
                cache_sha256=sha256(path),
                decompressed_sha256=hash_bytes(decompressed).hexdigest(),
                origin="Official download or already supplied course instance; cache hash records actual bytes.",
            )
        write_json(root / "provenance.json", provenance)
    print(root)
    return root
