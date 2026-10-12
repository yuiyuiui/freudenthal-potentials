"""Read the fixed finite certificates, with integrity checks before use."""

import gzip
import hashlib
import json
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy import sparse

CERTIFICATES = Path(__file__).resolve().parent / "certificates"


@lru_cache(maxsize=1)
def verify_data_integrity() -> dict:
    manifest = json.loads((CERTIFICATES / "manifest.json").read_text())
    for name, record in manifest["files"].items():
        blob = (CERTIFICATES / name).read_bytes()
        if len(blob) != record["bytes"] or hashlib.sha256(blob).hexdigest() != record["sha256"]:
            raise ValueError(f"Certificate integrity check failed: {name}")
    return manifest


@lru_cache(maxsize=1)
def rules() -> list[dict]:
    manifest = verify_data_integrity()
    blob = gzip.decompress((CERTIFICATES / "all_degree_rules.json.gz").read_bytes())
    if hashlib.sha256(blob).hexdigest() != manifest["all_degree_rules_uncompressed_sha256"]:
        raise ValueError("Uncompressed rule table has the wrong hash")
    return json.loads(blob)


def certificate_bytes(name: str) -> bytes:
    verify_data_integrity()
    with ZipFile(CERTIFICATES / "verification_data.zip") as archive:
        return archive.read(name)


def certificate_json(name: str):
    return json.loads(certificate_bytes(name))


def certificate_array(name: str):
    return np.load(BytesIO(certificate_bytes(name)), allow_pickle=False)


def certificate_sparse(name: str):
    return sparse.load_npz(BytesIO(certificate_bytes(name))).astype(np.int64).tocsr()
