"""Small portable records of the implementation and fixed certificate inputs."""

import hashlib
import json
from pathlib import Path

from .data import verify_data_integrity


def implementation_manifest() -> dict:
    package = Path(__file__).resolve().parent
    files = {
        path.relative_to(package.parent).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(package.rglob("*"))
        if path.is_file() and path.suffix in (".py", ".cpp", ".json", ".gz", ".zip")
    }
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {
        "implementation_sha256": hashlib.sha256(encoded).hexdigest(),
        "files": files,
        "certificates": verify_data_integrity(),
    }
