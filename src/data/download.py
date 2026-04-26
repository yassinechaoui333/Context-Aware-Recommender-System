from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final
from urllib.request import urlretrieve
from zipfile import ZipFile

DATASET_SPECS: Final[dict[str, dict[str, str]]] = {
    "1m": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
        "md5": "c4d9eecfca2ab87c1945afe126590906",
        "folder": "ml-1m",
    },
    "100k": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
        "md5": "0e33842e24a9c977be4e0107933c0723",
        "folder": "ml-100k",
    },
}


def _compute_md5(path: Path) -> str:
    hasher = hashlib.md5()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_movielens(version: str = "1m", dest: str = "data/raw/movielens/") -> Path:
    if version not in DATASET_SPECS:
        supported_versions = ", ".join(sorted(DATASET_SPECS))
        raise ValueError(
            f"Unsupported MovieLens version '{version}'. Supported: {supported_versions}"
        )

    spec = DATASET_SPECS[version]
    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)

    archive_path = dest_path / f"{spec['folder']}.zip"
    extracted_path = dest_path / spec["folder"]

    if not archive_path.exists():
        urlretrieve(spec["url"], archive_path)

    expected_md5 = spec["md5"]
    actual_md5 = _compute_md5(archive_path)
    if actual_md5 != expected_md5:
        raise ValueError(
            "MovieLens archive checksum mismatch: "
            f"expected={expected_md5}, actual={actual_md5}, file='{archive_path}'."
        )

    ratings_file = "ratings.dat" if version == "1m" else "u.data"
    if not (extracted_path / ratings_file).exists():
        with ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(dest_path)

    return extracted_path
