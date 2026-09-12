"""Where the shareware WAD lives, and how it gets there.

DOOM1.WAD is the freely redistributable shareware episode id Software released in
1993; it is not open source and it is not committed here. The first run fetches
the original `doom19s.zip` from the Internet Archive, unpacks its DEICE volumes
and verifies the WAD by size and hash before anything reads it.
"""
from __future__ import annotations

import hashlib
import io
import os
import urllib.request
import zipfile
from pathlib import Path

SHAREWARE_URL = "https://archive.org/download/doom_20230531/doom19s.zip"
WAD_SIZE = 4_196_020
WAD_SHA256 = "1d7d43be501e67d927e415e0b8f3e29c3bf33075e859721816f652a526cac771"
ENV_VAR = "STREAMLIB_DOOM_WAD"


def wad_path() -> Path:
    override = os.environ.get(ENV_VAR)
    if override:
        return Path(override)
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "streamlib-doom"
    return cache / "DOOM1.WAD"


def ensure_wad() -> Path:
    """The WAD on disk, downloaded and verified if it is not there yet."""
    path = wad_path()
    if path.exists() and path.stat().st_size == WAD_SIZE:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(SHAREWARE_URL, timeout=120) as response:
        shareware_zip = response.read()
    with zipfile.ZipFile(io.BytesIO(shareware_zip)) as outer:
        volumes = b"".join(outer.read(name) for name in sorted(n for n in outer.namelist() if n.upper().startswith("DOOMS_19.") and n[-1].isdigit()))
    with zipfile.ZipFile(io.BytesIO(volumes)) as inner:
        wad = inner.read(next(n for n in inner.namelist() if n.upper() == "DOOM1.WAD"))
    if len(wad) != WAD_SIZE:
        raise RuntimeError(f"downloaded DOOM1.WAD is {len(wad)} bytes, expected {WAD_SIZE}")
    digest = hashlib.sha256(wad).hexdigest()
    if digest != WAD_SHA256:
        raise RuntimeError(f"downloaded DOOM1.WAD has sha256 {digest}, expected {WAD_SHA256}")
    path.write_bytes(wad)
    return path
