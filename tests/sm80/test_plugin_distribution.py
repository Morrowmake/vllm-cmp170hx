# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The plugins/sm80 distribution ships the EXL3 license files.

Every payload that ships EXL3 code carries plugins/sm80/licenses/exl3/
{LICENSE,NOTICE}, the MIT grant of the adapted ExLlamaV3 code: the wheel (in
its dist-info), the sdist and an installed copy. They are the only copies in
the repository; license files must sit inside the project directory.
"""

import os
import shutil
import tarfile
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
# The distribution's project directory: the root of the plugin's source tree,
# or plugins/sm80/ in an engine source tree that carries the plugin.
PLUGIN = REPO / "plugins" / "sm80" if (REPO / "plugins" / "sm80").is_dir() else REPO
SOURCES = PLUGIN / "licenses" / "exl3"
NAMES = ("LICENSE", "NOTICE")
# PEP 639 project.license-files needs setuptools 77.
MIN_SETUPTOOLS = (77,)


def _expected() -> dict[str, bytes]:
    return {name: (SOURCES / name).read_bytes() for name in NAMES}


def test_license_files_carry_the_exl3_grant():
    # The adapted sources under csrc/exl3 cite this grant in their headers.
    license_text, notice = _expected()["LICENSE"], _expected()["NOTICE"]
    assert license_text.startswith(b"MIT License\n\nCopyright (c) 2025 Turboderp")
    assert b"github.com/turboderp-org/exllamav3" in notice


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    setuptools = pytest.importorskip("setuptools")
    version = tuple(int(p) for p in setuptools.__version__.split(".")[:1])
    if version < MIN_SETUPTOOLS:
        pytest.skip(f"setuptools {setuptools.__version__} < 77 cannot build license-files")
    from setuptools import build_meta

    root = tmp_path_factory.mktemp("plugin")
    source, dist = root / "sm80", root / "dist"
    shutil.copytree(PLUGIN, source)
    dist.mkdir()
    cwd = os.getcwd()
    os.chdir(source)
    try:
        wheel = build_meta.build_wheel(str(dist))
        sdist = build_meta.build_sdist(str(dist))
    finally:
        os.chdir(cwd)
    return dist / wheel, dist / sdist


def _members(names, wanted_dir):
    return {name.rsplit("/", 1)[1]: name for name in names
            if name.rsplit("/", 1)[0].endswith(wanted_dir)
            and name.rsplit("/", 1)[1] in NAMES}


def test_wheel_carries_the_license_files_in_its_dist_info(built):
    wheel, _ = built
    with zipfile.ZipFile(wheel) as archive:
        found = _members(archive.namelist(), ".dist-info/licenses/licenses/exl3")
        assert sorted(found) == sorted(NAMES), archive.namelist()
        for name, member in found.items():
            assert archive.read(member) == _expected()[name], member
        metadata = next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
        text = archive.read(metadata).decode()
    assert "License-Expression: Apache-2.0 AND MIT" in text
    for name in NAMES:
        assert f"License-File: licenses/exl3/{name}" in text


def test_sdist_carries_the_license_files(built):
    _, sdist = built
    with tarfile.open(sdist) as archive:
        found = _members(archive.getnames(), "/licenses/exl3")
        assert sorted(found) == sorted(NAMES), archive.getnames()
        for name, member in found.items():
            assert archive.extractfile(member).read() == _expected()[name], member
