# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The table segment holder (vllm_sm80.table_segment): exact sealed
images, identity and digest checks, the release modes and the client lease."""

import fcntl
import hashlib
import json
import mmap
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from vllm_sm80 import table_segment as ts

SCRIPT = Path(ts.__file__)
# Sizes off any alignment, and one over a fill chunk.
SIZES = (ts.CHUNK_BYTES + 12345, 3 * 4096 + 7)


@pytest.fixture
def files(tmp_path):
    paths = []
    for index, size in enumerate(SIZES):
        path = tmp_path / f"model-{index:05d}.safetensors"
        path.write_bytes(os.urandom(size))
        paths.append(path)
    return [(str(p), hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]


def _status(sock, timeout=30.0, want=("ready",)):
    deadline = time.monotonic() + timeout
    while True:
        try:
            state = ts.request(str(sock), {"op": "status"}, timeout=2.0)
            if state["state"] in want:
                return state
        except ts.SegmentError:
            pass
        if time.monotonic() > deadline:
            raise TimeoutError(f"holder at {sock} not {want}")
        time.sleep(0.05)


def _hold(sock, files, label="repo@rev"):
    argv = [sys.executable, str(SCRIPT), "hold", "--socket", str(sock), "--label", label]
    for path, sha in files:
        argv += ["--file", f"{path}={sha}"]
    return subprocess.Popen(argv, stderr=subprocess.PIPE, text=True)


@pytest.fixture
def holder(tmp_path, files):
    sock = tmp_path / "h.sock"
    proc = _hold(sock, files)
    try:
        _status(sock)
        yield sock, proc
    finally:
        if proc.poll() is None:
            proc.terminate()
        proc.wait(30)


def _manifest(label, named):
    """The manifest hash, spelled out: sha256 of the JSON of the label and
    each file's name as given, size and digest (sorted keys, no spaces)."""
    doc = {"label": label, "files": [
        {"name": os.path.basename(p), "size": os.path.getsize(p), "sha256": s}
        for p, s in named]}
    text = json.dumps(doc, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def _bytes(path, offset, nbytes):
    with open(path, "rb") as file:
        file.seek(offset)
        return file.read(nbytes)


def test_holder_serves_exact_sealed_images(holder, files):
    sock, _ = holder
    state = ts.request(str(sock), {"op": "status"})
    assert state["manifest"] == _manifest("repo@rev", files)
    assert state["bytes_done"] == state["bytes_total"] == sum(SIZES)
    assert [f["sha256"] for f in state["files"]] == [s for _, s in files]
    attachment = ts.Attachment(str(sock), [p for p, _ in files])
    try:
        assert [c["pid"] for c in ts.request(str(sock), {"op": "status"})["clients"]] == [
            os.getpid()
        ]
        for path, _ in files:
            size = os.path.getsize(path)
            for offset, nbytes in ((0, size), (4097, size - 4097 - 3), (size - 1, 1)):
                mapping, skip = attachment.map(path, offset, nbytes)
                assert mapping[skip : skip + nbytes] == _bytes(path, offset, nbytes)
                with pytest.raises(TypeError):
                    mapping[skip] = 0
            fd, _ = attachment._fds[os.path.realpath(path)]
            seals = fcntl.fcntl(fd, ts.F_GET_SEALS)
            assert seals & ts._SEALS == ts._SEALS
            assert not os.get_inheritable(fd)
            with pytest.raises(PermissionError):
                mmap.mmap(fd, 4096, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE)
        with pytest.raises(ValueError):
            attachment.map(files[1][0], 0, SIZES[1] + 1)
    finally:
        attachment.close()


def test_manifest_names_files_as_given(tmp_path, files):
    """A file named through a link is hashed under the link's name; the
    image, its identity and the attach are the real file's."""
    links = tmp_path / "links"
    links.mkdir()
    named = []
    for index, (path, sha) in enumerate(files):
        link = links / f"table-{index}.safetensors"
        link.symlink_to(path)
        named.append((str(link), sha))
    sock = tmp_path / "l.sock"
    proc = _hold(sock, named)
    try:
        state = _status(sock)
        assert state["manifest"] == _manifest("repo@rev", named)
        assert state["manifest"] != _manifest("repo@rev", files)
        assert [f["path"] for f in state["files"]] == [p for p, _ in files]
        attachment = ts.Attachment(str(sock), [p for p, _ in named])
        assert attachment.manifest == state["manifest"]
        attachment.close()
    finally:
        ts.request(str(sock), {"op": "release", "mode": "force"})
        assert proc.wait(30) == 0


def test_attach_refuses_a_changed_file(holder, files):
    sock, _ = holder
    path = files[1][0]
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    with pytest.raises(ts.SegmentError, match="not the file the holder read"):
        ts.Attachment(str(sock), [path])
    with pytest.raises(ts.SegmentError, match="does not hold"):
        ts.Attachment(str(sock), [str(SCRIPT)])


def test_wrong_digest_ends_the_holder(tmp_path, files):
    sock = tmp_path / "h.sock"
    bad = [(files[0][0], "0" * 64), files[1]]
    proc = _hold(sock, bad)
    _, err = proc.communicate(timeout=60)
    assert proc.returncode == ts.EXIT_VERIFY, err
    assert "sha256" in err
    assert not sock.exists()


def test_release_idle_and_drain_wait_for_clients(holder, files):
    sock, proc = holder
    attachment = ts.Attachment(str(sock), [files[0][0]])
    mapping, skip = attachment.map(files[0][0], 0, 4096)
    reply = ts.request(str(sock), {"op": "release", "mode": "idle"})
    assert reply["released"] is False and reply["clients"] == [os.getpid()]
    reply = ts.request(str(sock), {"op": "release", "mode": "drain"})
    assert reply["released"] is False and reply["state"] == "draining"
    with pytest.raises(ts.SegmentError, match="draining"):
        ts.Attachment(str(sock), [files[0][0]])
    assert proc.poll() is None
    attachment.close()
    assert proc.wait(30) == 0
    assert not sock.exists()
    # The mapping outlives the holder.
    assert mapping[skip : skip + 4096] == _bytes(files[0][0], 0, 4096)


def test_release_idle_without_clients_exits(holder):
    sock, proc = holder
    assert ts.request(str(sock), {"op": "release", "mode": "idle"})["released"] is True
    assert proc.wait(30) == 0


def test_force_release_keeps_client_mappings(holder, files):
    sock, proc = holder
    path = files[0][0]
    attachment = ts.Attachment(str(sock), [path])
    mapping, skip = attachment.map(path, 100, 5000)
    assert ts.request(str(sock), {"op": "release", "mode": "force"})["released"] is True
    assert proc.wait(30) == 0
    assert mapping[skip : skip + 5000] == _bytes(path, 100, 5000)
    attachment.close()


def test_second_holder_on_the_socket_is_refused(holder, files):
    sock, _ = holder
    second = _hold(sock, files)
    _, err = second.communicate(timeout=30)
    assert second.returncode == 1 and "another table segment holder" in err
    assert ts.request(str(sock), {"op": "status"})["state"] == "ready"


def test_no_holder(tmp_path):
    sock = tmp_path / "none.sock"
    with pytest.raises(ts.SegmentError, match="no table segment holder"):
        ts.Attachment(str(sock), [str(SCRIPT)])
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "status", "--socket", str(sock)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == ts.EXIT_NO_HOLDER


class _GatedHolder(ts.Holder):
    def __init__(self, *args, gate, **kwargs):
        super().__init__(*args, **kwargs)
        self.gate = gate

    def fill(self):
        self.gate.wait(30)
        super().fill()


def test_attach_waits_for_the_fill(tmp_path, files):
    sock = tmp_path / "h.sock"
    gate = threading.Event()
    holder = _GatedHolder(str(sock), "repo@rev", files, gate=gate, log=lambda msg: None)
    server = threading.Thread(target=holder.serve)
    server.start()
    try:
        _status(sock, want=("filling",))
        result = {}

        def attach():
            result["attachment"] = ts.Attachment(str(sock), [files[1][0]])

        client = threading.Thread(target=attach)
        client.start()
        client.join(0.5)
        assert client.is_alive(), "attach answered before the fill"
        # A server waiting for the fill is not idle.
        assert ts.request(str(sock), {"op": "status"})["pending"] == 1
        reply = ts.request(str(sock), {"op": "release", "mode": "idle"})
        assert reply["released"] is False
        gate.set()
        client.join(30)
        attachment = result["attachment"]
        mapping, skip = attachment.map(files[1][0], 0, SIZES[1])
        assert mapping[skip : skip + SIZES[1]] == _bytes(files[1][0], 0, SIZES[1])
        attachment.close()
    finally:
        gate.set()
        ts.request(str(sock), {"op": "release", "mode": "force"})
        server.join(30)
    assert holder.exit_code == 0


def _fill_failure(tmp_path, files, monkeypatch, read_into):
    monkeypatch.setattr(ts, "_read_into", read_into)
    holder = ts.Holder(str(tmp_path / "h.sock"), "repo@rev", files, log=lambda msg: None)
    holder.fill()
    return holder


def test_a_read_error_ends_the_fill_with_its_cause(tmp_path, files, monkeypatch):
    def fail(fd, view, offset, need):
        raise OSError(5, "Input/output error")

    holder = _fill_failure(tmp_path, files, monkeypatch, fail)
    assert holder.state == "failed"
    assert "read failed" in holder.error and "Input/output error" in holder.error
    assert holder.exit_code == 1


def test_a_file_that_ends_early_ends_the_fill(tmp_path, files, monkeypatch):
    holder = _fill_failure(
        tmp_path, files, monkeypatch, lambda fd, view, offset, need: need // 2
    )
    assert holder.state == "failed"
    assert "ended at byte" in holder.error
