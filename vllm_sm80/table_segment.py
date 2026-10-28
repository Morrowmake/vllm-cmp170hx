# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read-only host images of checkpoint files, held by one process across
server restarts.

Large host-resident tables (the DeepSeek V4.1 Engram tables: two files of
~95 GiB) take minutes to read from storage at every server start. A holder
process reads each named file once into its own sealed memory file and hands
read-only descriptors of them to the servers that attach, so a restart maps
the bytes instead of reading them again. The module needs only the standard
library and runs as a script:

    python3 table_segment.py hold --socket SOCK --label LABEL \\
        --file PATH=SHA256 [--file PATH=SHA256 ...] [--readers N]
    python3 table_segment.py status --socket SOCK
    python3 table_segment.py release --socket SOCK [--mode drain|idle|force]

``hold`` creates the socket at once (state ``filling``), copies each file
whole into a ``memfd`` through direct reads, checks that its sha256 equals the
given digest and that the file did not change while it was read (path,
device, inode, size, modification time), seals the memory file against any
write or resize, and then serves (state ``ready``). It runs in the foreground
until it is released, and exits 3 when a file fails its checks.

Protocol: ``AF_UNIX`` ``SOCK_SEQPACKET``; a client sends one JSON object and
the holder answers with one.

- ``{"op": "status"}``: the state, the files with their identity and digest,
  the manifest hash (sha256 of the label and each file's name as given,
  size and digest), the bytes filled, the attached clients and whether huge
  pages backed the fill. The name is the last part of the path named on the
  command line, so a file reached through a link is hashed under its name.
- ``{"op": "release", "mode": ...}``: ``idle`` exits only when no client is
  attached; ``drain`` exits now when none is, otherwise refuses new attaches
  and exits when the last client detaches; ``force`` exits now (attached
  clients keep valid mappings; the memory is freed when the last unmaps).
  The answer says whether the holder exits now.
- ``{"op": "attach", "files": [identity, ...]}``: answered once the fill is
  done, with the read-only descriptors of the named files (``SCM_RIGHTS``),
  only if every identity equals the held file's. The connection stays open
  as the client's lease; closing it (or the client's exit) detaches.

``attach`` is the client side: ``Attachment(socket, paths)`` and
``Attachment.map(path, offset, nbytes)``, a read-only shared mapping of a
byte range of a held file, populated at once so later reads take no page
faults.
"""

import argparse
import errno
import fcntl
import hashlib
import json
import mmap
import os
import selectors
import signal
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass

VERSION = 1
MAX_MESSAGE = 64 << 10
# The fill reads aligned chunks of this size, several at a time.
CHUNK_BYTES = 64 << 20
DEFAULT_READERS = 8
_ALIGN = 4096
# Memory files are sized to whole 2 MiB pages so huge pages can back them.
_HUGE = 2 << 20
# Linux fcntl memfd seal values; some Python builds do not name them.
F_ADD_SEALS = getattr(fcntl, "F_ADD_SEALS", 1033)
F_GET_SEALS = getattr(fcntl, "F_GET_SEALS", 1034)
_SEALS = (
    getattr(fcntl, "F_SEAL_SEAL", 0x1)
    | getattr(fcntl, "F_SEAL_SHRINK", 0x2)
    | getattr(fcntl, "F_SEAL_GROW", 0x4)
    | getattr(fcntl, "F_SEAL_WRITE", 0x8)
)
# madvise(MADV_POPULATE_READ), Linux 5.14; not every Python names it.
_MADV_POPULATE_READ = getattr(mmap, "MADV_POPULATE_READ", 22)
# Received descriptors close on exec, as every other descriptor here.
_MSG_CMSG_CLOEXEC = getattr(socket, "MSG_CMSG_CLOEXEC", 0x40000000)
EXIT_VERIFY = 3
EXIT_NO_HOLDER = 4


class SegmentError(RuntimeError):
    """A holder that is absent, refuses a request or fails its checks."""


class _FillError(RuntimeError):
    """A file that could not be read whole."""


def file_identity(path: str) -> dict:
    """What the holder records of a file and a client must match: its real
    path, device, inode, size and modification time."""
    real = os.path.realpath(path)
    st = os.stat(real)
    return {
        "path": real,
        "dev": st.st_dev,
        "ino": st.st_ino,
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
    }


def manifest_hash(label: str, files: list[dict]) -> str:
    """sha256 of the label and each file's name, size and digest, in order,
    as the JSON of ``{label, files: [{name, size, sha256}, ...]}`` with
    sorted keys and no spaces."""
    doc = {
        "label": label,
        "files": [
            {"name": f["name"], "size": f["size"], "sha256": f["sha256"]} for f in files
        ],
    }
    text = json.dumps(doc, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def _round_up(value: int, align: int) -> int:
    return -(-value // align) * align


def _send(sock: socket.socket, obj: dict, fds: list[int] | None = None) -> None:
    data = json.dumps(obj).encode()
    if fds:
        socket.send_fds(sock, [data], fds)
    else:
        sock.send(data)


def _recv(sock: socket.socket, maxfds: int = 0) -> tuple[dict, list[int]]:
    fds: list[int] = []
    if maxfds:
        # socket.recv_fds drops its flags argument: recvmsg directly, so the
        # descriptors arrive close-on-exec.
        data, ancdata, _, _ = sock.recvmsg(
            MAX_MESSAGE, socket.CMSG_LEN(maxfds * 4), _MSG_CMSG_CLOEXEC
        )
        for level, kind, payload in ancdata:
            if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                fds.extend(
                    int.from_bytes(payload[i:i + 4], sys.byteorder, signed=True)
                    for i in range(0, len(payload) - len(payload) % 4, 4)
                )
    else:
        data = sock.recv(MAX_MESSAGE)
    if not data:
        for fd in fds:
            os.close(fd)
        raise EOFError("the table segment holder closed the connection")
    return json.loads(data), list(fds)


def _connect(socket_path: str, timeout: float | None) -> socket.socket:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET | socket.SOCK_CLOEXEC)
    sock.settimeout(timeout)
    try:
        sock.connect(socket_path)
    except (FileNotFoundError, ConnectionRefusedError) as exc:
        sock.close()
        raise SegmentError(f"no table segment holder answers at {socket_path}: {exc}") from exc
    return sock


def request(socket_path: str, obj: dict, timeout: float | None = 10.0) -> dict:
    """One request and its answer (status or release)."""
    with _connect(socket_path, timeout) as sock:
        _send(sock, obj)
        return _recv(sock)[0]


# ---------------------------------------------------------------------------
# Holder
# ---------------------------------------------------------------------------


@dataclass
class HeldFile:
    path: str
    expected_sha256: str
    identity: dict
    # The last part of the path as given (a link's own name): the manifest's.
    name: str
    memfd: int = -1
    readonly_fd: int = -1
    segment_bytes: int = 0
    sha256: str = ""
    done: int = 0


class _Stopped(Exception):
    """The fill of a file stopped because another failed or the holder ends."""


def _read_into(fd: int, view: memoryview, offset: int, need: int) -> int:
    """Read from ``offset`` into ``view`` until ``need`` bytes are in or the
    file ends; return the bytes read."""
    got = 0
    while got < need:
        count = os.preadv(fd, [view[got:]], offset + got)
        if count == 0:
            break
        got += count
    return got


def _open_for_fill(path: str) -> tuple[int, bool]:
    try:
        return os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECT), True
    except OSError as exc:
        if exc.errno != errno.EINVAL:
            raise
    return os.open(path, os.O_RDONLY | os.O_CLOEXEC), False


def _shmem_pmd_bytes() -> int:
    """Bytes of shared memory this process maps with huge pages."""
    try:
        with open("/proc/self/smaps_rollup") as file:
            for line in file:
                if line.startswith("ShmemPmdMapped:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


class Holder:
    def __init__(self, socket_path: str, label: str, files: list[tuple[str, str]],
                 readers: int = DEFAULT_READERS, log=None):
        if not files:
            raise ValueError("a table segment holds at least one file")
        if readers < 1:
            raise ValueError("readers must be at least 1")
        self.socket_path = socket_path
        self.label = label
        self.readers = readers
        self.log = log or (lambda msg: print(f"table_segment: {msg}", file=sys.stderr, flush=True))
        self.files = [
            HeldFile(os.path.realpath(p), sha.lower(), file_identity(p), os.path.basename(p))
            for p, sha in files
        ]
        if len({f.path for f in self.files}) != len(self.files):
            raise ValueError("a file is named twice")
        self.state = "filling"
        self.error = ""
        self.huge_pages = False
        self.fill_seconds: float | None = None
        self.clients: dict[socket.socket, int] = {}
        self.pending: list[tuple[socket.socket, dict]] = []
        self.exit_code: int | None = None
        self._lock = threading.Lock()
        self._wake_r, self._wake_w = os.pipe2(os.O_CLOEXEC | os.O_NONBLOCK)
        self._listener: socket.socket | None = None
        self._listener_ino = 0
        self._filler: threading.Thread | None = None
        # Set when a file's fill fails or the holder ends: the other fills stop.
        self._stop = threading.Event()

    # -- fill ------------------------------------------------------------

    def _fill_file(self, held: HeldFile, pool: ThreadPoolExecutor) -> None:
        size = held.identity["size"]
        held.segment_bytes = _round_up(max(size, 1), _HUGE)
        held.memfd = os.memfd_create(
            os.path.basename(held.path), os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING
        )
        os.ftruncate(held.memfd, held.segment_bytes)
        mapping = mmap.mmap(held.memfd, held.segment_bytes, flags=mmap.MAP_SHARED,
                            prot=mmap.PROT_READ | mmap.PROT_WRITE)
        failure = ""
        try:
            try:
                mapping.madvise(mmap.MADV_HUGEPAGE)
            except (AttributeError, OSError):
                pass
            fd, direct = _open_for_fill(held.path)
            view = memoryview(mapping)
            futures = []
            try:
                def read(begin: int) -> int:
                    if self._stop.is_set():
                        raise _Stopped()
                    end = min(begin + CHUNK_BYTES, size)
                    length = _round_up(end - begin, _ALIGN)
                    chunk = view[begin:begin + length]
                    try:
                        got = _read_into(fd, chunk, begin, end - begin)
                    finally:
                        chunk.release()
                    if not direct:
                        os.posix_fadvise(fd, begin, length, os.POSIX_FADV_DONTNEED)
                    return begin + min(got, end - begin)

                digest = hashlib.sha256()
                window = 2 * self.readers
                starts = list(range(0, size, CHUNK_BYTES))
                futures = [pool.submit(read, s) for s in starts[:window]]
                for index, begin in enumerate(starts):
                    try:
                        end = futures[index].result()
                    except _Stopped:
                        failure = "stopped"
                        break
                    except Exception as exc:  # noqa: BLE001 - a read error ends the fill
                        failure = f"{held.path}: read failed: {exc}"
                        break
                    if index + window < len(starts):
                        futures.append(pool.submit(read, starts[index + window]))
                    digest.update(view[begin:end])
                    held.done = end
                    if end < min(begin + CHUNK_BYTES, size):
                        failure = f"{held.path} ended at byte {end} of {size}"
                        break
                held.sha256 = digest.hexdigest()
            finally:
                if failure:
                    self._stop.set()
                # No read may still use the descriptor or the view.
                for future in futures:
                    future.cancel()
                wait(futures)
                futures.clear()
                view.release()
                os.close(fd)
            if _shmem_pmd_bytes() > 0:
                self.huge_pages = True
        finally:
            mapping.close()
        if failure == "stopped":
            raise _Stopped()
        if failure:
            raise _FillError(failure)

    def fill(self) -> None:
        start = time.monotonic()
        try:
            with ThreadPoolExecutor(self.readers) as pool, ThreadPoolExecutor(len(self.files)) as files:
                futures = [files.submit(self._fill_file, f, pool) for f in self.files]
                wait(futures)
            errors = [f.exception() for f in futures if f.exception() is not None]
            real = [e for e in errors if not isinstance(e, _Stopped)]
            if real:
                raise real[0]
            if errors:
                raise _FillError("the fill stopped: the holder is ending")
            for held in self.files:
                if held.sha256 != held.expected_sha256:
                    raise SegmentError(
                        f"{held.path}: sha256 {held.sha256}, expected {held.expected_sha256}")
                if file_identity(held.path) != held.identity:
                    raise SegmentError(f"{held.path} changed while it was read")
                fcntl.fcntl(held.memfd, F_ADD_SEALS, _SEALS)
                held.readonly_fd = os.open(f"/proc/self/fd/{held.memfd}", os.O_RDONLY | os.O_CLOEXEC)
            self.fill_seconds = time.monotonic() - start
            total = sum(f.identity["size"] for f in self.files)
            self.log(f"filled {len(self.files)} file(s), {total / (1 << 30):.2f} GiB in "
                     f"{self.fill_seconds:.1f} s; sealed; huge pages "
                     f"{'on' if self.huge_pages else 'off'}; manifest {self.manifest}")
            with self._lock:
                if self.state == "filling":
                    self.state = "ready"
        except BaseException as exc:  # noqa: BLE001 - reported to the clients and the exit code
            with self._lock:
                self.state = "failed"
                self.error = str(exc)
                if self.exit_code is None:
                    self.exit_code = EXIT_VERIFY if isinstance(exc, SegmentError) else 1
            self.log(f"fill failed: {exc}")
        finally:
            os.write(self._wake_w, b"x")

    # -- serving ---------------------------------------------------------

    @property
    def manifest(self) -> str:
        return manifest_hash(self.label, [
            {"name": f.name, "size": f.identity["size"], "sha256": f.expected_sha256}
            for f in self.files])

    def status(self) -> dict:
        return {
            "ok": True,
            "version": VERSION,
            "state": self.state,
            "error": self.error,
            "pid": os.getpid(),
            "label": self.label,
            "manifest": self.manifest,
            "files": [dict(f.identity, sha256=f.expected_sha256) for f in self.files],
            "bytes_total": sum(f.identity["size"] for f in self.files),
            "bytes_done": sum(f.done for f in self.files),
            "segment_bytes": sum(_round_up(max(f.identity["size"], 1), _HUGE) for f in self.files),
            "clients": [{"pid": pid} for pid in self.clients.values()],
            "pending": len(self.pending),
            "huge_pages": self.huge_pages,
            "fill_seconds": self.fill_seconds,
        }

    def _bind(self) -> socket.socket:
        try:
            request(self.socket_path, {"op": "status"}, timeout=2.0)
        except SegmentError:
            pass
        else:
            raise SegmentError(f"another table segment holder serves {self.socket_path}")
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        os.makedirs(os.path.dirname(os.path.abspath(self.socket_path)), exist_ok=True)
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET | socket.SOCK_CLOEXEC)
        sock.bind(self.socket_path)
        os.chmod(self.socket_path, 0o600)
        sock.listen(64)
        sock.setblocking(False)
        self._listener_ino = os.stat(self.socket_path).st_ino
        return sock

    def _attach(self, conn: socket.socket, msg: dict) -> bool:
        """Answer an attach once ready; True when the connection becomes a
        client lease."""
        if self.state != "ready":
            _send(conn, {"ok": False, "error": f"table segment holder is {self.state}"
                         + (f": {self.error}" if self.error else "")})
            return False
        if msg.get("version") != VERSION:
            _send(conn, {"ok": False, "error": f"protocol version {msg.get('version')}, holder speaks {VERSION}"})
            return False
        held = {f.path: f for f in self.files}
        chosen = []
        for want in msg.get("files", []):
            f = held.get(want.get("path"))
            if f is None:
                _send(conn, {"ok": False, "error": f"the holder does not hold {want.get('path')}"})
                return False
            if f.identity != want:
                _send(conn, {"ok": False, "error": (
                    f"{f.path} is not the file the holder read (held {f.identity}, "
                    f"now {want}): hold it again")})
                return False
            chosen.append(f)
        if not chosen:
            _send(conn, {"ok": False, "error": "attach names no file"})
            return False
        _send(conn, {"ok": True, "manifest": self.manifest,
                     "files": [{"path": f.path, "size": f.identity["size"],
                                "segment_bytes": f.segment_bytes} for f in chosen]},
              [f.readonly_fd for f in chosen])
        return True

    def _release(self, mode: str) -> bool:
        if mode not in ("idle", "drain", "force"):
            raise ValueError(f"release mode must be idle, drain or force, not {mode!r}")
        # A server waiting for the fill counts as attached for an idle release.
        busy = bool(self.clients) or (mode == "idle" and bool(self.pending))
        if mode == "force" or not busy:
            self.exit_code = 0
            self._stop.set()
            return True
        if mode == "drain":
            with self._lock:
                self.state = "draining"
        return False

    def _handle(self, sel: selectors.BaseSelector, conn: socket.socket) -> None:
        try:
            data = conn.recv(MAX_MESSAGE)
        except OSError:
            data = b""
        if not data:
            sel.unregister(conn)
            pid = self.clients.pop(conn, None)
            self.pending = [(c, m) for c, m in self.pending if c is not conn]
            conn.close()
            if pid is not None:
                self.log(f"client {pid} detached ({len(self.clients)} left)")
                if self.state == "draining" and not self.clients:
                    self.exit_code = 0
            return
        if conn in self.clients:
            return  # a client lease sends nothing more
        try:
            msg = json.loads(data)
            op = msg.get("op")
            if op == "status":
                _send(conn, self.status())
            elif op == "release":
                released = self._release(msg.get("mode", "drain"))
                _send(conn, {"ok": True, "released": released, "state": self.state,
                             "clients": list(self.clients.values())})
                self.log(f"release {msg.get('mode', 'drain')}: "
                         f"{'exiting' if released else 'kept'} ({len(self.clients)} client(s))")
            elif op == "attach":
                if self.state == "filling":
                    self.pending.append((conn, msg))
                    return
                self._finish_attach(sel, conn, msg)
                return
            else:
                _send(conn, {"ok": False, "error": f"unknown op {op!r}"})
        except (ValueError, AttributeError) as exc:
            _send(conn, {"ok": False, "error": str(exc)})
        except OSError:
            pass

    def _finish_attach(self, sel: selectors.BaseSelector, conn: socket.socket, msg: dict) -> None:
        try:
            if self._attach(conn, msg):
                creds = conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
                pid = int.from_bytes(creds[:4], sys.byteorder)
                self.clients[conn] = pid
                self.log(f"client {pid} attached ({len(self.clients)} attached)")
        except OSError:
            pass

    def serve(self) -> int:
        self._listener = self._bind()
        self.log(f"serving {self.socket_path} (pid {os.getpid()}), filling "
                 f"{len(self.files)} file(s) with {self.readers} readers")
        self._filler = threading.Thread(target=self.fill, name="table-segment-fill", daemon=True)
        self._filler.start()
        sel = selectors.DefaultSelector()
        sel.register(self._listener, selectors.EVENT_READ, "listen")
        sel.register(self._wake_r, selectors.EVENT_READ, "wake")

        def stop(signum, frame):
            self.exit_code = 0
            os.write(self._wake_w, b"x")

        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGTERM, stop)
            signal.signal(signal.SIGINT, stop)
        try:
            while self.exit_code is None:
                for key, _ in sel.select():
                    if key.data == "listen":
                        try:
                            conn, _ = self._listener.accept()
                        except BlockingIOError:
                            continue
                        conn.setblocking(True)
                        sel.register(conn, selectors.EVENT_READ, "conn")
                    elif key.data == "wake":
                        try:
                            os.read(self._wake_r, 64)
                        except BlockingIOError:
                            pass
                        if self.state != "filling":
                            pending, self.pending = self.pending, []
                            for conn, msg in pending:
                                self._finish_attach(sel, conn, msg)
                    else:
                        self._handle(sel, key.fileobj)
                if self.state == "failed" and self.exit_code is None:
                    self.exit_code = EXIT_VERIFY
        finally:
            self._shutdown(sel)
        return self.exit_code

    def _shutdown(self, sel: selectors.BaseSelector) -> None:
        for conn, _ in self.pending:
            try:
                _send(conn, {"ok": False, "error": f"table segment holder ended ({self.state}"
                             + (f": {self.error})" if self.error else ")")})
            except OSError:
                pass
        for key in list(sel.get_map().values()):
            if key.data == "conn":
                key.fileobj.close()
        sel.close()
        if self._listener is not None:
            self._listener.close()
            try:
                if os.stat(self.socket_path).st_ino == self._listener_ino:
                    os.unlink(self.socket_path)
            except FileNotFoundError:
                pass
        for held in self.files:
            for fd in (held.readonly_fd, held.memfd):
                if fd >= 0:
                    os.close(fd)
        if self._filler is None or not self._filler.is_alive():
            # A fill still running writes the pipe when it ends.
            os.close(self._wake_r)
            os.close(self._wake_w)
        self.log(f"exit {self.exit_code}")


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class Attachment:
    """A client lease on a holder's files: the open connection and the
    read-only descriptors of the files it attached."""

    def __init__(self, socket_path: str, paths: list[str], log=None):
        identities = [file_identity(p) for p in paths]
        try:
            state = request(socket_path, {"op": "status"})
        except SegmentError:
            raise
        if state.get("state") == "filling" and log is not None:
            log("waiting for the table segment at %s to fill (%.1f of %.1f GiB)" % (
                socket_path, state["bytes_done"] / (1 << 30), state["bytes_total"] / (1 << 30)))
        self.socket_path = socket_path
        self.manifest = state.get("manifest", "")
        self._sock = _connect(socket_path, None)
        fds: list[int] = []
        try:
            _send(self._sock, {"op": "attach", "version": VERSION, "files": identities})
            reply, fds = _recv(self._sock, maxfds=len(identities))
            if not reply.get("ok"):
                raise SegmentError(f"table segment at {socket_path} refused the attach: {reply.get('error')}")
            if len(fds) != len(identities):
                raise SegmentError("table segment holder sent the wrong number of descriptors")
        except BaseException:
            for fd in fds:
                os.close(fd)
            self._sock.close()
            raise
        self.manifest = reply["manifest"]
        self._fds = {ident["path"]: (fd, info["size"])
                     for ident, fd, info in zip(identities, fds, reply["files"])}

    def map(self, path: str, offset: int, nbytes: int) -> tuple[mmap.mmap, int]:
        """A read-only shared mapping holding bytes [offset, offset + nbytes)
        of the held image of ``path``, and the offset of that range in it.
        The range is populated now, so reads of it take no page faults."""
        fd, size = self._fds[os.path.realpath(path)]
        if offset < 0 or nbytes <= 0 or offset + nbytes > size:
            raise ValueError(f"bytes [{offset}, {offset + nbytes}) are outside {path} ({size} bytes)")
        start = offset - offset % mmap.ALLOCATIONGRANULARITY
        mapping = mmap.mmap(fd, offset + nbytes - start, flags=mmap.MAP_SHARED,
                            prot=mmap.PROT_READ, offset=start)
        for advice in (getattr(mmap, "MADV_HUGEPAGE", None), _MADV_POPULATE_READ):
            if advice is None:
                continue
            try:
                mapping.madvise(advice)
            except OSError:
                pass  # huge pages off, or a kernel without POPULATE_READ
        return mapping, offset - start

    def close(self) -> None:
        """Detach (mappings already made stay valid)."""
        for fd, _ in self._fds.values():
            os.close(fd)
        self._fds = {}
        self._sock.close()


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def _parse_file(value: str) -> tuple[str, str]:
    path, sep, sha = value.rpartition("=")
    if not sep or len(sha) != 64 or any(c not in "0123456789abcdefABCDEF" for c in sha):
        raise argparse.ArgumentTypeError(f"--file wants PATH=SHA256, not {value!r}")
    return path, sha


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="table_segment", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    hold = sub.add_parser("hold", help="hold files until released")
    hold.add_argument("--socket", required=True)
    hold.add_argument("--label", required=True)
    hold.add_argument("--file", action="append", type=_parse_file, required=True)
    hold.add_argument("--readers", type=int, default=DEFAULT_READERS)
    status = sub.add_parser("status", help="print the holder's state as JSON")
    status.add_argument("--socket", required=True)
    release = sub.add_parser("release", help="release the holder")
    release.add_argument("--socket", required=True)
    release.add_argument("--mode", choices=("drain", "idle", "force"), default="drain")
    args = parser.parse_args(argv)
    if args.cmd == "hold":
        try:
            holder = Holder(args.socket, args.label, args.file, args.readers)
            code = holder.serve()
        except (SegmentError, ValueError, OSError) as exc:
            print(f"table_segment: {exc}", file=sys.stderr)
            code = 1
        # A release during the fill must not wait for the reader threads.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
    try:
        msg = {"op": "status"} if args.cmd == "status" else {"op": "release", "mode": args.mode}
        print(json.dumps(request(args.socket, msg), indent=1))
        return 0
    except SegmentError as exc:
        print(f"table_segment: {exc}", file=sys.stderr)
        return EXIT_NO_HOLDER


if __name__ == "__main__":
    sys.exit(main())
