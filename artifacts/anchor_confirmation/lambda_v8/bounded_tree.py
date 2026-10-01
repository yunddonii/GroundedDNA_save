"""Bounded source-state checks (audit 736): content hashes ONLY for an explicit, reviewed inventory of
implementation, test and harness sources; every other tracked file is documented by its git index
object id and its stat metadata, and its bytes are never read.

  inventory  -- the generation manifest's closure (its files_sha256 keys) plus explicit extras; each
                must be a tracked file with a reviewed source suffix (.py, .md, .sh, .txt, .ini).
  state      -- {"content": {path: sha256 of the bytes}, "other": {path: [mode, object id, size,
                mtime_ns, inode]}}: content for the inventory, index identity + stat for the rest.
                Comparing two states shows inventory byte equality and unchanged checkout
                membership of the rest; it does NOT prove the rest's working-tree bytes are equal.
  head-clean -- each inventory file's git blob id equals HEAD's, and the index equals HEAD for
                every tracked path (git ls-tree / ls-files read the object store and index only).
CLI: bounded_tree.py head-clean <root> <manifest> [extra ...]   (exit 0 iff clean)
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

SOURCE_SUFFIXES = (".py", ".md", ".sh", ".txt", ".ini")


def _git(root, *args) -> bytes:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=True).stdout


def tracked(root) -> dict:
    """path -> [mode, object id] from the index (no working-tree read)."""
    out = {}
    for line in _git(root, "ls-files", "-s", "-z").split(b"\0"):
        if line:
            meta, path = line.split(b"\t", 1)
            mode, obj, _stage = meta.split()
            out[path.decode()] = [mode.decode(), obj.decode()]
    return out


def inventory(root, manifest, extras=()) -> list:
    files = json.loads(Path(manifest).read_text())["new_generation"]["files_sha256"]
    names = sorted(set(files) | set(extras))
    known = tracked(root)
    bad = [n for n in names if n not in known or not n.endswith(SOURCE_SUFFIXES)]
    if bad:
        raise SystemExit(f"inventory members not tracked or not a reviewed source type: {bad[:6]}")
    return names


def state(root, names) -> dict:
    root = Path(root)
    content = {n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in names}
    other = {}
    for path, (mode, obj) in tracked(root).items():
        if path in content:
            continue
        try:
            st = os.lstat(root / path)
            other[path] = [mode, obj, st.st_size, st.st_mtime_ns, st.st_ino]
        except FileNotFoundError:
            other[path] = [mode, obj, None, None, None]
    return {"content": content, "other": other}


def head_clean(root, names) -> list:
    """Mismatches: inventory blob ids against HEAD, and index against HEAD for every path."""
    head = {}
    for line in _git(root, "ls-tree", "-r", "-z", "HEAD").split(b"\0"):
        if line:
            meta, path = line.split(b"\t", 1)
            mode, _kind, obj = meta.split()
            head[path.decode()] = [mode.decode(), obj.decode()]
    wrong = sorted(p for p, ident in tracked(root).items() if head.get(p) != ident)
    wrong += sorted(p for p in head if p not in tracked(root))
    for n in names:
        data = (Path(root) / n).read_bytes()
        blob = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
        if head.get(n, [None, None])[1] != blob:
            wrong.append(n)
    return sorted(set(wrong))


if __name__ == "__main__":
    command, root, manifest, *extras = sys.argv[1:]
    assert command == "head-clean", "usage: bounded_tree.py head-clean <root> <manifest> [extra ...]"
    mismatches = head_clean(root, inventory(root, manifest, extras))
    print(json.dumps({"head_clean": not mismatches, "mismatches": mismatches[:20]}))
    sys.exit(0 if not mismatches else 1)
