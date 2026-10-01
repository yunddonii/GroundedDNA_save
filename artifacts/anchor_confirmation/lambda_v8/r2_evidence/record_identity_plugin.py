"""pytest plugin: replace RunIdentity._artifact with a recorder that reads nothing, and write the
paths each test would have hashed (files), or whose meta.json it would have hashed (directories)."""
import json, os
import pytest
SEEN = {}
@pytest.fixture(autouse=True)
def _record_identity_artifacts(request, monkeypatch):
    from dna_utils.run_identity import RunIdentity
    paths = SEEN.setdefault(request.node.nodeid, set())
    def stand_in(cls, path):
        if path in (None, ""):
            return ""
        real = os.path.realpath(str(path))
        paths.add(real)
        return f"{real}#recorded"
    monkeypatch.setattr(RunIdentity, "_artifact", classmethod(stand_in))
def pytest_sessionfinish(session):
    out = {}
    for node, paths in SEEN.items():
        for p in paths:
            kind = "file" if os.path.isfile(p) else "dir" if os.path.isdir(p) else "absent"
            out.setdefault(p, {"kind": kind, "tests": 0})["tests"] += 1
    with open(os.environ["RECORD_OUT"], "w") as handle:
        json.dump(out, handle, indent=1, sort_keys=True)
