from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

from models.pretrained_backbone_clip import (
    CLIPBackbone, verify_local_clip_snapshot)


TOKENIZERS = (
    "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
    "special_tokens_map.json",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot(tmp_path: Path):
    revision = "a" * 40
    root = (tmp_path / "models--fixture--clip" / "snapshots" / revision)
    root.mkdir(parents=True)
    (root / "pytorch_model.bin").write_bytes(b"weight")
    (root / "config.json").write_text('{"model_type":"clip"}\n')
    for name in TOKENIZERS:
        (root / name).write_text(name + "\n")
    return root, revision


def test_exact_local_clip_snapshot_identity_and_tamper_refusal(tmp_path: Path):
    root, revision = _snapshot(tmp_path)
    tokenizers = {name: _sha(root / name) for name in TOKENIZERS}
    identity = verify_local_clip_snapshot(
        str(root), checkpoint="fixture/clip", revision=revision,
        weight_file="pytorch_model.bin",
        weight_sha256=_sha(root / "pytorch_model.bin"),
        config_sha256=_sha(root / "config.json"),
        tokenizer_files_sha256=tokenizers,
    )
    assert identity["snapshot_dir"] == str(root.resolve())
    assert identity["local_files_only"] is True
    expected = dict(identity)
    digest = expected.pop("identity_sha256")
    assert digest == hashlib.sha256(json.dumps(
        expected, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()

    (root / "config.json").write_text('{"model_type":"other"}\n')
    with pytest.raises(RuntimeError, match="config.json SHA-256"):
        verify_local_clip_snapshot(
            str(root), checkpoint="fixture/clip", revision=revision,
            weight_file="pytorch_model.bin",
            weight_sha256=_sha(root / "pytorch_model.bin"),
            config_sha256=identity["config_sha256"],
            tokenizer_files_sha256=tokenizers,
        )


def test_snapshot_must_be_exact_revision_directory(tmp_path: Path):
    root, revision = _snapshot(tmp_path)
    tokenizers = {name: _sha(root / name) for name in TOKENIZERS}
    with pytest.raises(RuntimeError, match="does not name revision"):
        verify_local_clip_snapshot(
            str(root), checkpoint="fixture/clip", revision="b" * 40,
            weight_file="pytorch_model.bin",
            weight_sha256=_sha(root / "pytorch_model.bin"),
            config_sha256=_sha(root / "config.json"),
            tokenizer_files_sha256=tokenizers,
        )


def test_phase3_loader_passes_only_exact_local_snapshot(monkeypatch, tmp_path):
    root, revision = _snapshot(tmp_path)
    tokenizers = {name: _sha(root / name) for name in TOKENIZERS}
    calls = []

    class FakeLoaded(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.vision_model = torch.nn.Identity()
            self.text_model = torch.nn.Identity()
            self.config = SimpleNamespace(
                vision_config=SimpleNamespace(hidden_size=768),
                text_config=SimpleNamespace(hidden_size=512),
                projection_dim=512,
            )

    class FakeCLIPModel:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            calls.append((name, kwargs))
            return FakeLoaded()

    monkeypatch.setitem(
        sys.modules, "transformers", SimpleNamespace(CLIPModel=FakeCLIPModel))
    backbone = CLIPBackbone("fixture/clip", phase3_snapshot={
        "checkpoint": "fixture/clip", "snapshot_dir": str(root),
        "revision": revision, "weight_file": "pytorch_model.bin",
        "weight_sha256": _sha(root / "pytorch_model.bin"),
        "config_sha256": _sha(root / "config.json"),
        "tokenizer_files_sha256": tokenizers,
    })
    assert calls == [(str(root.resolve()), {"local_files_only": True})]
    assert backbone.runtime_hf_identity["identity_sha256"]
