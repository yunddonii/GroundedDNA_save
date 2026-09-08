"""OpenAI-CLIP dual-encoder pretrained backbone.

Mirrors ``models/pretrained_backbone.SigLIP2Backbone`` so the downstream model
code (``VisualEncoder`` / ``TextEncoder`` / ``SigLIP2SemanticOTModel``) can
treat the two backbones interchangeably. The two encoders held by the parent
model both keep a reference to the SAME ``CLIPBackbone`` instance.

Why a parallel module (not a branch inside `pretrained_backbone.py`):
    The two HuggingFace checkpoints come from different model families
    (Siglip2Model vs CLIPModel). Their `vision_model` / `text_model`
    sub-modules expose different attributes (e.g. SigLIP2 uses MAP-pool, CLIP
    uses [CLS] pooling; their `last_hidden_state` shapes differ by the CLS
    token). Isolating them in separate files prevents one model's quirks from
    leaking into the other's path, and matches the project's "new file
    centric" change policy.

What this module does NOT do:
    - tokenize text (caller supplies ``input_ids`` / ``attention_mask``
      produced by the CLIP tokenizer)
    - preprocess images (caller supplies CLIP-normalized ``pixel_values``;
      the extraction script applies the CLIP-specific mean/std).
"""

from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
from typing import Mapping, Optional

import torch
import torch.nn as nn


DEFAULT_CLIP_BACKBONE = "openai/clip-vit-base-patch16"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_local_clip_snapshot(
    snapshot_dir: str,
    *,
    checkpoint: str,
    revision: str,
    weight_file: str,
    weight_sha256: str,
    config_sha256: str,
    tokenizer_files_sha256: Mapping[str, str],
) -> dict:
    """Verify all files the sealed Phase-3 CLIP identity commits to.

    The returned mapping is deterministic and is stored on the backbone for
    the trainer to compare with its launch authority.  It performs only local
    file reads and never resolves a Hugging Face branch or contacts the hub.
    """
    if _REVISION_RE.fullmatch(str(revision)) is None:
        raise RuntimeError("Phase-3 CLIP revision must be immutable 40-hex")
    root = Path(snapshot_dir).expanduser()
    if not root.is_absolute() or not root.is_dir():
        raise RuntimeError("Phase-3 CLIP snapshot must be an existing absolute directory")
    root = root.resolve(strict=True)
    if root.name != revision or root.parent.name != "snapshots":
        raise RuntimeError(
            f"Phase-3 CLIP snapshot path does not name revision {revision}"
        )
    expected_repo = "models--" + checkpoint.replace("/", "--")
    if root.parent.parent.name != expected_repo:
        raise RuntimeError(
            "Phase-3 CLIP snapshot repository does not match checkpoint"
        )
    if weight_file not in ("model.safetensors", "pytorch_model.bin"):
        raise RuntimeError("Phase-3 CLIP weight filename is unsupported")
    expected = {weight_file: weight_sha256, "config.json": config_sha256,
                **dict(tokenizer_files_sha256)}
    if set(tokenizer_files_sha256) != {
        "tokenizer.json", "tokenizer_config.json", "vocab.json",
        "merges.txt", "special_tokens_map.json",
    }:
        raise RuntimeError("Phase-3 CLIP tokenizer inventory is not the exact sealed set")
    malformed = sorted(name for name, digest in expected.items()
                       if not isinstance(digest, str)
                       or _SHA256_RE.fullmatch(digest) is None)
    if malformed:
        raise RuntimeError(f"Phase-3 CLIP expected SHA-256 is malformed for {malformed}")
    actual = {}
    for name, wanted in sorted(expected.items()):
        path = root / name
        try:
            resolved = path.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise RuntimeError(f"Phase-3 CLIP snapshot is missing {name}: {error}") from error
        if not resolved.is_file():
            raise RuntimeError(f"Phase-3 CLIP snapshot input is not regular: {name}")
        digest = _sha256(resolved)
        if digest != wanted:
            raise RuntimeError(
                f"Phase-3 CLIP {name} SHA-256 differs from the input seal"
            )
        actual[name] = digest
    tokenizer_values = {
        name: actual[name] for name in sorted(tokenizer_files_sha256)
    }
    tokenizer_set_sha256 = hashlib.sha256(
        json.dumps(tokenizer_values, sort_keys=True,
                   separators=(",", ":")).encode()
    ).hexdigest()
    identity = {
        "checkpoint": checkpoint,
        "revision": revision,
        "snapshot_dir": str(root),
        "weight_file": weight_file,
        "weight_sha256": actual[weight_file],
        "config_file": "config.json",
        "config_sha256": actual["config.json"],
        "tokenizer_files_sha256": tokenizer_values,
        "tokenizer_set_sha256": tokenizer_set_sha256,
        "local_files_only": True,
    }
    identity["identity_sha256"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return identity


class CLIPBackbone(nn.Module):
    """Thin holder around ``transformers.CLIPModel``.

    API parity with ``SigLIP2Backbone``:
        self.model              the full CLIPModel (vision + text + 2 projections)
        self.vision_hidden_dim  H_v -- vision_model last_hidden_state dim
        self.text_hidden_dim    H_t -- text_model last_hidden_state dim
        self.projection_dim     D   -- shared image/text embedding dim returned
                                       by get_image_features / get_text_features

    For ``openai/clip-vit-base-patch16``:
        H_v = 768, H_t = 512, D = 512
    """

    def __init__(self, pretrained_name: str = DEFAULT_CLIP_BACKBONE, *,
                 phase3_snapshot: Optional[Mapping[str, object]] = None) -> None:
        super().__init__()
        self.pretrained_name = pretrained_name
        self.is_fgclip = pretrained_name.startswith("qihoo360/fg-clip")

        if self.is_fgclip:
            self.model = _load_fgclip(pretrained_name)
        else:
            try:
                from transformers import CLIPModel
            except ImportError as e:
                raise ImportError(
                    "transformers is required to load CLIP. "
                    "Install via `pip install 'transformers>=4.45'`."
                ) from e
            if phase3_snapshot is None:
                self.runtime_hf_identity = None
                self.model = CLIPModel.from_pretrained(pretrained_name)
            else:
                checkpoint = str(phase3_snapshot["checkpoint"])
                if checkpoint != pretrained_name:
                    raise RuntimeError(
                        "Phase-3 CLIP checkpoint differs from --clip_backbone"
                    )
                tokenizers = phase3_snapshot.get("tokenizer_files_sha256")
                if not isinstance(tokenizers, Mapping):
                    raise RuntimeError("Phase-3 CLIP tokenizer authority is missing")
                self.runtime_hf_identity = verify_local_clip_snapshot(
                    str(phase3_snapshot["snapshot_dir"]),
                    checkpoint=checkpoint,
                    revision=str(phase3_snapshot["revision"]),
                    weight_file=str(phase3_snapshot["weight_file"]),
                    weight_sha256=str(phase3_snapshot["weight_sha256"]),
                    config_sha256=str(phase3_snapshot["config_sha256"]),
                    tokenizer_files_sha256=tokenizers,
                )
                # The immutable local snapshot path is the only model source.
                # Passing the checkpoint name here would consult a mutable
                # refs/main even under local-files-only mode.
                self.model = CLIPModel.from_pretrained(
                    self.runtime_hf_identity["snapshot_dir"],
                    local_files_only=True,
                )

        cfg = self.model.config
        self.vision_hidden_dim: int = int(cfg.vision_config.hidden_size)
        self.text_hidden_dim:   int = int(cfg.text_config.hidden_size)
        self.projection_dim:    int = int(cfg.projection_dim)

    # convenience accessors so encoders can stay readable
    @property
    def vision_model(self) -> nn.Module:
        return self.model.vision_model

    @property
    def text_model(self) -> nn.Module:
        return self.model.text_model

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device


def build_clip_backbone(pretrained_name: Optional[str] = None, *,
                        phase3_snapshot: Optional[Mapping[str, object]] = None
                        ) -> CLIPBackbone:
    """Build a shared CLIP backbone. Pass the same instance into both encoders."""
    if pretrained_name is None:
        pretrained_name = DEFAULT_CLIP_BACKBONE
    return CLIPBackbone(pretrained_name=pretrained_name,
                        phase3_snapshot=phase3_snapshot)


# ---------------------------------------------------------------------------
# FG-CLIP loader. qihoo360/fg-clip-{base,large} ships a `modeling_fgclip.py`
# whose `FGCLIPModel(config)` constructor (a) `isinstance`-checks the sub-
# configs against classes loaded under a separate dynamic-module revision
# path (always fails) and (b) leaves `position_ids` buffers in the saved
# safetensors with garbage values that crash forward.
# We:
#   1. Convert text_config / vision_config from dict to proper Config objects.
#   2. Reset position_ids buffers to arange(N) post-load.
#   3. Wrap `text_model.forward` so it defaults `walk_short_pos=True` — the
#      FG-CLIP short-caption path (matches standard CLIP behaviour and what
#      our v6b CUB / Qwen captions need; FILIP long-caption path is unused).
# ---------------------------------------------------------------------------
def _load_fgclip(pretrained_name: str):
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    FGCLIPModel  = get_class_from_dynamic_module("modeling_fgclip.FGCLIPModel",  pretrained_name)
    FGCLIPConfig = get_class_from_dynamic_module("modeling_fgclip.FGCLIPConfig", pretrained_name)
    CLIPTextConfig   = get_class_from_dynamic_module("modeling_clip.CLIPTextConfig",   pretrained_name)
    CLIPVisionConfig = get_class_from_dynamic_module("modeling_clip.CLIPVisionConfig", pretrained_name)

    cfg = FGCLIPConfig.from_pretrained(pretrained_name)
    if isinstance(cfg.text_config, dict):
        cfg.text_config = CLIPTextConfig(**cfg.text_config)
    if isinstance(cfg.vision_config, dict):
        cfg.vision_config = CLIPVisionConfig(**cfg.vision_config)

    model = FGCLIPModel.from_pretrained(pretrained_name, config=cfg)

    img = cfg.vision_config.image_size
    ps  = cfg.vision_config.patch_size
    nv  = (img // ps) ** 2 + 1
    model.vision_model.embeddings.position_ids = torch.arange(nv).expand(1, -1)
    nt = model.text_model.embeddings.position_embedding.weight.shape[0]
    model.text_model.embeddings.position_ids = torch.arange(nt).expand(1, -1)

    _orig_text_forward = model.text_model.forward
    def _text_forward(*args, **kwargs):
        kwargs.setdefault("walk_short_pos", True)
        return _orig_text_forward(*args, **kwargs)
    model.text_model.forward = _text_forward

    return model
