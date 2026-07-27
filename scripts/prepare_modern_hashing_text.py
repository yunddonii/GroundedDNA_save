"""Prepare source-traceable CLIP text assets for modern hashing baselines.

The script intentionally does not vendor WordNet or benchmark taxonomy files.
Those resources have their own licenses and must be supplied explicitly.  It
uses the exact Hugging Face CLIP checkpoint recorded in a visual cache's
``meta.json`` so image/text dimensions and the local-token projection agree.

Examples
--------
UMRCH taxonomy and visual-token adapter::

    python scripts/prepare_modern_hashing_text.py umrch \
      --cache_dir cache/flickr25k_clip_v4plus_qwen3_tokens \
      --terms_file path/to/flickr24_class_names.txt \
      --out_prefix artifacts/umrch_flickr24

DUH-EG already-selected WordNet nouns::

    python scripts/prepare_modern_hashing_text.py duheg-selected \
      --cache_dir cache/flickr25k_clip_v4plus_qwen3_tokens \
      --terms_file path/to/selected_wordnet_nouns.txt \
      --out_prefix artifacts/duheg_selected_nouns

For DUH-EG, this encodes an already selected noun list; the paper's WordNet
filtering/clustering stage must be run separately.  Passing arbitrary or
dataset class nouns would change the method and is rejected by the protocol
documentation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from baseline.asset_provenance import (
    ASSET_CONTRACT_VERSION,
    DUHEG_PROMPT_TEMPLATES,
    classify_term_information_condition,
    read_ordered_terms,
    sha256_ordered_terms,
)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _encode(model, tokenizer, prompts: list[str], device: str,
            batch_size: int, *, normalize: bool = True) -> torch.Tensor:
    pieces = []
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            tokens = tokenizer(
                prompts[start:start + batch_size], padding=True,
                truncation=True, return_tensors='pt',
            ).to(device)
            output = model.get_text_features(**tokens)
            # transformers<=4 returned a Tensor; newer releases return a
            # BaseModelOutputWithPooling.  In both cases this is the projected
            # CLIP-space pooled feature, not an unprojected token state.
            if isinstance(output, torch.Tensor):
                features = output
            else:
                features = getattr(output, 'pooler_output', None)
                if not isinstance(features, torch.Tensor):
                    features = getattr(output, 'text_embeds', None)
                if not isinstance(features, torch.Tensor):
                    raise TypeError(
                        'cannot extract pooled CLIP text features from '
                        f'{type(output).__name__}')
            features = features.float()
            pieces.append((F.normalize(features, dim=1) if normalize else features).cpu())
    return torch.cat(pieces, dim=0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('umrch', 'duheg-selected'))
    parser.add_argument('--cache_dir', required=True)
    parser.add_argument('--terms_file', required=True)
    parser.add_argument('--out_prefix', required=True)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--allow_download', action='store_true')
    args = parser.parse_args()

    meta_path = os.path.join(args.cache_dir, 'meta.json')
    with open(meta_path) as handle:
        cache_meta = json.load(handle)
    checkpoint = str(cache_meta.get('backbone', ''))
    if 'clip' not in checkpoint.lower():
        raise ValueError(
            f'{args.mode} requires a CLIP cache, but meta backbone is {checkpoint!r}')

    if not args.allow_download:
        # Set before importing transformers/huggingface_hub.  Recent versions
        # otherwise start a background safetensors-conversion network request
        # even when from_pretrained(local_files_only=True) succeeds locally.
        os.environ.setdefault('HF_HUB_OFFLINE', '1')
        os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    import transformers
    from transformers import AutoTokenizer, CLIPModel
    from transformers.utils import cached_file
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, local_files_only=not args.allow_download)
    model = CLIPModel.from_pretrained(
        checkpoint, local_files_only=not args.allow_download,
    ).to(args.device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    def resolved_file(name: str) -> str | None:
        try:
            path = cached_file(
                checkpoint, name, local_files_only=not args.allow_download)
        except (OSError, ValueError):
            return None
        return None if path is None else os.path.abspath(path)

    model_weight_path = resolved_file('model.safetensors')
    if model_weight_path is None:
        model_weight_path = resolved_file('pytorch_model.bin')
    if model_weight_path is None:
        raise FileNotFoundError(
            f'cannot resolve a local weight artifact for {checkpoint!r}')
    tokenizer_paths = {}
    for name in ('tokenizer.json', 'tokenizer_config.json', 'vocab.json',
                 'merges.txt', 'special_tokens_map.json'):
        path = resolved_file(name)
        if path is not None:
            tokenizer_paths[path] = _sha256_file(path)
    if not tokenizer_paths:
        raise FileNotFoundError('cannot resolve any tokenizer provenance files')
    model_revision = getattr(model.config, '_commit_hash', None)
    if not isinstance(model_revision, str) or not model_revision:
        raise ValueError(
            'the resolved CLIP model does not expose an immutable revision hash')
    hf_provenance = {
        'checkpoint': checkpoint,
        'model_revision': model_revision,
        'transformers_version': str(transformers.__version__),
        'model_weight_file': model_weight_path,
        'model_weight_sha256': _sha256_file(model_weight_path),
        'tokenizer_files_sha256': tokenizer_paths,
    }

    terms = read_ordered_terms(args.terms_file)
    if args.mode == 'umrch':
        prompts = [f'a photo of the {term}' for term in terms]
        embeddings = _encode(
            model, tokenizer, prompts, args.device, args.batch_size,
            normalize=True)
    else:
        prompt_features = []
        for template in DUHEG_PROMPT_TEMPLATES:
            prompts = [template.format(term) for term in terms]
            prompt_features.append(_encode(
                model, tokenizer, prompts, args.device, args.batch_size,
                normalize=False))
        # The public preparation averages raw CLIP features first and only
        # then normalizes the seven-template ensemble.
        embeddings = F.normalize(torch.stack(prompt_features).mean(dim=0), dim=1)

    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    embedding_path = str(prefix) + '_embeddings.npy'
    np.save(embedding_path, embeddings.numpy().astype(np.float32))

    outputs = {'embeddings': embedding_path}
    if args.mode == 'umrch':
        post_ln = model.vision_model.post_layernorm
        # HF visual_projection is Linear(H_v,D); cached tokens are the
        # pre-LN last_hidden_state, hence OpenAI CLIP's ln_post + @proj path.
        projection = model.visual_projection.weight.detach().float().T.cpu().numpy()
        adapter_path = str(prefix) + '_vision_adapter.npz'
        np.savez(
            adapter_path,
            ln_weight=post_ln.weight.detach().float().cpu().numpy(),
            ln_bias=post_ln.bias.detach().float().cpu().numpy(),
            projection=projection,
        )
        outputs['vision_adapter'] = adapter_path

    terms_file_sha256 = _sha256_file(args.terms_file)
    information_condition = classify_term_information_condition(
        args.mode, terms_file_sha256, terms)
    manifest = {
        'asset_contract_version': ASSET_CONTRACT_VERSION,
        'mode': args.mode,
        'cache_meta': os.path.abspath(meta_path),
        'cache_meta_sha256': _sha256_file(meta_path),
        'cache_backbone': checkpoint,
        'hf_provenance': hf_provenance,
        'terms_file': os.path.abspath(args.terms_file),
        'terms_file_sha256': terms_file_sha256,
        'ordered_terms': terms,
        'ordered_terms_sha256': sha256_ordered_terms(terms),
        'n_terms': len(terms),
        'embedding_dim': int(embeddings.shape[1]),
        'prompt': ('a photo of the {class}' if args.mode == 'umrch'
                   else list(DUHEG_PROMPT_TEMPLATES)),
        'term_source': ('audited UMRCH release taxonomy'
                        if args.mode == 'umrch'
                        else 'user-supplied list; WordNet selection unverified'),
        'duheg_selection_included': False if args.mode == 'duheg-selected' else None,
        'outputs': {key: os.path.abspath(value) for key, value in outputs.items()},
        'outputs_sha256': {key: _sha256_file(value)
                           for key, value in outputs.items()},
    }
    manifest.update(information_condition)
    manifest_path = str(prefix) + '_manifest.json'
    with open(manifest_path, 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
