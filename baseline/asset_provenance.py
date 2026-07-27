"""Strict provenance checks for external semantic baseline assets."""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any

import numpy as np


ASSET_CONTRACT_VERSION = 2

# Byte-identical class-name files in the audited UMRCH release at commit
# 5e60e8d3d35e435506f86dd135e90763c07b5592.  These hashes are also useful
# when auditing a purported DUH-EG noun bank: feeding one of these target
# benchmark taxonomies is a U2 condition, not generic U1 external knowledge.
OFFICIAL_UMRCH_TAXONOMY_SHA256 = {
    'Flickr25k': '5f05ab599b09add2bcc8c324fe51fb244c169a4226a75ea9bdd509a62209e4e2',
    'MSCOCO': 'bd17f1ee35d5f3c862a4894605855abbb9dda4b0621fdb0ac4c2c8c7bb7e730a',
    'NUSWIDE': '09a56a19f992f5ef23cbfb6711e13c2d549f268b1b458772adccd71b92336242',
}
OFFICIAL_UMRCH_TAXONOMIES = {
    'Flickr25k': (
        'animals', 'baby', 'bird', 'car', 'clouds', 'dog', 'female', 'flower',
        'food', 'indoor', 'lake', 'male', 'night', 'people', 'plant_life',
        'portrait', 'river', 'sea', 'sky', 'structures', 'sunset', 'transport',
        'tree', 'water',
    ),
    'MSCOCO': (
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
        'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
        'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
        'teddy bear', 'hair drier', 'toothbrush',
    ),
    'NUSWIDE': (
        'animal', 'beach', 'building', 'clouds', 'flowers', 'grass', 'lake',
        'mountain', 'ocean', 'person', 'plants', 'reflection', 'road', 'rocks',
        'sky', 'snow', 'sunset', 'tree', 'vehicle', 'water', 'window',
    ),
}

# The seven-template ensemble used by the DUH-EG public implementation.
# Keeping the audited value in the validation module prevents the trainer
# from importing an executable preparation script merely to check metadata.
DUHEG_PROMPT_TEMPLATES = (
    'itap of a {}.',
    'a bad photo of the {}.',
    'a origami {}.',
    'a photo of the large {}.',
    'a {} in a video game.',
    'art of the {}.',
    'a photo of the small {}.',
)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_ordered_terms(path: str) -> list[str]:
    """Parse a term source exactly as the asset preparation command does."""
    terms: list[str] = []
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            term = line.strip().split(',')[0].strip()
            if term and term.lower() not in ('name', 'noun'):
                terms.append(term)
    if not terms:
        raise ValueError(f'no non-empty terms in {path}')
    if len(set(terms)) != len(terms):
        raise ValueError('terms_file contains duplicate parsed terms')
    return terms


def sha256_ordered_terms(terms: list[str]) -> str:
    """Hash the semantic sequence, independently of source-file formatting."""
    payload = json.dumps(
        terms, ensure_ascii=False, separators=(',', ':'),
    ).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def classify_term_information_condition(
        mode: str, terms_file_sha256: str,
        ordered_terms: list[str] | tuple[str, ...]) -> dict[str, str]:
    """Classify the auditable U1/U2 condition from the actual term source."""
    taxonomy_dataset = next(
        (dataset for dataset, digest in OFFICIAL_UMRCH_TAXONOMY_SHA256.items()
         if digest == terms_file_sha256),
        None,
    )
    semantic_taxonomy_dataset = None
    normalized_terms = {
        ' '.join(term.casefold().replace('_', ' ').replace('-', ' ').split())
        for term in ordered_terms
    }
    for dataset, official_terms in OFFICIAL_UMRCH_TAXONOMIES.items():
        normalized_official = {
            ' '.join(term.casefold().replace('_', ' ').replace('-', ' ').split())
            for term in official_terms
        }
        if normalized_terms == normalized_official:
            semantic_taxonomy_dataset = dataset
            break
    matched_dataset = taxonomy_dataset or semantic_taxonomy_dataset
    if matched_dataset is not None:
        return {
            'information_tier': 'U2',
            'term_source_condition': (
                'target-benchmark-taxonomy-byte-exact'
                if taxonomy_dataset is not None
                else 'target-benchmark-taxonomy-semantic-set-match'),
            'matched_official_taxonomy_dataset': matched_dataset,
        }
    if mode == 'duheg-selected':
        return {
            'information_tier': 'U?',
            'term_source_condition': (
                'external-source-and-term-selection-unverified'),
            'matched_official_taxonomy_dataset': '',
        }
    return {
        'information_tier': 'U?',
        'term_source_condition': 'benchmark-taxonomy-source-unverified',
        'matched_official_taxonomy_dataset': '',
    }


def load_and_verify_text_asset_manifest(
        manifest_path: str, *, expected_mode: str,
        expected_embeddings: str, expected_prompt: str | list[str],
        expected_extra_outputs: dict[str, str] | None = None) -> dict[str, Any]:
    """Verify paths, hashes, mode, prompt, and ordered-term bookkeeping."""
    with open(manifest_path, encoding='utf-8') as handle:
        manifest = json.load(handle)
    if manifest.get('asset_contract_version') != ASSET_CONTRACT_VERSION:
        raise ValueError(
            f'{manifest_path} has unsupported asset contract '
            f'{manifest.get("asset_contract_version")!r}')
    if manifest.get('mode') != expected_mode:
        raise ValueError(
            f'asset mode {manifest.get("mode")!r} != {expected_mode!r}')
    if manifest.get('prompt') != expected_prompt:
        raise ValueError('asset prompt/template set does not match the audited method')

    outputs = manifest.get('outputs')
    output_hashes = manifest.get('outputs_sha256')
    if not isinstance(outputs, dict) or not isinstance(output_hashes, dict):
        raise ValueError('asset manifest lacks outputs/outputs_sha256 mappings')
    expected = {'embeddings': expected_embeddings}
    expected.update(expected_extra_outputs or {})
    for name, path in expected.items():
        manifest_path_value = outputs.get(name)
        if not isinstance(manifest_path_value, str):
            raise ValueError(f'asset manifest lacks output {name!r}')
        if os.path.realpath(manifest_path_value) != os.path.realpath(path):
            raise ValueError(
                f'manifest output {name} points to {manifest_path_value}, not {path}')
        actual_hash = sha256_file(path)
        if output_hashes.get(name) != actual_hash:
            raise ValueError(f'{name} SHA-256 does not match its manifest')

    terms = manifest.get('ordered_terms')
    if not isinstance(terms, list) or not all(isinstance(term, str) for term in terms):
        raise ValueError('asset manifest must retain the ordered term list')
    if len(terms) != int(manifest.get('n_terms', -1)):
        raise ValueError('ordered term count does not match n_terms')
    embeddings = np.load(expected_embeddings, mmap_mode='r')
    if embeddings.ndim != 2:
        raise ValueError(f'text embeddings must be rank 2, got {embeddings.shape}')
    if embeddings.shape[0] != len(terms):
        raise ValueError(
            'embedding row count does not match the ordered term list')
    if embeddings.shape[1] != int(manifest.get('embedding_dim', -1)):
        raise ValueError('embedding width does not match manifest embedding_dim')
    terms_path = manifest.get('terms_file')
    terms_hash = manifest.get('terms_file_sha256')
    if not isinstance(terms_path, str) or not terms_path:
        raise ValueError('asset manifest lacks a source terms_file')
    if not os.path.isfile(terms_path):
        raise ValueError(f'source terms file is missing: {terms_path}')
    actual_terms_file_hash = sha256_file(terms_path)
    if not isinstance(terms_hash, str) or actual_terms_file_hash != terms_hash:
        raise ValueError('source terms file changed after asset preparation')
    parsed_terms = read_ordered_terms(terms_path)
    if parsed_terms != terms:
        raise ValueError(
            'ordered_terms does not equal the parsed source terms in order')
    ordered_terms_hash = sha256_ordered_terms(terms)
    if manifest.get('ordered_terms_sha256') != ordered_terms_hash:
        raise ValueError('ordered term sequence SHA-256 does not match manifest')
    expected_condition = classify_term_information_condition(
        expected_mode, actual_terms_file_hash, terms)
    for key, expected_value in expected_condition.items():
        if manifest.get(key) != expected_value:
            raise ValueError(
                f'asset manifest {key} does not match audited term source: '
                f'{manifest.get(key)!r} != {expected_value!r}')

    hf = manifest.get('hf_provenance')
    if not isinstance(hf, dict):
        raise ValueError('asset manifest lacks Hugging Face revision provenance')
    for key in ('checkpoint', 'model_revision', 'transformers_version',
                'model_weight_file', 'model_weight_sha256'):
        if not isinstance(hf.get(key), str) or not hf[key]:
            raise ValueError(f'asset manifest lacks hf_provenance.{key}')
    revision = hf['model_revision']
    if re.fullmatch(r'[0-9a-f]{40}', revision) is None:
        raise ValueError(
            'hf_provenance.model_revision must be an immutable 40-hex commit')
    tokenizer_hashes = hf.get('tokenizer_files_sha256')
    if not isinstance(tokenizer_hashes, dict) or not tokenizer_hashes:
        raise ValueError('asset manifest lacks tokenizer file hashes')
    for path, expected_hash in tokenizer_hashes.items():
        if not isinstance(path, str) or not isinstance(expected_hash, str):
            raise ValueError('invalid tokenizer provenance entry')
        if not os.path.isfile(path):
            raise ValueError(f'cached tokenizer provenance file is missing: {path}')
        if sha256_file(path) != expected_hash:
            raise ValueError('cached tokenizer file changed after asset preparation')
    model_weight_file = hf['model_weight_file']
    if not os.path.isfile(model_weight_file):
        raise ValueError(
            f'cached CLIP weight provenance file is missing: {model_weight_file}')
    normalized_parts = os.path.normpath(model_weight_file).split(os.sep)
    if 'snapshots' in normalized_parts and revision not in normalized_parts:
        raise ValueError(
            'cached CLIP weight path does not match model_revision')
    if sha256_file(model_weight_file) != hf['model_weight_sha256']:
        raise ValueError('cached CLIP weight file changed after asset preparation')
    return manifest


def verify_manifest_cache(manifest: dict[str, Any], cache_dir: str) -> dict:
    """Reject same-dimensional assets produced by a different CLIP cache."""
    meta_path = os.path.join(cache_dir, 'meta.json')
    with open(meta_path, encoding='utf-8') as handle:
        cache_meta = json.load(handle)
    if manifest.get('cache_backbone') != cache_meta.get('backbone'):
        raise ValueError(
            'semantic asset/cache backbone mismatch: '
            f'{manifest.get("cache_backbone")!r} vs {cache_meta.get("backbone")!r}')
    if manifest['hf_provenance'].get('checkpoint') != cache_meta.get('backbone'):
        raise ValueError('semantic asset checkpoint differs from cache backbone')
    recorded_hash = manifest.get('cache_meta_sha256')
    if not isinstance(recorded_hash, str) or recorded_hash != sha256_file(meta_path):
        raise ValueError(
            'cache meta SHA-256 differs from the semantic asset provenance')
    return cache_meta
