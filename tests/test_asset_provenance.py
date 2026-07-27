import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from baseline.asset_provenance import (
    ASSET_CONTRACT_VERSION,
    classify_term_information_condition,
    load_and_verify_text_asset_manifest,
    sha256_file,
    sha256_ordered_terms,
    verify_manifest_cache,
)


class AssetProvenanceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.cache = self.root / 'cache'
        self.cache.mkdir()
        self.meta = self.cache / 'meta.json'
        self.meta.write_text(
            json.dumps({'backbone': 'openai/clip-vit-base-patch16'}),
            encoding='utf-8',
        )
        self.terms = self.root / 'terms.txt'
        self.terms.write_text('cat\ndog\n', encoding='utf-8')
        self.embeddings = self.root / 'embeddings.npy'
        np.save(self.embeddings, np.zeros((2, 3), dtype=np.float32))
        self.weights = self.root / 'model.safetensors'
        self.weights.write_bytes(b'frozen-model-weights')
        self.tokenizer = self.root / 'tokenizer.json'
        self.tokenizer.write_text('{"version":"unit"}', encoding='utf-8')
        self.manifest_path = self.root / 'manifest.json'
        self.manifest = {
            'asset_contract_version': ASSET_CONTRACT_VERSION,
            'mode': 'unit-test',
            'cache_backbone': 'openai/clip-vit-base-patch16',
            'hf_provenance': {
                'checkpoint': 'openai/clip-vit-base-patch16',
                'model_revision': 'a' * 40,
                'transformers_version': 'unit-test',
                'model_weight_file': str(self.weights),
                'model_weight_sha256': sha256_file(str(self.weights)),
                'tokenizer_files_sha256': {
                    str(self.tokenizer): sha256_file(str(self.tokenizer)),
                },
            },
            'cache_meta_sha256': sha256_file(str(self.meta)),
            'terms_file': str(self.terms),
            'terms_file_sha256': sha256_file(str(self.terms)),
            'ordered_terms': ['cat', 'dog'],
            'ordered_terms_sha256': sha256_ordered_terms(['cat', 'dog']),
            'n_terms': 2,
            'embedding_dim': 3,
            'prompt': 'a photo of the {class}',
            'outputs': {'embeddings': str(self.embeddings)},
            'outputs_sha256': {
                'embeddings': sha256_file(str(self.embeddings)),
            },
        }
        self.manifest.update(classify_term_information_condition(
            'unit-test', sha256_file(str(self.terms)), ['cat', 'dog']))
        self._write_manifest()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_manifest(self) -> None:
        self.manifest_path.write_text(
            json.dumps(self.manifest), encoding='utf-8')

    def _verify(self):
        return load_and_verify_text_asset_manifest(
            str(self.manifest_path), expected_mode='unit-test',
            expected_embeddings=str(self.embeddings),
            expected_prompt='a photo of the {class}',
        )

    def test_valid_manifest_binds_output_terms_and_cache(self) -> None:
        manifest = self._verify()
        cache_meta = verify_manifest_cache(manifest, str(self.cache))
        self.assertEqual(cache_meta['backbone'], self.manifest['cache_backbone'])

    def test_mutated_output_is_rejected(self) -> None:
        self.embeddings.write_bytes(b'mutated')
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            self._verify()

    def test_wrong_prompt_and_reordered_terms_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, 'prompt'):
            load_and_verify_text_asset_manifest(
                str(self.manifest_path), expected_mode='unit-test',
                expected_embeddings=str(self.embeddings),
                expected_prompt='wrong prompt',
            )
        self.manifest['ordered_terms'] = ['dog']
        self._write_manifest()
        with self.assertRaisesRegex(ValueError, 'term count'):
            self._verify()

    def test_missing_or_semantically_different_term_source_is_rejected(self) -> None:
        self.terms.unlink()
        with self.assertRaisesRegex(ValueError, 'terms file is missing'):
            self._verify()

        self.terms.write_text('dog\ncat\n', encoding='utf-8')
        self.manifest['terms_file_sha256'] = sha256_file(str(self.terms))
        self.manifest.update(classify_term_information_condition(
            'unit-test', self.manifest['terms_file_sha256'], ['cat', 'dog']))
        self._write_manifest()
        with self.assertRaisesRegex(ValueError, 'parsed source terms'):
            self._verify()

    def test_term_condition_is_derived_from_source_hash(self) -> None:
        self.manifest['information_tier'] = 'U1'
        self._write_manifest()
        with self.assertRaisesRegex(ValueError, 'information_tier'):
            self._verify()

    def test_reformatted_reordered_target_taxonomy_is_still_u2(self) -> None:
        condition = classify_term_information_condition(
            'duheg-selected', '0' * 64,
            ['water', 'tree', 'transport', 'sunset', 'structures', 'sky',
             'sea', 'river', 'portrait', 'plant life', 'people', 'night',
             'male', 'lake', 'indoor', 'food', 'flower', 'female', 'dog',
             'clouds', 'car', 'bird', 'baby', 'animals'])
        self.assertEqual(condition['information_tier'], 'U2')
        self.assertEqual(
            condition['matched_official_taxonomy_dataset'], 'Flickr25k')

    def test_missing_hf_provenance_files_are_rejected(self) -> None:
        self.tokenizer.unlink()
        with self.assertRaisesRegex(ValueError, 'tokenizer.*missing'):
            self._verify()

    def test_cache_metadata_change_is_rejected_even_at_same_dimension(self) -> None:
        manifest = self._verify()
        self.meta.write_text(
            json.dumps({'backbone': 'openai/clip-vit-base-patch16', 'D_proj': 512}),
            encoding='utf-8',
        )
        with self.assertRaisesRegex(ValueError, 'meta SHA-256'):
            verify_manifest_cache(manifest, str(self.cache))

    def test_mutable_or_symbolic_model_revision_is_rejected(self) -> None:
        self.manifest['hf_provenance']['model_revision'] = 'main'
        self._write_manifest()
        with self.assertRaisesRegex(ValueError, 'immutable 40-hex commit'):
            self._verify()


if __name__ == '__main__':
    unittest.main()
