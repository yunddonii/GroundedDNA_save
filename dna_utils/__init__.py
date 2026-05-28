"""Utility package for the DNA hashing framework.

Bundles small reusable helpers that the SigLIP2 + DNA hashing pipeline shares
across training, extraction, and evaluation:

    csv_logger                  per-epoch CSV append logger
    dna_code_utils              base index ↔ 2-bit ↔ one-hot + Hamming distances
    text_description_processor  SigLIP2 codebook-text tokenizer helpers
    vlm_qwen25_descriptions     Qwen2.5-VL offline description utilities

The package is named ``dna_utils`` (not ``utils``) to avoid clashing with the
project's pre-existing ``utils.py`` module that the legacy training scripts
import from.
"""

from .csv_logger import EpochCSVLogger

from .dna_code_utils import (
    BASE_TO_BITS,
    base_indices_to_2bit,
    base_indices_to_2bit_flat,
    two_bit_flat_to_base_indices,
    onehot_to_base_indices,
    base_indices_to_onehot,
    base_hamming_distance,
    bit_hamming_distance_2bit,
)

from .text_description_processor import (
    CODEBOOK_TEXT_KEYS,
    DEFAULT_FALLBACK_TEXT,
    DEFAULT_SIGLIP2_TOKENIZER_NAME,
    DEFAULT_CLIP_TOKENIZER_NAME,
    extract_codebook_texts,
    batch_extract_codebook_texts,
    build_siglip2_text_tokenizer,
    build_clip_text_tokenizer,
    tokenize_codebook_texts,
)

from .vlm_qwen25_descriptions import (
    DEFAULT_VLM,
    build_qwen25_vl_generator,
    generate_object_centric_scene_graph,
)

from .training_utils import (
    get_transform,
    print_one_epoch_info,
    get_summarywriter,
)

from .visualization import (
    visualize_routing,
    visualize_codebook_tsne,
)

from .bio_constraints import (
    DEFAULT_GC_MIN_FRAC,
    DEFAULT_GC_MAX_FRAC,
    DEFAULT_MAX_HOMOPOLYMER_RUN,
    is_valid,
    is_valid_batch,
    violation_report,
    project_to_valid,
    batch_project_to_valid,
)


__all__ = [
    # csv_logger
    "EpochCSVLogger",
    # dna_code_utils
    "BASE_TO_BITS",
    "base_indices_to_2bit",
    "base_indices_to_2bit_flat",
    "two_bit_flat_to_base_indices",
    "onehot_to_base_indices",
    "base_indices_to_onehot",
    "base_hamming_distance",
    "bit_hamming_distance_2bit",
    # text_description_processor
    "CODEBOOK_TEXT_KEYS",
    "DEFAULT_FALLBACK_TEXT",
    "DEFAULT_SIGLIP2_TOKENIZER_NAME",
    "DEFAULT_CLIP_TOKENIZER_NAME",
    "extract_codebook_texts",
    "batch_extract_codebook_texts",
    "build_siglip2_text_tokenizer",
    "build_clip_text_tokenizer",
    "tokenize_codebook_texts",
    # vlm_qwen25_descriptions
    "DEFAULT_VLM",
    "build_qwen25_vl_generator",
    "generate_object_centric_scene_graph",
    # training_utils
    "get_transform",
    "print_one_epoch_info",
    "get_summarywriter",
    # visualization
    "visualize_routing",
    "visualize_codebook_tsne",
    # bio_constraints
    "DEFAULT_GC_MIN_FRAC",
    "DEFAULT_GC_MAX_FRAC",
    "DEFAULT_MAX_HOMOPOLYMER_RUN",
    "is_valid",
    "is_valid_batch",
    "violation_report",
    "project_to_valid",
    "batch_project_to_valid",
]
