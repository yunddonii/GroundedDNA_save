"""Encode the MS-COCO TRAIN split of one finished F09 run, in memory, without writing into the run.

`scripts/extract_train_split.py` does the same encoding but adds files to the run directory and
rewrites its completion marker; the F09 runs are sealed campaign outputs, so this reproduces its
exact steps (restored args, extraction state-dict loader, inference epoch, test transform,
`encode_split`) and writes only to --out. The encoder is validated first: the same code path
applied to the first --check rows of the DATABASE must reproduce the run's own extract_db.npz
row for row (identities and base indices), or nothing is written.

    python mscoco_train_codes.py --config_path <F09 run dir> --out <npz> [--check 2048]
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)


def main():
    own = argparse.ArgumentParser(add_help=False)
    own.add_argument("--out", required=True)
    own.add_argument("--check", type=int, default=2048)
    mine, rest = own.parse_known_args()
    sys.argv = [sys.argv[0], *rest]                     # the rest is Config's (--config_path ...)

    from config import Config, set_random_seed
    from dataloaders import load_dataset
    from dna_utils import get_transform
    from extraction_siglip2 import encode_split, _find_model_checkpoint, _resume_args_flat_or_legacy
    from model_siglip2 import SigLIP2SemanticOTModel
    from dna_utils.run_identity import load_model_state_dict_for_extraction
    from dna_utils.runtime_state import apply_inference_epoch

    set_random_seed(42)
    args = Config()
    _resume_args_flat_or_legacy(args)
    model = SigLIP2SemanticOTModel(args).to(args.device)
    ckpt = _find_model_checkpoint(args.save_model_state_path)
    load_model_state_dict_for_extraction(model, ckpt, map_location=args.device)
    resolved = apply_inference_epoch(model, ckpt, args)
    transform = get_transform("test")
    common = dict(setting=args.setting, train_transform=transform, test_transform=transform,
                  return_index=True, qwen_text_cache_path=getattr(args, "qwen_text_cache_path", None),
                  siglip2_feature_cache_dir=getattr(args, "siglip2_feature_cache_dir", None))
    batch = int(getattr(args, "extract_batch_size", 256))

    # 1. validate the encoder against the run's own DB extraction
    _, _, database = load_dataset(args.dataset_dir, args.dataset, load_train=False,
                                  load_database=True, load_test=False, **common)
    subset = torch.utils.data.Subset(database, list(range(min(mine.check, len(database)))))
    check = encode_split(model, torch.utils.data.DataLoader(subset, batch_size=batch, shuffle=False,
                                                            num_workers=args.num_workers),
                         args.device, split_name="db-check")
    db = np.load(os.path.join(args.save_result_path, "extract_db.npz"), allow_pickle=True)
    n = len(check["base_indices"])
    same_ids = [os.path.basename(str(p)) for p in check["image_paths"]] == \
               [os.path.basename(str(p)) for p in db["image_paths"][:n]]
    same_codes = np.array_equal(check["base_indices"], db["base_indices"][:n])
    if not (same_ids and same_codes):
        raise SystemExit(f"encoder does not reproduce extract_db.npz: ids {same_ids}, codes {same_codes}")

    # 2. the train split
    trainset, _, _ = load_dataset(args.dataset_dir, args.dataset, load_train=True,
                                  load_database=False, load_test=False, **common)
    out = encode_split(model, torch.utils.data.DataLoader(trainset, batch_size=batch, shuffle=False,
                                                          num_workers=args.num_workers),
                       args.device, split_name="train")
    np.savez(mine.out, **out)
    with open(mine.out, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    ckpt_sha = hashlib.sha256(open(ckpt, "rb").read()).hexdigest()
    json.dump({"run_dir": args.save_result_path, "checkpoint": ckpt, "checkpoint_sha256": ckpt_sha,
               "inference_epoch": int(resolved.epoch), "rows": int(len(out["base_indices"])),
               "npz_sha256": digest, "db_check_rows": n, "db_check": "identical ids and base indices"},
              open(mine.out + ".json", "w"), indent=1)
    print(f"[train-codes] validated on {n} DB rows; {len(out['base_indices'])} train rows -> {mine.out}")


if __name__ == "__main__":
    main()
