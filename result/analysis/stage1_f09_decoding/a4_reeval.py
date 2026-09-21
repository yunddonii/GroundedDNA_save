"""Stage 1a: re-score the F09 shared-codebook ablation under a deployment rule that keeps slots apart.

With `--share_codebook`, codebook-mean deployment anchors are identical for every local slot, so all
four local slots pick the same codeword (verified on the F09 extractions). Both arms are therefore
re-encoded here under ONE common rule with slot-specific anchors -- `eval_routing_mode =
text_prototype`, the per-slot EMA of training caption tokens that every caption-trained checkpoint
already stores -- and decoded with the paper's s4.7 probe settings on the approved rows.
Nothing is trained or written into the runs.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
TAGS = {"flickr25k": "flickr_A_v4", "nuswide": "nuswide_A_v4", "mscoco": "mscoco_A_v5b"}


def encode(run_dir, device):
    sys.argv = [sys.argv[0], "--config_path", run_dir]
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
    args.device = device
    model = SigLIP2SemanticOTModel(args).to(device)
    ckpt = _find_model_checkpoint(args.save_model_state_path)
    load_model_state_dict_for_extraction(model, ckpt, map_location=device)
    apply_inference_epoch(model, ckpt, args)
    assert bool(model._text_prototype_initialized.item()), "no stored text prototypes"
    model.eval_routing_mode = "text_prototype"
    t = get_transform("test")
    tr, te, _ = load_dataset(args.dataset_dir, args.dataset, setting=args.setting, train_transform=t,
                             test_transform=t, load_train=True, load_database=False, load_test=True,
                             return_index=True, qwen_text_cache_path=getattr(args, "qwen_text_cache_path", None),
                             siglip2_feature_cache_dir=getattr(args, "siglip2_feature_cache_dir", None))
    mk = lambda ds: torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False, num_workers=args.num_workers)  # noqa: E731
    return encode_split(model, mk(tr), device, split_name="train"), encode_split(model, mk(te), device, split_name="query")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(TAGS))
    ap.add_argument("--arm", required=True, choices=["A5_both", "A4_shared_codebook"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run = sorted(glob.glob(os.path.join(REPO, f"result/*_setting1_promptAblA_{TAGS[a.dataset]}_{a.arm}_P0refit_*")))
    assert len(run) == 1, run
    train, query = encode(run[0], a.device)
    sys.argv = [sys.argv[0]]
    import scripts.interp_control_a_partitions as drv
    H, _ = drv.load_probe()
    project = drv.make_projector()
    record, aggregate, _ = drv.admitted_record(f"{a.dataset}_seed42")
    settings, policy = aggregate["decoder_settings"], record["input_admission"]["applied_analysis_policy"]
    H._set_n_slots(5)
    tr_u = H.codon_ids(project(train["base_indices"], policy), 3)
    te_u = H.codon_ids(project(query["base_indices"], policy), 3)
    lab = lambda z: np.asarray(z["multi_hot_labels"], np.int64)  # noqa: E731
    res = H.decode_all_slots(tr_u, lab(train), te_u, lab(query), 64, float(settings["alpha"]),
                             int(settings["min_support"]))
    cw = np.asarray(query["codebook_indices"])
    out = {"dataset": a.dataset, "arm": a.arm, "run_dir": run[0], "routing": "text_prototype",
           "concept_mAP": res["slot_mean"]["concept_mAP"],
           "per_slot": [s.get("concept_mAP") for s in res["per_slot"]],
           "local_codewords_all_equal": float(np.mean((cw[:, 1] == cw[:, 2]) & (cw[:, 2] == cw[:, 3]) & (cw[:, 3] == cw[:, 4]))),
           "distinct_codewords_per_slot": [int(len(np.unique(cw[:, m]))) for m in range(5)],
           "unique_full_codes_query": float(len(np.unique(np.asarray(query["base_indices"]), axis=0)) / len(cw)),
           "n_train": int(len(tr_u)), "n_query": int(len(te_u)), "_ap": res["_ap_per_sample"].tolist()}
    json.dump(out, open(a.out, "w"))
    print(json.dumps({k: v for k, v in out.items() if k not in ("_ap", "run_dir")}))


if __name__ == "__main__":
    main()
