"""Regenerate `viz_routing_heatmap.png` for a completed training result dir.

Why: older runs (num_tokens=196, single 14×14 grid) saved the heatmap by
default; recent localL8K3 / grid3K3 runs (num_tokens=K×196) crashed
visualize_routing because the token count is not a single square. The fix in
`dna_utils/visualization.py` (per-crop K-row layout) handles K>1; this script
applies the fix retroactively to already-trained result dirs.

Usage:
    python scripts/regen_viz_routing.py \\
        --result_dir result/260622+cub_200_setting1_cub200_v160b_v6b_K64_localL8K3_stackedText_partialWhiten_gamma0.25+bs+64+e+60+proj_lr+0.001 \\
        [--num_samples 12]

Reads `args.txt` for hyperparameters, builds the model, loads
`model_state_dict.pth`, builds the dataset, and runs `visualize_routing`.
Saves to `<result_dir>/viz_routing_heatmap.png` (overwriting any existing).
"""
from __future__ import annotations
import argparse
import os
import sys
from types import SimpleNamespace

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from model_siglip2 import SigLIP2SemanticOTModel
from dna_utils import visualize_routing


def _parse_args_txt(path: str) -> SimpleNamespace:
    """Parse the dashed key/value file train_siglip2 writes at run start.

    Each line is `<key><dashes><value>`. Values get coerced to int / float /
    bool / None / list-of-str where the obvious shape matches.
    """
    ns = SimpleNamespace()
    with open(path) as f:
        for raw in f:
            line = raw.rstrip("\n")
            # split on the FIRST run of >=3 dashes
            for k_end in range(len(line)):
                if line[k_end] == "-":
                    break
            else:
                continue
            v_start = k_end
            while v_start < len(line) and line[v_start] == "-":
                v_start += 1
            key = line[:k_end]
            val = line[v_start:].strip()
            if not key:
                continue
            # coerce
            if val == "None":
                val = None
            elif val == "True":
                val = True
            elif val == "False":
                val = False
            else:
                try:
                    val = int(val)
                except ValueError:
                    try:
                        val = float(val)
                    except ValueError:
                        # leave as string; nested list like ['x'] handled below
                        if val.startswith("[") and val.endswith("]"):
                            inner = val[1:-1].strip()
                            if inner.startswith("'") and inner.endswith("'"):
                                val = [inner[1:-1]]
            setattr(ns, key, val)
    return ns


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", required=True,
                    help="Path to result/<date>+<setting>+<tag>+... directory "
                         "containing args.txt + model_state_dict.pth.")
    ap.add_argument("--num_samples", type=int, default=12,
                    help="Number of trainset images to visualize.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cache_dir_override", default=None,
                    help="Override siglip2_feature_cache_dir from args.txt "
                         "(useful for whole-image viz on a FAIRrank-trained ckpt).")
    ap.add_argument("--tsne", action="store_true",
                    help="Also regenerate viz_codebook_tsne.png (test split).")
    ap.add_argument("--tsne_only", action="store_true",
                    help="Regenerate ONLY the t-SNE, skip the routing heatmap.")
    ap.add_argument("--tsne_samples", type=int, default=2000)
    ap.add_argument("--tsne_save_name", default="viz_codebook_tsne.png")
    ap.add_argument("--epoch", type=int, default=None,
                    help="Epoch to place the model at before drawing. This is "
                         "NOT cosmetic: `_current_epoch` defaults to 0 and is "
                         "not stored in the state dict, so a freshly loaded "
                         "checkpoint runs the Sinkhorn epsilon at its INITIAL "
                         "value. On Flickr25k that is eps=1.0 instead of the "
                         "annealed 0.1, which raises the slot-to-slot routing "
                         "cosine from 0.77 to 1.00 and makes every per-slot "
                         "heatmap look identical. Pass the run's total epochs "
                         "to draw what the trained model actually does.")
    ap.add_argument("--save_name", default="viz_routing_heatmap.png",
                    help="Output filename inside result_dir.")
    cli = ap.parse_args()

    args_path = os.path.join(cli.result_dir, "args.txt")
    ckpt_path = os.path.join(cli.result_dir, "model_state_dict.pth")
    if not os.path.isfile(args_path):
        print(f"ERROR: {args_path} not found.")
        return 1
    if not os.path.isfile(ckpt_path):
        print(f"ERROR: {ckpt_path} not found.")
        return 1

    args = _parse_args_txt(args_path)
    args.device = cli.device
    # Optional: override cache (e.g. whole-image cache for a FAIRrank-trained ckpt).
    if cli.cache_dir_override:
        old_cache = getattr(args, 'siglip2_feature_cache_dir', '?')
        args.siglip2_feature_cache_dir = cli.cache_dir_override
        # Also override text_whiten_npz to match the new cache
        _maybe_whiten = os.path.join(cli.cache_dir_override, "text_whiten.npz")
        if os.path.exists(_maybe_whiten):
            args.text_whiten_npz = _maybe_whiten
        print(f"[regen] cache OVERRIDE: {old_cache} -> {cli.cache_dir_override}")
    print(f"[regen] parsed {sum(1 for _ in vars(args))} args from {args_path}")
    print(f"[regen] dataset={getattr(args, 'dataset', '?')}  "
          f"cache={getattr(args, 'siglip2_feature_cache_dir', '?')}")

    # build model + load state
    model = SigLIP2SemanticOTModel(args).to(cli.device)
    sd = torch.load(ckpt_path, map_location=cli.device, weights_only=False)
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"[regen] state_dict loaded: missing={len(miss)} unexpected={len(unexp)}")
    model.eval()
    if cli.epoch is not None:
        model.set_current_epoch(cli.epoch)
        _eps = getattr(model, "_current_sinkhorn_epsilon", lambda: None)()
        print(f"[regen] epoch set to {cli.epoch}"
              + (f"  -> sinkhorn epsilon {_eps:.4f}" if _eps is not None else ""))

    # build trainset (we want trainset because v6b Qwen captions live in
    # the trainset cache; visualize_routing reads qwen_jsonl by image_id).
    from dataloaders import ImgRtvCUB2011, ImgRtvDataset
    from extract_clip_features import _default_transform  # noqa
    from torchvision import transforms

    cache_dir = getattr(args, "siglip2_feature_cache_dir", None)
    dataset_name = getattr(args, "dataset", None)
    setting = getattr(args, "setting", "setting1")
    dataset_dir = getattr(args, "dataset_dir", "dataset")
    qwen_path = getattr(args, "qwen_text_cache_path", None)
    if dataset_name == "CIFAR10":
        # CIFAR10 is md5/byte-hash keyed and has no setting1/*.txt path
        # manifest, so ImgRtvDataset cannot build it. Use the same dispatcher
        # train_siglip2.py uses for its in-run visualization.
        from dataloaders import load_dataset as _load_dataset
        trainset, _, _ = _load_dataset(
            dataset_dir, dataset_name, setting=setting,
            train_transform=None, test_transform=None,
            load_train=True, load_database=False, load_test=False,
            return_index=True,
            qwen_text_cache_path=qwen_path,
            siglip2_feature_cache_dir=cache_dir,
        )
    elif dataset_name == "CUB_200":
        trainset = ImgRtvCUB2011(
            root=os.path.join(dataset_dir, "CUB_200"),
            mode="train",
            setting_name=setting,
            qwen_text_cache_path=qwen_path,
            siglip2_feature_cache_dir=cache_dir,
        )
    else:
        # ImgRtvDataset expects root pointing at the dataset folder
        # (e.g. dataset_dir/Flickr25k, dataset_dir/MSCOCO).
        ds_root = os.path.join(dataset_dir, dataset_name)
        trainset = ImgRtvDataset(
            root=ds_root, mode="train", setting_name=setting,
            qwen_text_cache_path=qwen_path,
            siglip2_feature_cache_dir=cache_dir,
        )

    save_path = os.path.join(cli.result_dir, cli.save_name)
    if not cli.tsne_only:
        visualize_routing(
            model, trainset,
            save_path=save_path,
            qwen_jsonl_path=getattr(args, "qwen_text_cache_path", None),
            num_samples=cli.num_samples,
            device=cli.device,
        )
        print(f"[regen] DONE -> {save_path}")

    # Optional: also regenerate viz_codebook_tsne.png. train_siglip2.py builds
    # this one on the TEST split (not train), so mirror that exactly.
    if cli.tsne or cli.tsne_only:
        from dna_utils import visualize_codebook_tsne
        from dataloaders import load_dataset as _load_dataset_t
        testset, _t = None, None
        try:
            _tr, _te, _ = _load_dataset_t(
                dataset_dir, dataset_name, setting=setting,
                train_transform=None, test_transform=None,
                load_train=False, load_database=False, load_test=True,
                return_index=True,
                qwen_text_cache_path=qwen_path,
                siglip2_feature_cache_dir=cache_dir,
            )
            testset = _te
        except Exception as e:                                   # noqa: BLE001
            print(f"[regen-tsne] test split build failed: {e}")
        if testset is not None:
            tsne_path = os.path.join(cli.result_dir, cli.tsne_save_name)
            try:
                visualize_codebook_tsne(
                    model, testset,
                    save_path=tsne_path,
                    num_samples=cli.tsne_samples,
                    device=cli.device,
                )
                print(f"[regen-tsne] DONE -> {tsne_path}")
            except Exception as e:                               # noqa: BLE001
                print(f"[regen-tsne] failed: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
