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
    print(f"[regen] parsed {sum(1 for _ in vars(args))} args from {args_path}")
    print(f"[regen] dataset={getattr(args, 'dataset', '?')}  "
          f"cache={getattr(args, 'siglip2_feature_cache_dir', '?')}")

    # build model + load state
    model = SigLIP2SemanticOTModel(args).to(cli.device)
    sd = torch.load(ckpt_path, map_location=cli.device, weights_only=False)
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"[regen] state_dict loaded: missing={len(miss)} unexpected={len(unexp)}")
    model.eval()

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
    if dataset_name == "CUB_200":
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

    save_path = os.path.join(cli.result_dir, "viz_routing_heatmap.png")
    visualize_routing(
        model, trainset,
        save_path=save_path,
        qwen_jsonl_path=getattr(args, "qwen_text_cache_path", None),
        num_samples=cli.num_samples,
        device=cli.device,
    )
    print(f"[regen] DONE -> {save_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
