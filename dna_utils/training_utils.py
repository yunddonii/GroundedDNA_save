"""Training-side utilities used by the SigLIP2 framework.

Lifted verbatim from the legacy ``utils.py`` so that ``train_siglip2.py`` and
``extraction_siglip2.py`` do not have to import that file (which transitively
loads ``DWKM.py`` and the optional ``sinkhornbarycenters`` package — neither
is needed for the SigLIP2 pipeline).

Three exports:
    get_transform          torchvision pipeline (train / test)
    print_one_epoch_info   pretty-print per-epoch loss dict
    get_summarywriter      build TensorBoard SummaryWriter pair (train / val)
"""

from __future__ import annotations

import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from timm.data.transforms import RandomResizedCropAndInterpolation
from timm.data.random_erasing import RandomErasing


def get_transform(typ=['train', 'test']):
    """Per-stage torchvision transform.

    Mirrors `utils.get_transform` exactly so the dataloader behaves identically
    whether we import from ``utils`` (legacy) or ``dna_utils`` (new framework).
    """
    if typ == 'train':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            RandomResizedCropAndInterpolation(
                224, scale=(1.0, 1.0), ratio=(1.0, 1.0), interpolation='bicubic',
            ),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.0),
            transforms.ColorJitter(0),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616),
            ),
            RandomErasing(0.25, mode='const', max_area=1, device='cpu'),
        ])
    elif typ == 'test':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616),
            ),
        ])
    else:
        raise NotImplementedError(f"unknown transform type: {typ!r}")


def print_one_epoch_info(epoch, loss: list):
    """Pretty-print a list of [train_result, val_result] loss dicts."""
    bar = "-" * 40
    typ = ['train', 'val']
    print(bar)
    print(f"{'epoch':25s}{epoch:>15d}")
    for t, i in enumerate(loss):
        for k, v in i.items():
            print(f"{typ[t] + '_' + k:25s}{v:>15.5f}")
        print(bar)


def get_summarywriter(loss, log_path):
    """Build (train_writer, val_writer) with a custom_scalars Multiline layout.

    ``log_path`` may have a trailing separator (the SigLIP2 flat-save path uses
    one); ``SummaryWriter(log_dir=log_path + '/train')`` resolves correctly on
    POSIX even with a double slash.
    """
    contents = ["Multiline", [f"loss/{l}" for l in loss]]
    train_layout = {'train': {"loss": contents}}
    val_layout   = {'val':   {"loss": contents}}

    train_writer = SummaryWriter(log_dir=log_path + '/train')
    train_writer.add_custom_scalars(train_layout)
    val_writer = SummaryWriter(log_dir=log_path + '/val')
    val_writer.add_custom_scalars(val_layout)

    return train_writer, val_writer
