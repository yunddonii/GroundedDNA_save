#!/usr/bin/env -S -u PYTHONPATH -u PYTHONHOME -u PYTHONSTARTUP -u PYTHONINSPECT -u BASH_ENV -u ENV /home/yschoi/.conda/envs/dna_hashing/bin/python -I
"""Authoritative direct entrypoint for the Phase-5 Python supervisor.

The historical filename is retained for callers, but this is deliberately not
a shell script: Bash startup variables/functions must never execute before the
campaign gate.  The supervisor ultimately launches scripts/_ablation_exec.py.
"""
from pathlib import Path
import os
import sys

REPO = Path(__file__).resolve().parents[1]
for name in (
        "BASH_ENV", "ENV", "PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP",
        "PYTHONINSPECT", "LD_PRELOAD", "CDPATH", "GLOBIGNORE"):
    os.environ.pop(name, None)
os.environ.update({
    "PATH": (
        "/usr/local/cuda-12.4/bin:/usr/local/sbin:/usr/local/bin:"
        "/usr/sbin:/usr/bin:/sbin:/bin"),
    "LD_LIBRARY_PATH": (
        "/usr/local/cuda-12.4/lib64:/usr/local/cuda/extras/CUPTI"),
    "PYTHONNOUSERSITE": "1",
})
sys.path.insert(0, str(REPO))

from scripts._ablation_campaign_launch import main  # noqa: E402

raise SystemExit(main())
