"""Loaded by every guarded Python child through PYTHONPATH: the same boundary as the pytest process."""
import os

if os.environ.get("GDNA_GUARD_PRIVATE"):
    import gdna_guard_policy
    gdna_guard_policy.install()
