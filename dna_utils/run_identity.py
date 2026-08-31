"""Immutable run identity and fail-closed lookup (F08).

`train_siglip2._resolve_save_path` derives the result directory from date, tag,
batch, epoch and LR and then calls `os.makedirs(..., exist_ok=True)`. The seed,
the slot count, the bases per slot and the selection/refit mode never reach the
path, so two runs differing only in those share a directory and the second
overwrites the first.

That happened. On 2026-08-12 eight 3-seed cells were launched with a variable
name the launcher does not read, and every seed for a given (dataset, N) wrote
to one path. What survived was worse than nothing: `args.txt` came from whichever
process started last and the evaluation JSON from whichever finished last, so a
directory labelled seed 44 contained seed 43's numbers. It was caught only
because two seeds agreed to full float precision. The wreckage is kept under
`result_quarantine_collided_20260812/`.

Lookup has the same defect in the other direction. `ls -dt <glob> | head -1`
returns the newest match, so an ablation or a lambda variant sharing the prefix
can be picked as the reference run and evaluated as if it were.

The contract here:

    identity   digest over everything that distinguishes one run from another
    claim      refuse a directory already claimed by a DIFFERENT identity, and
               refuse a non-empty directory with no manifest at all
    resolve    match on identity, never on mtime; ambiguity and absence both
               raise instead of guessing

Quarantined collision roots are excluded from resolution: they are preserved as
evidence, and silently reading from them is exactly what the quarantine exists
to prevent.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, fields
from typing import Any, List, Optional

MANIFEST_NAME = "run_identity.json"
#: v2 added the scientific axes below. v1 digests covered only the geometry and
#: the schedule, so changing the codebook size, the feature/caption/whitening
#: cache, the Sinkhorn epsilon endpoints, the validation split, the batch size
#: or the projection learning rate left the digest identical -- ten axes that
#: define different experiments, all colliding on one directory.
#:
#: v3 added the RECIPE axes. Until it, three cells differing only in the
#: confidence-adaptive top-p window produced one digest, and so did the four
#: cells of the A5 factorial, which differ in the Gumbel relaxation and in
#: L_joint. They were kept apart only by `--selection_mode`, a free string the
#: launcher passes: a cell whose flags were composed wrongly -- which is what
#: the shell chain did -- still matched the label it claimed. These fields are
#: what the cells actually change, so the label is now redundant rather than
#: load-bearing.
#:
#: Bumping the version makes `load_run_manifest` return None for a v2 manifest.
#: That is deliberate and it is why this landed now: the sixteen Phase-3
#: selection records are being re-run anyway, because they chose N at a top-p
#: window the draft itself calls the M=6 tuning. Adding these fields while
#: leaving the version at 2 -- which an earlier attempt did -- is the actual
#: mistake: it makes old manifests unreadable while still claiming to be the
#: schema they were written under.
#:
#: v4 adds `no_routing_adaptive_topp`. `--routing_adaptive_topp` and its
#: negation are SEPARATE arguments, and the model reads the conjunction
#: (model_siglip2.py:2277), so a run with the mechanism switched off carried an
#: identity that said it was on: the two digests were equal. The version is
#: bumped rather than the field being added in place, which is the mistake v3
#: made -- three manifests written under v3 became unreadable while still
#: claiming to be v3.
_SCHEMA_VERSION = 4

#: Claimed by a live process. Created with O_EXCL, so two processes cannot both
#: believe they own the directory; removed by `release_run_dir` when the run
#: finishes.
ACTIVE_CLAIM_NAME = "run_active_claim.json"

#: Directory-name fragments that must never be resolved as a live run.
_EXCLUDED_FRAGMENTS = ("quarantine", "_QUARANTINE", "smoke")


def _opt_float(value: Any) -> Optional[float]:
    return None if value is None else float(value)


class RunCollision(RuntimeError):
    """A directory is already claimed by a different run, or two claim one."""


class RunNotFound(RuntimeError):
    """No directory carries this identity. Raised instead of falling back."""


@dataclass(frozen=True)
class RunIdentity:
    schema_version: int
    dataset: str
    setting: str
    seed: int
    num_slots: int
    bases_per_slot: int
    total_bases: int
    total_bits: int
    stop_after_epoch: Optional[int]
    epoch_budget: int
    lr_schedule_horizon: int
    sinkhorn_schedule_horizon: int
    selection_mode: str
    # ---- the scientific axes (v2) ----
    codebook_size: int
    feature_cache: str
    eval_cache: str
    qwen_cache: str
    text_whiten: str
    sinkhorn_epsilon_init: Optional[float]
    sinkhorn_epsilon_final: Optional[float]
    val_split_ratio: float
    val_split_seed: int
    batch_size: int
    proj_lr: float
    # ---- the recipe axes (v3) ----
    routing_adaptive_topp: bool
    no_routing_adaptive_topp: bool
    routing_adaptive_topp_min: Optional[float]
    routing_adaptive_topp_max: Optional[float]
    # The window means nothing without the switch that reads it and the rule
    # that computes it: with `--routing_adaptive_topp` off, min/max are inert
    # defaults, and `--routing_adaptive_topp_entropy` changes the threshold
    # from max-probability confidence to normalised routing entropy -- a
    # different mask at identical min/max. `--routing_perplexity_topk` is
    # mutually exclusive with the whole mechanism, so a run using it is not a
    # top-p run at all.
    routing_adaptive_topp_entropy: bool
    routing_perplexity_topk: bool
    # L_joint's scope and its log floor. `--codon_joint_slots 0` applies the
    # term to the global slot only, which is a different experiment at the
    # same lambda; the floor sets where the log is clamped.
    codon_joint_slots: str
    codon_joint_floor: Optional[float]
    share_codebook: bool
    disable_text_supervision: bool
    use_gumbel_softmax: bool
    lambda_codon_joint: float
    lambda_text_code_kl: float
    lambda_text_hash_ntxent: float
    lambda_xmodal_commit: float
    lambda_codeword_codon_sinkhorn: float

    @property
    def digest(self) -> str:
        payload = {f.name: getattr(self, f.name) for f in fields(self)
                   if f.name != "schema_version"}
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    #: Files up to this size enter the identity by content. The caption
    #: caches are the largest at ~9 MiB, and hashing one takes milliseconds --
    #: the first cap was 8 MiB, which put NUS-WIDE (8.8 MiB) and MS-COCO
    #: (8.3 MiB) on the path-only branch, so swapping their contents under the
    #: same filename left the identity unchanged.
    _CONTENT_LIMIT = 256 << 20

    @staticmethod
    def _digest_file(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()[:16]

    @classmethod
    def _artifact(cls, path: Any) -> str:
        """How an input artefact enters the identity.

        A file enters by content. A feature cache is a DIRECTORY of tens of
        gigabytes, which cannot be digested per run -- but `meta.json` is the
        file the provenance gate already checks, and it carries the backbone
        revision, the canonical transform and the row count. Hashing that
        binds the directory to the snapshot that produced it, which is the
        property that matters here; the array bytes are the loader's gate to
        defend.

        A path that exists as neither is still recorded, so a cache that is not
        on this machine does not collapse to the same identity as no cache.
        """
        if path in (None, ""):
            return ""
        real = os.path.realpath(str(path))
        if os.path.isfile(real):
            if os.path.getsize(real) <= cls._CONTENT_LIMIT:
                return f"{real}#{cls._digest_file(real)}"
            return f"{real}#too-large-to-digest"
        if os.path.isdir(real):
            meta = os.path.join(real, "meta.json")
            if os.path.isfile(meta):
                return f"{real}#meta:{cls._digest_file(meta)}"
            return f"{real}#no-meta"
        return f"{real}#absent"

    @classmethod
    def from_args(cls, args: Any) -> "RunIdentity":
        m = int(getattr(args, "num_semantic_parts", 0) or 0)
        l = int(getattr(args, "num_codons_per_codebook", 3) or 3)
        budget = int(getattr(args, "epoch", 60) or 60)
        stop = getattr(args, "stop_after_epoch", None)
        lr_h = getattr(args, "lr_schedule_horizon", None)
        sk_h = getattr(args, "sinkhorn_schedule_horizon", None)
        return cls(
            schema_version=_SCHEMA_VERSION,
            dataset=str(getattr(args, "dataset", "")),
            setting=str(getattr(args, "setting", "")),
            seed=int(getattr(args, "random_seed", 42) or 42),
            num_slots=m, bases_per_slot=l,
            total_bases=m * l, total_bits=2 * m * l,
            stop_after_epoch=None if stop is None else int(stop),
            epoch_budget=budget,
            lr_schedule_horizon=int(lr_h) if lr_h else budget,
            sinkhorn_schedule_horizon=int(sk_h) if sk_h else budget,
            selection_mode=str(getattr(args, "selection_mode", "refit")),
            codebook_size=int(getattr(args, "codebook_size", 0) or 0),
            feature_cache=cls._artifact(
                getattr(args, "siglip2_feature_cache_dir", None)),
            eval_cache=cls._artifact(getattr(args, "eval_cache_dir", None)),
            qwen_cache=cls._artifact(
                getattr(args, "qwen_text_cache_path", None)),
            text_whiten=cls._artifact(getattr(args, "text_whiten_npz", None)),
            sinkhorn_epsilon_init=_opt_float(
                getattr(args, "sinkhorn_epsilon_init", None)),
            sinkhorn_epsilon_final=_opt_float(
                getattr(args, "sinkhorn_epsilon_final", None)),
            val_split_ratio=float(getattr(args, "val_split_ratio", 0.0) or 0.0),
            val_split_seed=int(getattr(args, "val_split_seed", 42) or 42),
            batch_size=int(getattr(args, "batch_size", 0) or 0),
            proj_lr=float(getattr(args, "proj_lr", 0.0) or 0.0),
            routing_adaptive_topp=bool(
                getattr(args, "routing_adaptive_topp", False)),
            no_routing_adaptive_topp=bool(
                getattr(args, "no_routing_adaptive_topp", False)),
            routing_adaptive_topp_min=_opt_float(
                getattr(args, "routing_adaptive_topp_min", None)),
            routing_adaptive_topp_max=_opt_float(
                getattr(args, "routing_adaptive_topp_max", None)),
            routing_adaptive_topp_entropy=bool(
                getattr(args, "routing_adaptive_topp_entropy", False)),
            routing_perplexity_topk=bool(
                getattr(args, "routing_perplexity_topk", False)),
            codon_joint_slots=str(getattr(args, "codon_joint_slots", "") or ""),
            codon_joint_floor=_opt_float(
                getattr(args, "codon_joint_floor", None)),
            share_codebook=bool(getattr(args, "share_codebook", False)),
            disable_text_supervision=bool(
                getattr(args, "disable_text_supervision", False)),
            # `--no_gumbel_softmax` stores into `use_gumbel_softmax`, so the
            # axis is read under the name the parser keeps. Reading the flag's
            # own spelling would give None for every run and collapse the A5
            # factorial back to two configurations.
            use_gumbel_softmax=bool(getattr(args, "use_gumbel_softmax", True)),
            lambda_codon_joint=float(
                getattr(args, "lambda_codon_joint", 0.0) or 0.0),
            lambda_text_code_kl=float(
                getattr(args, "lambda_text_code_kl", 0.0) or 0.0),
            lambda_text_hash_ntxent=float(
                getattr(args, "lambda_text_hash_ntxent", 0.0) or 0.0),
            lambda_xmodal_commit=float(
                getattr(args, "lambda_xmodal_commit", 0.0) or 0.0),
            lambda_codeword_codon_sinkhorn=float(
                getattr(args, "lambda_codeword_codon_sinkhorn", 0.0) or 0.0),
        )

    def differing_fields(self, other: "RunIdentity") -> List[str]:
        out = []
        for f in fields(self):
            if f.name == "schema_version":
                continue
            if getattr(self, f.name) != getattr(other, f.name):
                out.append(f.name)
        return out

    def as_record(self) -> dict:
        d = asdict(self)
        d["digest"] = self.digest
        return d


def _manifest_path(run_dir: str) -> str:
    return os.path.join(run_dir, MANIFEST_NAME)


def load_run_manifest(run_dir: str) -> Optional[RunIdentity]:
    p = _manifest_path(run_dir)
    if not os.path.exists(p):
        return None
    try:
        d = json.load(open(p))
    except (OSError, json.JSONDecodeError):
        return None
    stored = d.pop("digest", None)
    if d.get("schema_version") != _SCHEMA_VERSION:
        return None
    try:
        identity = RunIdentity(**d)
    except TypeError:
        return None
    # The digest is a property of the fields, so a forged one cannot change
    # what this run IS -- but a manifest whose two halves disagree has been
    # edited, and saying so is better than silently preferring one half.
    if stored is not None and stored != identity.digest:
        return None
    return identity


def write_run_manifest(run_dir: str, identity: RunIdentity) -> str:
    os.makedirs(run_dir, exist_ok=True)
    out = _manifest_path(run_dir)
    tmp = out + f".{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        json.dump(identity.as_record(), fh, indent=2, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, out)
    return out


def _looks_occupied(run_dir: str) -> bool:
    """Artefacts that mean a previous run wrote here."""
    if not os.path.isdir(run_dir):
        return False
    markers = ("model_state_dict.pth", "extract_db.npz", "extract_query.npz",
               "args.txt", "log.csv")
    return any(os.path.exists(os.path.join(run_dir, m)) for m in markers)


def _active_claim_path(run_dir: str) -> str:
    return os.path.join(run_dir, ACTIVE_CLAIM_NAME)


def read_active_claim(run_dir: str) -> Optional[dict]:
    try:
        with open(_active_claim_path(run_dir), encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _claim_is_live(claim: Optional[dict]) -> bool:
    """Is the process that took this claim still running?

    A machine that lost power leaves a claim behind; refusing forever would
    turn a crash into a permanent block. `os.kill(pid, 0)` distinguishes a live
    owner from a stale file, and the boot id guards against PID reuse across
    reboots.
    """
    if not isinstance(claim, dict):
        return False
    pid = claim.get("pid")
    if not isinstance(pid, int):
        return False
    if claim.get("boot_id") != _boot_id():
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _boot_id() -> str:
    try:
        with open("/proc/sys/kernel/random/boot_id", encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return ""


def release_run_dir(run_dir: str) -> None:
    """Give up the active claim. Safe to call when nothing is claimed."""
    try:
        os.unlink(_active_claim_path(run_dir))
    except FileNotFoundError:
        pass


def claim_run_dir(run_dir: str, identity: RunIdentity, *,
                  resume: bool = False) -> str:
    """Take exclusive ownership of `run_dir`, or refuse.

    Three refusals, all of which were reachable before:

    * a DIFFERENT identity already owns the directory -- merging those produced
      a directory holding one seed's args and another's metrics;
    * artefacts with no manifest at all, so the run that wrote them cannot be
      named;
    * the SAME identity, held by a process that is still running. Same-identity
      re-claim used to be allowed unconditionally as "a resume", so two
      concurrent cells both believed they owned the directory and the last
      manifest simply overwrote the first. Forty concurrent claims were admitted
      forty times. A genuine resume of a finished run is still possible, but the
      caller has to say `resume=True` rather than get it by default.

    The active claim is created with O_EXCL, so the check and the take are one
    operation instead of two racing ones. What it does NOT survive: someone
    deleting the claim file by hand, and a shared filesystem whose O_EXCL is not
    atomic. It is a guard against two launchers on this machine, not against an
    adversary.
    """
    existing = load_run_manifest(run_dir)
    if existing is not None and existing.digest != identity.digest:
        diff = identity.differing_fields(existing)
        raise RunCollision(
            f"{run_dir} is already claimed by a different run; they differ "
            f"in {diff}. Writing here would merge two runs\' artefacts -- "
            f"give this one its own directory instead.")
    if existing is None and _looks_occupied(run_dir):
        raise RunCollision(
            f"{run_dir} holds run artefacts but no {MANIFEST_NAME}, so the run "
            f"that produced them cannot be identified. Move it aside rather "
            f"than writing into it.")

    os.makedirs(run_dir, exist_ok=True)
    payload = json.dumps({
        "pid": os.getpid(),
        "boot_id": _boot_id(),
        "digest": identity.digest,
    }, sort_keys=True)
    try:
        fd = os.open(_active_claim_path(run_dir),
                     os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        held = read_active_claim(run_dir)
        # The owner re-claiming its own directory is not a collision; it is the
        # same run asking twice.
        if isinstance(held, dict) and held.get("pid") == os.getpid() \
                and held.get("boot_id") == _boot_id():
            return run_dir
        if _claim_is_live(held):
            raise RunCollision(
                f"{run_dir} is being written RIGHT NOW by pid "
                f"{held.get('pid')}. Two processes claiming one directory is "
                f"how a run labelled one seed came to hold another\'s numbers.")
        if not resume:
            raise RunCollision(
                f"{run_dir} carries a stale claim from pid "
                f"{(held or {}).get('pid')}, which is no longer running. Pass "
                f"resume=True to continue that run deliberately, or move the "
                f"directory aside.")
        release_run_dir(run_dir)
        fd = os.open(_active_claim_path(run_dir),
                     os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(payload)

    if existing is None:
        write_run_manifest(run_dir, identity)
    return run_dir


def resolve_run_dir(search_root: str, identity: RunIdentity) -> str:
    """Find the ONE directory carrying this identity.

    Matching is on the manifest, never on mtime, so a newer ablation cannot be
    returned as the reference run. Absence and ambiguity both raise: falling
    back to the newest match is the defect this replaces.
    """
    hits: List[str] = []
    # `followlinks=True`: completed runs are archived to /data with a symlink
    # left at the original path, so the default would walk straight past every
    # archived run and report RunNotFound for a run that exists -- which reads
    # as "not done yet" and invites re-running it. The excluded-fragment filter
    # below and the manifest digest still bound what can match, and run
    # directories do not nest, so there are no cycles to follow.
    seen: set = set()
    for dirpath, dirnames, _ in os.walk(search_root, followlinks=True):
        real = os.path.realpath(dirpath)
        if real in seen:                 # a link back into an already-walked tree
            dirnames[:] = []
            continue
        seen.add(real)
        dirnames[:] = [d for d in dirnames
                       if not any(x in d for x in _EXCLUDED_FRAGMENTS)]
        if any(x in dirpath for x in _EXCLUDED_FRAGMENTS):
            continue
        md = load_run_manifest(dirpath)
        if md is not None and md.digest == identity.digest:
            hits.append(dirpath)
    if not hits:
        raise RunNotFound(
            f"no run under {search_root} carries identity {identity.digest[:12]} "
            f"(dataset={identity.dataset} seed={identity.seed} "
            f"M={identity.num_slots} L={identity.bases_per_slot} "
            f"stop={identity.stop_after_epoch} mode={identity.selection_mode}). "
            f"Run it, or point at the right root -- do not substitute a "
            f"similar directory.")
    if len(hits) > 1:
        raise RunCollision(
            f"{len(hits)} directories claim identity {identity.digest[:12]}: "
            f"{sorted(hits)}. Resolve which is authoritative; picking the "
            f"newest is how an ablation got evaluated as the reference run.")
    return hits[0]
