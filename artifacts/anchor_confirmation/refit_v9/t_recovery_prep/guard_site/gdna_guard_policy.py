"""The r8 preparation test boundary (audits 797.3, 802.2, 804): ONE policy, enforced by an audit hook in the
pytest process AND in every Python child (sitecustomize.py next to this file, loaded through PYTHONPATH;
a Python child that would start without it is refused). Configured by environment:
GDNA_GUARD_WORKTREE, GDNA_GUARD_PRIVATE (the only place synthetic payloads and writes may exist),
GDNA_GUARD_LOGDIR (append-only logs), GDNA_GUARD_SITE (this directory).

Paths are judged on their real path; a dir_fd-relative path is resolved through /proc/self/fd.
  allowed reads : the interpreter installation and system runtime; reviewed source files of the worktree
                  (.py .pyc .sh .md .ini .txt .cfg .toml .yml .yaml, outside artifacts/ dataset/
                  result*/ logs/ cache*/); this directory; source files beside a test file named on the
                  command line (GDNA_GUARD_NAMED_DIRS); everything under the private root
  allowed writes: the private root; the log directory (also readable)
  absent names  : a read attempt under /seals/ (the tests' synthetic seal placeholders) while /seals is absent
  refused       : every other path (other /tmp, /data, /home: live data, results, caches, ledgers,
                  claims, other worktrees), /dev/nvidia*, payload suffixes outside the private root,
                  inet sockets, os.system, and every os.exec* except the launcher's own
                  pre-import self-exec (same interpreter, worktree launcher, environment keeping this hook)
Children (explicit forms, for subprocess.Popen and os.posix_spawn alike; anything else refused):
  * the interpreter, carrying this hook (no -I -E -s -S -m), running `-c CODE`, a script under the
    private root, a worktree tests/*.py file, or one of the three analysis scripts (their opens are
    then policed in the child); never anchor_terminal_test.py, extract_train_split.py, train_siglip2.py
    or any other worktree script;
  * `python <private>/entry_probe.py <worktree> <private result> <mode>`: only when its text is exactly the
    reviewed ENTRY_PROBE of tests/test_anchor_refit_stage.py (it runs without PYTHONPATH, which the launcher
    requires, and installs its own open guard);
  * `python <private>/driver.py <worktree> <private out> ...`: only as the reviewed DRIVER text of
    tests/test_anchor_confirm_env_handoff.py (it runs under a minimal start-up block by design);
  * `python -I -B <private script> <args>` with every absolute argument under the private root: a byte copy
    of the worktree's scripts/seal_phase3_inputs.py (the historical-tree fixture), or a private script whose
    syntax tree imports only json/os/sys/time/hashlib/pathlib, calls no process/exec/import machinery and
    whose absolute path literals all lie under the private root (the verifier stand-in); -I cannot carry the
    hook, so these are bound by content;
  * git, in the worktree, in exactly: rev-parse --verify HEAD | rev-parse HEAD | rev-parse <rev>:<src> |
    show <rev>:<src> | diff <rev> -- <src>, with <rev> HEAD or a hex commit and <src> a reviewed source
    path (no other option, no payload or data path);
  * bash running a pinned dataset wrapper in capture-only form (bytes = the launcher's pinned digest,
    cwd under the private root, PY = the reviewed capture shim under the private root,
    GDNA_RENDER_ARGV_OUT under the private root), or a private copy whose bytes equal a worktree
    scripts/*.sh file (the phase-2 wiring test).
"""
import hashlib
import json
import os
import re
import sys
import threading

WORKTREE = os.path.realpath(os.environ["GDNA_GUARD_WORKTREE"]) + "/"
PRIVATE = os.path.realpath(os.environ["GDNA_GUARD_PRIVATE"])
PRIVATE_ = PRIVATE + "/"
LOGDIR = os.path.realpath(os.environ["GDNA_GUARD_LOGDIR"])
SITE = os.path.realpath(os.environ["GDNA_GUARD_SITE"])
RUNNER = os.path.realpath(os.environ.get("GDNA_GUARD_RUNNER", SITE))     # the runner (pytest's __main__)
NAMED_DIRS = tuple(d for d in os.environ.get("GDNA_GUARD_NAMED_DIRS", "").split(os.pathsep) if d)
RUNTIME = tuple(sorted({os.path.realpath(p) + "/" for p in (sys.prefix, sys.base_prefix)})) + (
    "/usr/", "/lib/", "/lib64/", "/etc/", "/proc/", "/sys/", "/dev/shm/")
DEVICES = ("/dev/null", "/dev/zero", "/dev/urandom", "/dev/random", "/dev/tty")
#: synthetic placeholder roots the tests name but never create (a read is admitted only while absent)
PLACEHOLDERS = ("/seals/",)
SOURCE_SUFFIXES = (".py", ".pyc", ".sh", ".md", ".ini", ".txt", ".cfg", ".toml", ".yml", ".yaml")
WORKTREE_DATA = ("artifacts/", "dataset/", "result", "logs/", "cache")
PAYLOAD = (".npz", ".npy", ".pt", ".pth", ".safetensors", ".bin", ".ckpt", ".pkl")
ANALYSIS = ("scripts/eval_cell_bioproj.py", "scripts/pairwise_nmi.py", "scripts/seal_cell_analysis.py")
NEVER = ("anchor_terminal_test.py", "extract_train_split.py", "train_siglip2.py")
FIXTURE_MODULES = {"json", "os", "sys", "time", "hashlib", "pathlib"}
FIXTURE_CALLS = {"system", "popen", "Popen", "run", "call", "check_call", "check_output", "exec", "eval", "compile",
                 "__import__", "import_module", "run_path", "run_module", "execv", "execve", "execl", "execvp",
                 "spawnv", "spawnl", "posix_spawn", "fork", "startfile"}
REV = re.compile(r"^(HEAD|[0-9a-f]{7,40})$")
#: /proc/<pid|self>/<leaf> and /proc/<leaf>: cannot name another file system path (unlike /proc/self/root/...)
_PROC_LEAF = re.compile(r"^/proc/((\d+|self|thread-self)/)?[a-z_]+$")
WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
SHIM_TEXT = None                     # the launcher's reviewed capture shim (filled lazily from the source)
VIOLATIONS = []
_LOG = {}


def _log(name: str, row: dict) -> None:
    handle = _LOG.get(name)
    if handle is None:
        handle = _LOG[name] = open(os.path.join(LOGDIR, f"{name}.{os.getpid()}.jsonl"), "a", buffering=1)
    handle.write(json.dumps(row) + "\n")


def _source_rel_ok(rel: str) -> bool:
    return (not rel.startswith(WORKTREE_DATA) and not rel.endswith(PAYLOAD) and ".." not in rel.split("/")
            and rel.endswith(SOURCE_SUFFIXES))


def path_verdict(real: str, write: bool) -> str:
    if real == PRIVATE or real.startswith(PRIVATE_):
        return "allow-private"
    if real.startswith(LOGDIR + "/"):
        return "allow-log"
    if real in DEVICES:
        return "allow-runtime"
    if write:
        return "deny"
    if real.startswith(RUNTIME) or real in (SITE, RUNNER) or real.startswith(SITE + "/"):
        return "allow-runtime"
    if real + "/" == WORKTREE:
        return "allow-source"                    # the worktree root directory itself
    if real.startswith(WORKTREE):
        rel = real[len(WORKTREE):]
        if os.path.isdir(real) and not rel.startswith(WORKTREE_DATA):
            return "allow-source"
        return "allow-source" if _source_rel_ok(rel) or rel == "pytest.ini" else "deny"
    if NAMED_DIRS and real.startswith(NAMED_DIRS) and (os.path.isdir(real) or real.endswith(SOURCE_SUFFIXES)):
        return "allow-named-test"
    if real.startswith(PLACEHOLDERS) and not os.path.lexists(real) and not os.path.lexists(PLACEHOLDERS[0]):
        return "allow-absent-placeholder"        # the tests' synthetic /seals/... names: nothing exists there
    return "deny"


def _resolve(path, dir_fd=None) -> str:
    raw = os.fsdecode(path)
    if not os.path.isabs(raw):
        base = os.readlink(f"/proc/self/fd/{dir_fd}") if isinstance(dir_fd, int) and dir_fd >= 0 else os.getcwd()
        raw = os.path.join(base, raw)
    return os.path.realpath(raw)


def _shim_text() -> str:
    global SHIM_TEXT
    if SHIM_TEXT is None:
        src = open(os.path.join(WORKTREE, "scripts/phase3_selection_matrix.py"), encoding="utf-8").read()
        match = re.search(r'_RENDER_SHIM = \("(.*?)"\)\n', src.replace('"\n                "', ""), re.S)
        SHIM_TEXT = match.group(1).encode().decode("unicode_escape") if match else ""
    return SHIM_TEXT


def _pinned_wrappers() -> dict:
    src = open(os.path.join(WORKTREE, "scripts/phase3_selection_matrix.py"), encoding="utf-8").read()
    block = src[src.index("DATASET_SCRIPT_SHA256 = {"):]
    block = block[:block.index("}") + 1]
    return dict(re.findall(r'"(scripts/[^"]+\.sh)":\s*"([0-9a-f]{64})"', block))


def _sha(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _hooked_env(env) -> bool:
    env = os.environ if env is None else {os.fsdecode(k): os.fsdecode(v) for k, v in dict(env).items()}
    path = env.get("PYTHONPATH", "").split(os.pathsep)
    return path[:1] == [SITE] and all(env.get(k) == os.environ.get(k) for k in
                                      ("GDNA_GUARD_WORKTREE", "GDNA_GUARD_PRIVATE", "GDNA_GUARD_LOGDIR",
                                       "GDNA_GUARD_SITE"))


def _reviewed_entry_probe() -> str:
    """The r7 entrypoint probe exactly as the reviewed test file defines it (it installs its own open guard
    and stops at the first admission refusal)."""
    src = open(os.path.join(WORKTREE, "tests/test_anchor_refit_stage.py"), encoding="utf-8").read()
    start = src.index("ENTRY_PROBE = r\'\'\'") + len("ENTRY_PROBE = r\'\'\'")
    return src[start:src.index("\'\'\'", start)]


def _test_constant(rel: str, name: str) -> str:
    """A fixture text exactly as a reviewed test file of the worktree defines it (NAME = r\'\'\'...\'\'\')."""
    src = open(os.path.join(WORKTREE, rel), encoding="utf-8").read()
    marker = name + " = r\'\'\'"
    start = src.index(marker) + len(marker)
    return src[start:src.index("\'\'\'", start)]


def _abs_args_private(args) -> bool:
    return all(os.path.realpath(a).startswith(PRIVATE_) for a in args if a.startswith("/"))


def _fixture_ok(script: str) -> bool:
    """A private -I fixture, bound by its syntax tree: it imports only json/os/sys/time/hashlib/pathlib,
    calls no process, exec or import machinery, and every absolute path literal lies under the private root."""
    import ast
    try:
        tree = ast.parse(open(script, encoding="utf-8").read())
    except (OSError, UnicodeDecodeError, SyntaxError):
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods = {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            mods = {(node.module or "").split(".")[0]}
        else:
            mods = None
        if mods is not None and not mods <= FIXTURE_MODULES:
            return False
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name in FIXTURE_CALLS:
                return False
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith("/") \
                and not os.path.realpath(node.value).startswith(PRIVATE_):
            return False
    return True


def child_verdict(argv, cwd, env) -> str:
    if not argv:
        return "deny"
    exe = os.path.basename(argv[0])
    cwd = os.path.realpath(cwd)
    if exe == "git":
        rest = argv[1:]
        if cwd + "/" != WORKTREE:
            return "deny"
        if rest in (["rev-parse", "--verify", "HEAD"], ["rev-parse", "HEAD"], ["rev-parse", "--abbrev-ref", "HEAD"]):
            return "allow-git"
        if len(rest) == 2 and rest[0] in ("rev-parse", "show") and ":" in rest[1]:
            rev, src = rest[1].split(":", 1)
            return "allow-git" if REV.match(rev) and _source_rel_ok(src) else "deny"
        if len(rest) == 4 and rest[0] == "diff" and REV.match(rest[1]) and rest[2] == "--" and _source_rel_ok(rest[3]):
            return "allow-git"
        return "deny"
    if exe == "bash":
        if len(argv) < 2:
            return "deny"
        script = os.path.realpath(os.path.join(cwd, argv[1]))
        envd = os.environ if env is None else {os.fsdecode(k): os.fsdecode(v) for k, v in dict(env).items()}
        rel = script[len(WORKTREE):] if script.startswith(WORKTREE) else None
        pinned = _pinned_wrappers()
        if rel in pinned and _sha(script) == pinned[rel] and cwd.startswith(PRIVATE_):
            shim, out = envd.get("PY", ""), envd.get("GDNA_RENDER_ARGV_OUT", "")
            if os.path.realpath(shim).startswith(PRIVATE_) and os.path.realpath(out).startswith(PRIVATE_):
                body = open(shim, encoding="utf-8").read().split("\n", 1)[-1]
                return "allow-capture-render" if body == _shim_text() else "deny"
            return "deny"
        if script.startswith(PRIVATE_) and script.endswith(".sh"):
            twin = os.path.join(WORKTREE, "scripts", os.path.basename(script))
            if os.path.isfile(twin) and _sha(twin) == _sha(script):
                return "allow-private-copy-of-reviewed-script"
        return "deny"
    if not (exe.startswith("python") or os.path.realpath(argv[0]) == os.path.realpath(sys.executable)):
        return "deny"
    flags, i = [], 1
    while i < len(argv) and argv[i].startswith("-") and argv[i] != "-c":
        flags.append(argv[i])
        i += 1
    if i >= len(argv):
        return "deny"
    isolated = "-I" in flags
    if any(f in ("-E", "-s", "-S", "-m") or f.startswith("-m") for f in flags) or \
            any(f not in ("-B", "-u", "-I") and not f.startswith(("-W", "-X")) for f in flags):
        return "deny"
    if argv[i] == "-c":
        return "allow-python-c" if not isolated and _hooked_env(env) else "deny"
    script = os.path.realpath(os.path.join(cwd, argv[i]))
    if not isolated and not flags and script.startswith(PRIVATE_) and os.path.basename(script) == "entry_probe.py" \
            and len(argv) == i + 4 and os.path.realpath(argv[i + 1]) + "/" == WORKTREE \
            and os.path.realpath(argv[i + 2]).startswith(PRIVATE_) \
            and argv[i + 3] in ("nothing-imported-first", "launcher-imported-first"):
        # the r7 entrypoint probe runs without PYTHONPATH (the launcher requires that), so it cannot carry
        # this hook: it is admitted only as the reviewed, self-guarded fixture text
        return "allow-reviewed-entry-probe" if open(script, encoding="utf-8").read() == _reviewed_entry_probe() \
            else "deny"
    if not isolated and not flags and script.startswith(PRIVATE_) and os.path.basename(script) == "driver.py" \
            and len(argv) == i + 7 and os.path.realpath(argv[i + 1]) + "/" == WORKTREE \
            and os.path.realpath(argv[i + 2]).startswith(PRIVATE_):
        # the environment-handoff driver runs under a minimal start-up block (no PYTHONPATH) by design: it is
        # admitted only as the reviewed DRIVER text of tests/test_anchor_confirm_env_handoff.py
        return "allow-reviewed-handoff-driver" if open(script, encoding="utf-8").read() == \
            _test_constant("tests/test_anchor_confirm_env_handoff.py", "DRIVER") else "deny"
    if isolated:
        if not script.startswith(PRIVATE_) or not _abs_args_private(argv[i + 1:]):
            return "deny"
        twin = os.path.join(WORKTREE, "scripts", os.path.basename(script))
        if script.endswith("/scripts/seal_phase3_inputs.py") and _sha(script) == _sha(twin):
            return "allow-isolated-reviewed-copy"        # a byte copy of the reviewed verifier on private inputs
        return "allow-isolated-private-fixture" if _fixture_ok(script) else "deny"
    if not _hooked_env(env) or os.path.basename(script) in NEVER:
        return "deny"
    if script.startswith(PRIVATE_):
        return "allow-private-script"
    rel = script[len(WORKTREE):] if script.startswith(WORKTREE) else ""
    if rel in ANALYSIS or (rel.startswith("tests/") and rel.endswith(".py") and "/" not in rel[6:]):
        return "allow-hooked-worktree-script"
    return "deny"


_WRITE_EVENTS = {"os.remove": (0, 1), "os.rmdir": (0, 1), "os.mkdir": (0, 2), "os.chmod": (0, 2),
                 "os.chown": (0, 3), "os.truncate": (0, None), "os.utime": (0, 4), "os.symlink": (1, 2),
                 "os.link": (1, 3)}


def _deny(kind: str, what: str, message: str):
    VIOLATIONS.append(f"{kind}{what}")
    row = {"kind": kind, "what": what}
    if os.environ.get("GDNA_GUARD_TRACE") and os.environ["GDNA_GUARD_TRACE"] in what:
        import traceback
        row["stack"] = traceback.format_stack()[-12:-2]           # diagnosis only
    _log("violations", row)
    raise PermissionError(f"boundary guard refused {message}: {what}")


_STATE = threading.local()


def hook(event, args):
    # the guard's own reads and log writes raise audit events too: they are not re-judged
    if getattr(_STATE, "busy", False):
        return
    _STATE.busy = True
    try:
        _judge(event, args)
    finally:
        _STATE.busy = False


def _exec_ok(args) -> bool:
    """The launcher's pre-import self-exec only: the same interpreter re-running the worktree launcher with
    an environment that keeps this hook (so the new image is guarded again)."""
    try:
        path, argv, env = os.fsdecode(args[0]), [os.fsdecode(a) for a in args[1]], args[2]
    except (IndexError, TypeError):
        return False
    script = [a for a in argv[1:] if not a.startswith("-")][:1]
    return (os.path.realpath(path) == os.path.realpath(sys.executable) and _hooked_env(env) and bool(script)
            and os.path.realpath(script[0]) == os.path.join(WORKTREE, "scripts/phase3_selection_matrix.py"))


def _judge(event, args):
    if event == "os.exec" and _exec_ok(args):
        _log("children", {"exec": [os.fsdecode(a) for a in args[1]][:6], "decision": "allow-launcher-self-exec"})
        return
    if event in ("os.scandir", "os.listdir"):
        target = args[0] if args else None
        if isinstance(target, int):
            try:
                listed = os.path.realpath(os.readlink(f"/proc/self/fd/{target}"))
            except OSError:
                return
            if listed.startswith(PRIVATE_):
                _STATE.listed = (getattr(_STATE, "listed", ()) + (listed,))[-16:]
        return
    if event == "os.posix_spawn":
        # subprocess uses posix_spawn for some Popen calls (e.g. close_fds=False): the same child rules apply
        try:
            argv = [os.fsdecode(a) for a in args[1]]
        except (IndexError, TypeError):
            argv = []
        decision = child_verdict(argv, os.getcwd(), args[2] if len(args) > 2 else None) if argv else "deny"
        _log("children", {"argv": argv[:10], "cwd": os.getcwd(), "decision": decision, "via": "posix_spawn"})
        if decision == "deny":
            _deny("spawn:", " ".join(argv[:5]), "a child process")
        return
    if event in ("os.exec", "os.system", "os.spawn"):
        _deny("exec:", str(args[0])[:200], "a process image replacement or shell")
    if event == "socket.connect":
        address = args[1] if len(args) > 1 else None
        if isinstance(address, tuple):
            _deny("net:", str(address)[:200], "a network connection")
        return
    if event == "subprocess.Popen":
        argv = [os.fsdecode(a) for a in (args[1] or [])] if len(args) > 1 and args[1] else []
        cwd = os.fsdecode(args[2]) if len(args) > 2 and args[2] else os.getcwd()
        env = args[3] if len(args) > 3 else None
        try:
            decision = child_verdict(argv, cwd, env)
        except (OSError, ValueError, UnicodeDecodeError):
            decision = "deny"
        _log("children", {"argv": argv[:10], "cwd": cwd, "decision": decision})
        if decision == "deny":
            _deny("spawn:", " ".join(argv[:5]), "a child process")
        return
    if event == "open":
        if not args or isinstance(args[0], int):
            return
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 else None
        write = (isinstance(mode, str) and any(c in mode for c in "wax+")) or \
            (isinstance(flags, int) and bool(flags & WRITE_FLAGS))
        raw = os.fsdecode(args[0])
        if _PROC_LEAF.match(raw) and not write:
            return                               # a single-level /proc file (no traversal possible): fast path
        if raw in ("<string>", "<stdin>"):
            return                               # a traceback looking up the source of `-c` code (no file)
        if not os.path.isabs(raw) and isinstance(flags, int) and flags & getattr(os, "O_DIRECTORY", 0) \
                and not write:
            return                               # a dir_fd-relative directory walk (no payload read)
        real = _resolve(raw)
        if not os.path.isabs(raw) and "/" not in raw and not write:
            # shutil.rmtree opens entries by bare name relative to a directory descriptor the audit event
            # does not carry; resolve the name against the private directories this thread just listed
            for listed in reversed(getattr(_STATE, "listed", ())):
                if os.path.lexists(os.path.join(listed, raw)):
                    real = os.path.realpath(os.path.join(listed, raw))
                    break
        if any(real == os.path.realpath(h.name) for h in _LOG.values()):
            return
        decision = path_verdict(real, write)
        if decision not in ("allow-runtime", "allow-source"):
            _log("opened", {"path": raw, "real": real, "write": write, "decision": decision})
        if decision == "deny":
            _deny("write:" if write else "", real, "an open")
        return
    if event in _WRITE_EVENTS:
        index, fd_index = _WRITE_EVENTS[event]
        try:
            dir_fd = args[fd_index] if fd_index is not None and fd_index < len(args) else None
            real = _resolve(args[index], dir_fd)
        except (TypeError, ValueError, OSError, IndexError):
            return
        if path_verdict(real, True) == "deny":
            _deny("write:", real, f"{event}")
        return
    if event in ("os.rename", "shutil.copyfile", "shutil.copytree", "shutil.move"):
        try:
            targets = [_resolve(args[1], args[3] if event == "os.rename" and len(args) > 3 else None)]
        except (TypeError, ValueError, OSError, IndexError):
            return
        for real in targets:
            if path_verdict(real, True) == "deny":
                _deny("write:", real, f"{event}")


def install() -> None:
    sys.addaudithook(hook)
