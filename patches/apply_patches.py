"""Rebuild the harness tree that the E154 notebook runs, from the inherited Duck harness and the patches in this folder.

Plain words: the notebook does not ship the harness. It copies the Tufa Labs Duck harness bundle, applies Daniel
Franzen's Milestone 2 patch, then our eight patches, in a fixed order. This script does the same on a copy, so you can
read, test or diff the exact files the agent ran with.

  python -I patches/apply_patches.py --harness <dir that contains ARC3-Inference/> --out <new dir> [--e149 v3|v4]

--harness  a copy of the Duck harness source (for example the `src/` folder of the Kaggle dataset
           dfranzen/taaf-kaggle-source-bundle-copy, or Tufa Labs' duck-harness at commit 7652836). It is not changed.
--out      where the patched copy is written (must not exist).
--e149     also apply the E149 research patches (v3: display clock; v4: v3 + clock origin).

Franzen's patch is read from the notebook's `%%writefile /kaggle/harness-changes.patch` cell (it is his code, Apache-2.0,
so it is not duplicated here). Patches are applied with `git apply` (git must be installed), with core.autocrlf off, as
on Kaggle's Linux. The script prints the sha256 of solver.py and tool_agent.py. For E154 the kernel's own log
(version 189) printed solver.py sha256 0f36330a...; `--check` exits 1 if the rebuilt file differs.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REL = os.path.dirname(HERE)
NOTEBOOK = os.path.join(REL, "notebook", "arc3-e154-keepalive.ipynb")
ORDER = ["01-e040-context-window.patch", "02-arm-reasoning-effort-switch.patch", "03-e055-bad-words-switch.patch",
         "04-e058-temperature-per-request.patch", "05-e077-state-handoff.patch", "06-e091-max-output.patch",
         "07-e097-harness-v3.patch", "08-e154-keepalive.patch"]
E154_SOLVER_SHA256 = "0f36330aea89e29227e8ab1bfb923789ad0d57df5a3d5111eca67c9d799e419b"   # printed by the v189 commit log
E149_V3_TOOL_AGENT_SHA256 = "9cdd1cef97fb42b9b778c204f72d9837213f608a6cfec214aec68960d4c0177e"  # printed by the v190 commit log
E149_V4_AFTER = ("13a8ee4821089282", "e9d5548e32be3503")   # 16-digit prefixes: the v4 apply cell pins them, version 191 logged them


def franzen_patch():
    nb = json.load(open(NOTEBOOK, encoding="utf-8"))
    for c in nb["cells"]:
        s = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        if s.startswith("%%writefile /kaggle/harness-changes.patch\n"):
            body = s.split("\n", 1)[1]
            return body if body.endswith("\n") else body + "\n"
    raise SystemExit("harness-changes.patch cell not found in the notebook")


def git_apply(patch_path, cwd, include=None):
    env = dict(os.environ, GIT_CONFIG_PARAMETERS="'core.autocrlf=false'", GIT_CEILING_DIRECTORIES=os.path.dirname(cwd))
    args = ["git", "apply", "-v"] + (["--include=" + include] if include else []) + [patch_path]
    r = subprocess.run(args, cwd=cwd, capture_output=True, text=True, env=env)
    return r.returncode, (r.stderr or r.stdout).strip()


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--harness", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--e149", choices=["v3", "v4"])
    ap.add_argument("--check", action="store_true", help="exit 1 unless solver.py matches the hash the v189 run logged")
    ap.add_argument("--lf", action="store_true", help="convert CRLF to LF in the copy first (a Windows checkout of the harness)")
    a = ap.parse_args(argv)
    if not os.path.isdir(os.path.join(a.harness, "ARC3-Inference")):
        print(f"{a.harness} has no ARC3-Inference/ folder")
        return 2
    if os.path.exists(a.out):
        print(f"{a.out} exists; choose a new folder")
        return 2
    os.makedirs(a.out)
    shutil.copytree(os.path.join(a.harness, "ARC3-Inference"), os.path.join(a.out, "ARC3-Inference"))
    if a.lf:
        for dp, _, fn in os.walk(a.out):
            for f in fn:
                p = os.path.join(dp, f)
                b = open(p, "rb").read()
                if b"\r\n" in b:
                    open(p, "wb").write(b.replace(b"\r\n", b"\n"))
    work = os.path.join(a.out, ".patches")
    os.makedirs(work)
    fp = os.path.join(work, "harness-changes.patch")
    open(fp, "w", encoding="utf-8", newline="\n").write(franzen_patch())
    steps = [("harness-changes.patch (Franzen, from the notebook)", fp, "ARC3-Inference/*")]
    steps += [(n, os.path.join(HERE, "harness", n), None) for n in ORDER]
    if a.e149:
        steps.append(("e149-v3-display-clock.patch", os.path.join(HERE, "e149", "e149-v3-display-clock.patch"), None))
    if a.e149 == "v4":
        steps.append(("e149-v4-clock-origin.patch", os.path.join(HERE, "e149", "e149-v4-clock-origin.patch"), None))
    ok = True
    for name, path, inc in steps:
        rc, msg = git_apply(path, a.out, inc)
        print(f"{'ok  ' if rc == 0 else 'FAIL'} {name}" + ("" if rc == 0 else f": {msg[-400:]}"))
        ok &= rc == 0
    shutil.rmtree(work, ignore_errors=True)
    solver = sha_file(os.path.join(a.out, "ARC3-Inference", "inference", "framework", "solver.py"))
    agent = sha_file(os.path.join(a.out, "ARC3-Inference", "inference", "agent", "tool_agent.py"))
    print(f"solver.py     sha256 {solver}")
    print(f"tool_agent.py sha256 {agent}")
    if not a.e149:
        same = solver == E154_SOLVER_SHA256
        print(f"solver.py {'MATCHES' if same else 'DIFFERS FROM'} the hash the kernel version 189 commit logged (0f36330a...)")
        if a.check and not same:
            ok = False
    elif a.e149 == "v3":
        same = agent == E149_V3_TOOL_AGENT_SHA256 and solver == E154_SOLVER_SHA256
        print(f"solver.py and tool_agent.py {'MATCH' if same else 'DIFFER FROM'} the hashes the kernel version 190 commit logged "
              "(0f36330a..., 9cdd1cef...)")
        if a.check and not same:
            ok = False
    else:
        same = solver.startswith(E149_V4_AFTER[0]) and agent.startswith(E149_V4_AFTER[1])
        print(f"solver.py and tool_agent.py {'MATCH' if same else 'DIFFER FROM'} the post-patch hashes the kernel version 191 "
              f"commit logged ({E149_V4_AFTER[0]}..., {E149_V4_AFTER[1]}...; 16-digit prefixes)")
        if a.check and not same:
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
