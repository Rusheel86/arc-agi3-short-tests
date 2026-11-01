"""Regenerate the paper figures from data/ and compare them with the published ones in expected/.

  python -I tools/paper_figures/check_figures.py [--against DIR] [--keep DIR]

How files are compared:
  .png        sha256; if the bytes differ, a pixel diff (number of differing pixels, largest channel difference)
  .csv / .md  sha256 of the bytes, and again after normalising line endings
  .svg        matplotlib writes a creation date and random ids into every SVG, so SVGs are compared after removing
              the <dc:date> element and renumbering the generated ids (clip paths, markers) in order of appearance
Exit 0 = every file matches (PNG: byte-identical or 0 differing pixels).
"""
import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
EXPECTED = os.path.join(HERE, "expected")


def sha(b):
    return hashlib.sha256(b).hexdigest()


def svg_norm(b):
    t = b.decode("utf-8").replace("\r\n", "\n")
    t = re.sub(r"<dc:date>.*?</dc:date>", "<dc:date/>", t, flags=re.S)
    ids = {}

    def sub(m):
        return ids.setdefault(m.group(0), f"ID{len(ids)}")
    return re.sub(r"\b[pm][0-9a-f]{10}\b", sub, t).encode("utf-8")


def png_diff(a, b):
    import io
    import numpy as np
    from matplotlib import image
    x = image.imread(io.BytesIO(a), format="png")
    y = image.imread(io.BytesIO(b), format="png")
    if x.shape != y.shape:
        return f"size {x.shape} vs {y.shape}"
    d = np.abs(x.astype(float) - y.astype(float))
    n = int((d.max(axis=-1) > 0).sum()) if d.ndim == 3 else int((d > 0).sum())
    return n, float(d.max())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--against", default=EXPECTED, help="folder with the published figures (default: expected/)")
    ap.add_argument("--keep", default="", help="write the regenerated figures here instead of a temp folder")
    a = ap.parse_args(argv)
    out = a.keep or tempfile.mkdtemp(prefix="figs_")
    os.makedirs(out, exist_ok=True)
    r = subprocess.run([sys.executable, "-I", os.path.join(HERE, "make_paper_figures.py"), "--out", out],
                       capture_output=True, text=True, timeout=600)
    print(r.stdout.strip())
    if r.returncode != 0:
        print("make_paper_figures.py failed:", r.stderr[-1500:])
        return 1
    r = subprocess.run([sys.executable, "-I", os.path.join(HERE, "make_cover_v6.py"), "--out", out],   # the 2026-10-30 cover
                       capture_output=True, text=True, timeout=600)
    print(r.stdout.strip())
    if r.returncode != 0:
        print("make_cover_v6.py failed:", r.stderr[-1500:])
        return 1
    names = sorted(f for f in os.listdir(a.against) if not f.startswith(".") and f.lower() != "readme.md")
    made = sorted(os.listdir(out))
    ok = True
    print(f"\ncomparing {len(names)} published files in {os.path.basename(os.path.normpath(a.against))}/ with the regenerated ones")
    for n in names:
        pa, pb = os.path.join(a.against, n), os.path.join(out, n)
        if not os.path.exists(pb):
            print(f"  MISSING   {n}")
            ok = False
            continue
        A, B = open(pa, "rb").read(), open(pb, "rb").read()
        if n.endswith(".png"):
            if A == B:
                print(f"  IDENTICAL {n} (sha256 {sha(A)[:12]})")
            else:
                res = png_diff(A, B)
                same = isinstance(res, tuple) and res[0] == 0
                print(f"  {'PIXELS=   ' if same else 'DIFFERENT '}{n}: bytes differ; pixel diff {res}")
                ok &= same
        elif n.endswith(".svg"):
            same = svg_norm(A) == svg_norm(B)
            print(f"  {'SAME-SVG  ' if same else 'DIFFERENT '}{n} (after removing the date and renumbering ids)"
                  + (" bytes identical" if A == B else ""))
            ok &= same
        else:
            same = A == B
            same_lf = A.replace(b"\r\n", b"\n") == B.replace(b"\r\n", b"\n")
            print(f"  {'IDENTICAL ' if same else ('SAME-LF   ' if same_lf else 'DIFFERENT ')}{n}"
                  + ("" if same else " (equal after normalising line endings)" if same_lf else ""))
            ok &= same_lf
    extra = [m for m in made if m not in names]
    if extra:
        print("  regenerated but not published:", extra)
    if not a.keep:
        shutil.rmtree(out, ignore_errors=True)
    print("\nRESULT:", "ALL MATCH" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
