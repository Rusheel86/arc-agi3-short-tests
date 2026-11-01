"""Check that arc3-e154-keepalive.ipynb is the pushed E154 notebook plus one licence header cell, and nothing else.

  python -I notebook/verify_notebook.py

It removes the first cell (the MIT-0 header), writes the notebook back in the format Kaggle's push used (JSON, indent 1,
UTF-8, one trailing newline) and compares the sha256 with the file pushed for kernel version 189
(6cf982cd..., 989,125 bytes). It also checks every remaining cell against original_cells.json. Exit 0 = identical.
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(HERE, "arc3-e154-keepalive.ipynb")
MANIFEST = os.path.join(HERE, "original_cells.json")
HEADER_START = "# ARC-AGI-3 entry E154"


def src_of(c):
    return "".join(c["source"]) if isinstance(c["source"], list) else c["source"]


def main():
    man = json.load(open(MANIFEST, encoding="utf-8"))
    nb = json.load(open(NB, encoding="utf-8"))
    ok = True
    first = nb["cells"][0]
    if first["cell_type"] != "markdown" or not src_of(first).startswith(HEADER_START):
        print("FAIL: the first cell is not the licence header")
        return 1
    print(f"header cell: {len(src_of(first)):,} characters, markdown")
    rest = dict(nb, cells=nb["cells"][1:])
    raw = (json.dumps(rest, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
    got = hashlib.sha256(raw).hexdigest()
    same = got == man["original_file_sha256"] and len(raw) == man["original_file_bytes"]
    print(f"without the header: {len(raw):,} bytes, sha256 {got[:16]}... "
          f"{'= ' if same else '!= '}the pushed file ({man['original_file_bytes']:,} bytes, {man['original_file_sha256'][:16]}...)")
    ok &= same
    cells = rest["cells"]
    if len(cells) != man["n_cells"]:
        print(f"FAIL: {len(cells)} cells, the manifest lists {man['n_cells']}")
        return 1
    bad = [m["index"] for m, c in zip(man["cells"], cells)
           if hashlib.sha256(src_of(c).encode("utf-8")).hexdigest() != m["sha256"] or c["cell_type"] != m["type"]]
    print(f"cells: {len(cells) - len(bad)} of {len(cells)} match the manifest" + (f"; differ: {bad}" if bad else ""))
    ok &= not bad
    outputs = sum(1 for c in cells if c.get("outputs"))
    print(f"cells with stored outputs: {outputs} (the pushed notebook had none)")
    ok &= outputs == 0
    size = os.path.getsize(NB)
    print(f"release notebook: {size:,} bytes ({'under' if size < 1_000_000 else 'OVER'} Kaggle's 1,000,000-byte source limit)")
    ok &= size < 1_000_000
    print("RESULT:", "IDENTICAL apart from the header cell" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
