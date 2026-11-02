"""cover_v7.png: cover_v6.png plus one annotation (skeptic judge review of writeup v10, item 8, accepted 2026-11-02).

The v10 review found that the cover's score axis shows no visible early lead (the two score curves overlap in the first
40k tokens). v7 adds one text block next to the shaded 40k band, "10 games ahead, / 2 behind at 40k", in the cover's own
font size (22 pt) and secondary ink. The numbers are read from fig2()'s data (r2["a40"], r2["z40"]), whose assert
already ties them to the ledger (E058: +4 levels, 10 games ahead, 2 behind at 40k; ledger l.3212).

Everything else is make_cover_v6.py's draw_cover() unchanged (same data, palette, layout, title and note). Nothing
existing is modified: cover.png and cover_v6.png stay as they are.

Usage (repo root): ./.venv/Scripts/python.exe reports/paper_figures/make_cover_v7.py
CPU only, about 25 s. Writes reports/paper_figures/cover_v7.png only.
"""
import importlib.util
import io
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "paper_figures"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


mc6 = load("make_cover_v6", OUT / "make_cover_v6.py")


def draw_cover_v7(r2):
    fig = mc6.draw_cover(r2, mc6.TITLE_V6, mc6.NOTE)
    ax = fig.axes[0]   # the score panel (added first in draw_cover)
    # right of the 0-40k band, above both curves (the control is below 32 before 110k)
    ax.text(44, 50, f"{r2['a40']} games ahead,\n{r2['z40']} behind at 40k", fontsize=22, color=mc6.INK2,
            ha="left", va="top", linespacing=1.15, zorder=5)
    return fig


def main():
    from PIL import Image
    with tempfile.TemporaryDirectory() as tmp:
        r2 = mc6.mfp.fig2(Path(tmp))
    assert (r2["d40"], r2["a40"], r2["z40"]) == (4, 10, 2)
    # self-check: v6's drawing code still reproduces cover_v6.png from today's data
    again = Image.open(io.BytesIO(mc6.to_png_bytes(mc6.draw_cover(r2, mc6.TITLE_V6, mc6.NOTE)))).convert("RGB")
    v6 = Image.open(OUT / "cover_v6.png").convert("RGB")
    n = int((np.asarray(again).astype(int) != np.asarray(v6).astype(int)).any(axis=2).sum())
    print(f"self-check: cover_v6 redrawn from current data, {n} pixels differ from cover_v6.png")
    data = mc6.to_png_bytes(draw_cover_v7(r2))
    (OUT / "cover_v7.png").write_bytes(data)
    new = Image.open(io.BytesIO(data)).convert("RGB")
    diff = (np.asarray(new).astype(int) != np.asarray(v6).astype(int)).any(axis=2)
    ys, xs = np.nonzero(diff)
    print(f"wrote {OUT / 'cover_v7.png'} {new.size}; differs from cover_v6.png in y {ys.min()}-{ys.max()}, "
          f"x {xs.min()}-{xs.max()} ({int(diff.sum())} pixels; the annotation only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
