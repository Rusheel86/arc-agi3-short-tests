"""cover_v6.png: the Paper Track cover with the paper's new title (skeptic v5 judge review, Decisions, 2026-10-30).

Same data, same palette, same layout and same drawing code as `cover()` in make_paper_figures.py, with two
changes only:
  1. the title is "Short Tests Pick the Wrong Agent" (was "Short tests reward the wrong agent");
  2. a small note "one draw each, different code" under the right panel (the hidden-game draws are single runs of
     different code, so the panel is not a controlled comparison).

Nothing existing is modified: cover.png and every other file in this folder stay as they are. The data (r2) comes from
the original script's own fig2() (E058 benchmark logs, E072 floor, leaderboard draws), run into a throw-away temp
directory, so its asserts against the ledger (+4 / 10 / 2 at 40k, -20 / 0 / 11 at the end) still apply.
As a self-check the script also redraws the ORIGINAL cover in memory and compares it with cover.png: if the data
files changed since cover.png was made, it says so.

RELEASE COPY of the project's reports/paper_figures/make_cover_v6.py (2026-10-30): only the paths changed (it loads this
folder's make_paper_figures.py, compares with expected/cover.png and writes to --out).
Usage: python -I tools/paper_figures/make_cover_v6.py [--out DIR]   (default ./paper_figures_out)
Needs numpy, scipy, matplotlib, Pillow. CPU only, about 25 s. Writes <out>/cover_v6.png only
(the original cover has no SVG, so there is none here either).
"""
import importlib.util
import io
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXPECTED = HERE / "expected"
OUT = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else Path.cwd() / "paper_figures_out"
TITLE_OLD = "Short tests reward the wrong agent"
TITLE_V6 = "Short Tests Pick the Wrong Agent"
NOTE = "one draw each, different code"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


mfp = load("make_paper_figures", HERE / "make_paper_figures.py")
plt, ORANGE, BLUE, SURF, INK, INK2, grid = mfp.plt, mfp.ORANGE, mfp.BLUE, mfp.SURF, mfp.INK, mfp.INK2, mfp.grid


def draw_cover(r2, title, note=None):
    """verbatim copy of mfp.cover() up to the save call; `title` and `note` are the only additions."""
    b, sc, tmax, hv, cross = r2["budgets"], r2["sc"], r2["tmax"], r2["hv"], r2["cross"]
    fig = plt.figure(figsize=(12, 6.3), dpi=100)    # 1200 x 630 px; every text >= 22 pt = 30 px
    fig.text(0.04, 0.94, title, fontsize=42, fontweight="semibold", color=INK, va="top")
    fig.text(0.04, 0.80, "Banning self-check words leads early, then loses: 18 games, 60 min each", fontsize=23,
             color=INK2, va="top")
    ax = fig.add_axes([0.09, 0.17, 0.57, 0.50])
    axh = fig.add_axes([0.765, 0.17, 0.205, 0.50])
    ax.axvspan(0, 40, color=ORANGE, alpha=0.10, lw=0, zorder=0)   # the first 40k tokens, where the ledger reads the lead
    ax.text(20, 47, f"ban\n{r2['d40']:+d} levels", fontsize=22, color=INK2, ha="center", va="top")
    for a, col in (("ctrl", BLUE), ("ban", ORANGE)):
        m = b <= tmax[a]
        ax.plot(b[m] / 1e3, sc[a][m], color=col, lw=5, solid_joinstyle="round", zorder=3)
        ax.plot([tmax[a] / 1e3], [sc[a][m][-1]], "o", ms=16, color=col, mec=SURF, mew=2.5, zorder=4)
    ax.text(tmax["ctrl"] / 1e3 - 4, sc["ctrl"][b <= tmax["ctrl"]][-1] + 2.5, "no ban", fontsize=26,
            fontweight="semibold", ha="right", va="bottom")
    ax.text(tmax["ban"] / 1e3, sc["ban"][b <= tmax["ban"]][-1] - 3.5, "ban", fontsize=26, fontweight="semibold",
            ha="center", va="top")
    ax.set_xlim(0, 225)
    ax.set_ylim(0, 52)
    ax.set_xticks([0, 100, 200])
    ax.set_xticklabels(["0", "100k", "200k"], fontsize=22)
    ax.set_yticks([0, 25, 50])
    ax.set_yticklabels(["0", "25", "50"], fontsize=22)
    ax.tick_params(axis="x", length=6, width=1.2)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(1.4)
    grid(ax, "y")
    ax.set_xlabel("tokens per game", fontsize=22, color=INK2, labelpad=4)
    ax.set_ylabel("score", fontsize=22, color=INK2)
    for x, (lab, v, c) in zip((0, 1.2, 1.8), hv):
        axh.plot([x], [v], "o", ms=20, color=c, mec=SURF, mew=2.5, zorder=3)
    axh.text(0, hv[0][1] + 2.4, f"{hv[0][1]:.0f}", fontsize=24, ha="center", va="bottom", fontweight="semibold")
    axh.text(1.5, max(v for _, v, _ in hv[1:]) + 2.4, f"{hv[1][1]:.0f}, {hv[2][1]:.0f}", fontsize=24, ha="center",
             va="bottom", fontweight="semibold")
    axh.set_xlim(-0.6, 2.4)
    axh.set_ylim(0, 30)
    axh.set_xticks([0, 1.5])
    axh.set_xticklabels(["ban", "no ban"], fontsize=22)
    axh.tick_params(axis="x", length=0)
    axh.set_yticks([0, 10, 20, 30])
    axh.set_yticklabels(["0", "10", "20", "30"], fontsize=22)
    for s in ("left", "bottom"):
        axh.spines[s].set_linewidth(1.4)
    grid(axh, "y")
    axh.set_title("hidden games", fontsize=22, color=INK2, loc="left", pad=10)
    if note:   # v6 addition: right-aligned to the right panel's edge (x = 0.97), below its tick labels
        fig.text(0.97, 0.072, note, fontsize=22, color=INK2, ha="right", va="top", style="italic")
    return fig


def to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    return buf.getvalue()


def main():
    from PIL import Image
    with tempfile.TemporaryDirectory() as tmp:       # fig2() writes its own figure and CSV; keep them out of the repo
        r2 = mfp.fig2(Path(tmp))
    # self-check: the unchanged drawing code reproduces the existing cover.png from today's data files
    again = Image.open(io.BytesIO(to_png_bytes(draw_cover(r2, TITLE_OLD)))).convert("RGB")
    orig = Image.open(EXPECTED / "cover.png").convert("RGB")
    if again.size == orig.size:
        n = int((np.asarray(again).astype(int) != np.asarray(orig).astype(int)).any(axis=2).sum())
        print(f"self-check: original cover redrawn from current data, {n} of {again.size[0] * again.size[1]} pixels "
              f"differ from cover.png ({'identical' if n == 0 else 'DATA OR FONTS CHANGED SINCE cover.png'})")
    else:
        print("self-check: size mismatch", again.size, orig.size)
    data = to_png_bytes(draw_cover(r2, TITLE_V6, NOTE))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cover_v6.png").write_bytes(data)
    new = Image.open(io.BytesIO(data)).convert("RGB")
    diff = (np.asarray(new).astype(int) != np.asarray(orig).astype(int)).any(axis=2)
    ys, xs = np.nonzero(diff)
    print(f"wrote {OUT / 'cover_v6.png'} {new.size}; differs from cover.png in y {ys.min()}-{ys.max()}, x {xs.min()}-{xs.max()}"
          f" (title row and note row only: title y < 100, note y > 575)")
    mid = diff[100:575]
    print("pixels differing between y=100 and y=575:", int(mid.sum()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
