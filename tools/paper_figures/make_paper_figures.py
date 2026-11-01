"""Paper Track figures from the data files in data/ (release copy of the project's analysis/make_paper_figures.py).

Every number drawn is read from a data file (benchmark logs, summary JSONs, the E138 submission tables, run-notebook
output lines, result texts); the few ledger sentences that are the only record of a number are parsed from
data/ledger_excerpts.json, never retyped. One assumption is a stated guess: DECODE_HOURS (Figure 1 and the cover).

RELEASE COPY. The plotting code below is the project's script of 2026-10-27 (commit 25e3f9b), unchanged. Only the
inputs moved: data/ holds the exact lines, fields and rows the original read from the project (slim copies; see
README.md), frozen at that commit, so the figures regenerate as published. The helpers sd / pooled / mde / n_balanced
and OLD_SD / ALPHA1 are the project's leaderboard_noise.py ones, inlined (same formulas as tools/noise_floor).

Outputs (--out, default ./paper_figures_out): fig1..fig5 as PNG (300 dpi) + SVG, cover.png (1200 x 630), one CSV per figure with the
plotted numbers, captions.md (one caption per figure, <= 40 words, every number tagged with its E-id).
UPDATE 2026-10-27 (paper draft v4 + skeptic v3 review): cover.png is now fig2 reduced (the ban crossing, all measured);
the fig1-based cover is kept as cover_alt.png; fig1 is relabelled to the narrowed claim and renamed
fig1_half_tokens_did_not_cost; captions.md lists the ledger line of every number that has no data file.

Usage: python -I tools/paper_figures/make_paper_figures.py [--out DIR]; check: python -I tools/paper_figures/check_figures.py
Needs numpy, scipy, matplotlib. CPU only, read-only on every input, ~20 s. Text uses Segoe UI, else DejaVu Sans
(a machine without Segoe UI draws the same numbers in a different font, so the PNGs then differ in pixels).

Colours: the dataviz skill's reference palette (slot 1 blue, slot 2 orange; blue ordinal ramp), validated with its
validate_palette.js against the white page surface (categorical: all checks pass; ordinal ramps of 3, 4 and 5 steps pass).
"""
import argparse
import csv
import importlib.util
import json
import math
import re
import statistics
import sys
from pathlib import Path

import logging

import numpy as np
import matplotlib

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)   # fallback fonts lack 'semibold'; harmless
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
if str(HERE) not in sys.path:     # -I leaves the script's folder off sys.path; game_curve.py is our own code
    sys.path.insert(0, str(HERE))
import game_curve as kk  # noqa: E402  (game_curve: the E058 / T3 scoring rule, copied unchanged from k_killtests.py)
from scipy import stats as _stats  # noqa: E402


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _LBN:
    """the parts of the project's analysis/leaderboard_noise.py (E138) this script uses; SUBS = data/agi3_draws.csv"""
    OLD_SD = 3.9           # the SD the project used before E138 (other people's copies of the baseline)
    ALPHA1 = 0.05          # one-sided
    POWER = 0.90

    def __init__(self, path, versions=None):
        with open(path, newline="", encoding="utf-8") as f:
            self.SUBS = [(int(r["ref"]), r["date_utc"], float(r["score"]), int(r["script_version"]), "")
                         for r in csv.DictReader(f)]
        self.VERSIONS = versions

    @staticmethod
    def sd(x):
        return float(np.std(x, ddof=1)) if len(x) > 1 else float("nan")

    @staticmethod
    def pooled(groups):
        ss = sum(float(np.sum((np.array(g) - np.mean(g)) ** 2)) for g in groups)
        df = sum(len(g) - 1 for g in groups)
        return math.sqrt(ss / df), df, ss

    @staticmethod
    def z(a):
        return float(_stats.norm.ppf(a))

    def n_balanced(self, delta, sigma, alpha, power=POWER):
        zz = self.z(1 - alpha) + self.z(power)
        return 2 * sigma ** 2 * zz ** 2 / delta ** 2

    def mde(self, sigma, nc, n0, alpha=ALPHA1, power=POWER):
        return (self.z(1 - alpha) + self.z(power)) * sigma * math.sqrt(1.0 / nc + 1.0 / n0)


LBN = _LBN(DATA / "agi3_draws.csv")                                                     # E138, ARC-AGI-3 draws (10-27)
LBN2 = _LBN(DATA / "agi2_draws.csv",                                                    # E138 addendum, ARC-AGI-2
            {int(k): v for k, v in json.loads((DATA / "agi2_versions.json").read_text(encoding="utf-8")).items()})

# ---------------------------------------------------------------------------------------------------- inputs
F = dict(   # release: data/ holds the lines and fields the original read from these project files (named in README.md)
    ledger=DATA / "ledger_excerpts.json",                       # experiments/ledger.md (3 lines)
    e058_ctrl=DATA / "e058_arm_long_control_benchmark.json",    # ARC-AGI3/screen_e058/arm_long_control/benchmark.json
    e058_ban=DATA / "e058_arm_long_nohmm_benchmark.json",       # ARC-AGI3/screen_e058/arm_long_nohmm/benchmark.json
    t3=DATA / "t3_recheck_results_excerpt.txt",                 # analysis/t3_recheck_results.txt
    e081_nb=DATA / "e081_run_excerpt.txt",                      # ARC-AGI3/arc-agi-3-e081_run.ipynb
    e137=DATA / "e137_summary_subset.json",                     # ARC-AGI3/e137_out/e137/summary.json
    t2a=DATA / "t2a_result_excerpt.md",                         # analysis/dflash2/t2a_result.md
    e073_nb=DATA / "e073_run_excerpt.txt",                      # ARC-AGI3/arc-agi-3-e073_run.ipynb
    pools=DATA / "throughput_arithmetic_excerpt.txt",           # analysis/throughput_arithmetic_results.txt
    e142=DATA / "agi2" / "e142_decision.json",                  # ARC-AGI2/e142_out_v2/e142/summary.json
    e151=DATA / "agi2" / "e151_decision.json",                  # ARC-AGI2/e151_out/e151/summary.json
    e152=DATA / "agi2" / "e152_c_rescuable.json",               # ARC-AGI2/e152_out_a/e152/summary.json
    v6_verdict=DATA / "agi2" / "v6_smoke_verdict_subset.json",  # ARC-AGI2/v6_out/v6/smoke_verdict.json
    v6_selection=DATA / "agi2" / "v6_selection_subset.json",    # ARC-AGI2/v6_out/v6/selection.json
)

DECODE_HOURS = 8.5   # [guess] decoding time of the 9-hour scoring run: 9 h minus ~30 min server start (E058: 33 min, E154: 33 min)
N_HIDDEN = 110       # games in the scoring run (CLAUDE.md hard constraints)

# ---------------------------------------------------------------------------------------------------- style
SURF, INK, INK2, MUTED, GRID, AXIS = "#ffffff", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"                         # categorical slots 1, 2
RAMP3 = ["#86b6ef", "#2a78d6", "#104281"]                   # ordinal blue 250 / 450 / 650
RAMP5 = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]  # ordinal blue 250 .. 650
WIDTH = 7.0          # inches; a Kaggle writeup column shows it at ~750 px, so 8.5 pt text renders at ~12-13 px

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.major.size": 3, "ytick.major.size": 0, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "svg.fonttype": "path", "axes.unicode_minus": True, "legend.frameon": False, "legend.fontsize": 8.5,
})
LW, MS, RING = 1.6, 6.5, 1.2   # line width (pt), marker diameter (pt), surface ring (pt)


def grid(ax, axis="y"):
    ax.grid(True, axis=axis, color=GRID, linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)


def ptitle(ax, text, x=0.0):
    ax.set_title(text, loc="left", x=x, fontweight="semibold", color=INK, pad=6)


def figtitle(fig, text, y=0.985):
    fig.text(0.012, y, text, fontsize=11.5, fontweight="semibold", color=INK, va="top", ha="left")


def per_inch(ax):
    """data units per inch along x and y (axes position must be final)."""
    bb = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    return abs(x1 - x0) / bb.width, abs(y1 - y0) / bb.height


def hbar(ax, y, x0, w, h, color, round_end=True, r_in=0.03, z=3):
    """horizontal bar from x0 to x0 + w, height h (data units), 4px-style rounded data end, square at the baseline."""
    if w <= 0:
        return
    xpi, ypi = per_inch(ax)
    rx, ry = min(r_in * xpi, w), min(r_in * ypi, h / 2)
    x1, yb, yt = x0 + w, y - h / 2, y + h / 2
    if not round_end:
        rx = ry = 0.0
    verts = [(x0, yb), (x1 - rx, yb), (x1, yb), (x1, yb + ry), (x1, yt - ry), (x1, yt), (x1 - rx, yt), (x0, yt), (x0, yb)]
    codes = [MPath.MOVETO, MPath.LINETO, MPath.CURVE3, MPath.CURVE3, MPath.LINETO, MPath.CURVE3, MPath.CURVE3,
             MPath.LINETO, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, edgecolor="none", zorder=z))


def h_in_data(ax, inches):
    return inches * per_inch(ax)[1]


def save(fig, out, stem):
    fig.savefig(out / f"{stem}.png", dpi=300)
    fig.savefig(out / f"{stem}.svg")
    plt.close(fig)


def write_csv(out, stem, header, rows):
    with open(out / f"{stem}_data.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def words(s):
    """the writeup's own count: whitespace-separated tokens that contain a letter or digit."""
    return sum(1 for t in s.split() if re.search(r"[A-Za-z0-9]", t))


def k(x):
    return f"{x / 1e3:.0f}k"


# ---------------------------------------------------------------------------------------------------- data readers
def _ledger_lines():
    return json.loads(F["ledger"].read_text(encoding="utf-8"))["lines"]


def ledger_text():
    return "\n".join(x["text"] for x in _ledger_lines())


def agi3_draws():
    """every ARC-AGI-3 submission of E138's table, plus draws recorded in the ledger after the table's last update."""
    rows = [dict(ref=r, date=d, score=s, sv=sv) for r, d, s, sv, _ in LBN.SUBS]
    have = {r["ref"] for r in rows}
    led = ledger_text()
    if 57669233 not in have:   # E149 v3 half-clock draw 1 (kernel version 190), ledger E149 "Result, board draw 1"
        m = re.search(r"ref 57669233[^\]]*?submitted (\d\d:\d\d) UTC\]: \*\*([0-9.]+)\*\*", led)
        assert m, "E149 draw 1 not found in the ledger"
        rows.append(dict(ref=57669233, date="2026-10-26 " + m.group(1), score=float(m.group(2)), sv="E149v3"))
    return rows


def e058_curves(budgets):
    out = {}
    for arm in ("ctrl", "ban"):
        runs = kk.load_runs(F[f"e058_{arm}"])
        cv = {g["game_id"][:4]: kk.game_curve(g) for g in runs}
        out[arm] = cv
    assert sorted(out["ctrl"]) == sorted(out["ban"]) and len(out["ctrl"]) == 18
    return out


def t3_elasticity():
    s = F["t3"].read_text(encoding="utf-8")
    pt = float(re.search(r"140->180k: E058 [0-9.]+, new [0-9.]+, POOLED ([0-9.]+)", s).group(1))
    lo, hi = map(float, re.search(r"game-cluster bootstrap 90%: 140->180k ([0-9.]+)\.\.([0-9.]+)", s).group(1, 2))
    floor = int(re.search(r"noise floor for E061 = max\(\d+, \d+\) = (\d+) levels", s).group(1))
    return pt, lo, hi, floor


def e081_tok_s():
    s = F["e081_nb"].read_text(encoding="utf-8")
    m = re.search(r"generated ([0-9.]+) tok/s over ([0-9.]+) min \(minute 3 to the end\)", s)
    return float(m.group(1))


def e137_tok_s():
    d = json.loads(F["e137"].read_text(encoding="utf-8"))
    return d["arms"]["mtp"]["agg_tok_s"], d["arms"]["mtp_repeat"]["agg_tok_s"], d["cfg"]["in_flight"]


def t2a_rows():
    s = F["t2a"].read_text(encoding="utf-8")
    hdr = next(l for l in s.splitlines() if l.startswith("| | MTP"))
    names = [c.strip() for c in hdr.strip("|").split("|")][1:]
    blk = next(l for l in s.splitlines() if l.startswith("| blocks per 163,840-token request"))
    fit = next(l for l in s.splitlines() if l.startswith("| **full-window fit, E098's rule**"))
    rows = []
    for name, bc, fc in zip(names, blk.strip("|").split("|")[1:], fit.strip("|").split("|")[1:]):
        total = int(re.match(r"\s*(\d+)\s*=", bc).group(1))
        drafter = re.search(r"\*\*(\d+)\*\*", bc)
        drafter = int(drafter.group(1)) if drafter else 0
        rows.append(dict(name=name, total=total, drafter=drafter, model=total - drafter,
                         fit=int(re.search(r"\*\*(\d+)\*\*", fc).group(1))))
    return rows


def e073_numbers():
    s = F["e073_nb"].read_text(encoding="utf-8")
    sp = dict(re.findall(r"(control|prune384)\s+all_busy:\s+([0-9.]+) generated tok/s", s))
    p = F["pools"].read_text(encoding="utf-8")
    un = re.search(r"unpruned util 0\.98 \(E058/E061b mtp3\)\s+([0-9.]+) GiB ->\s+(\d+) blocks of (\d+)", p)
    pr = re.search(r"prune384 util 0\.98 \(E073/E081, 24 seqs\)\s+([0-9.]+) GiB ->\s+(\d+) blocks of (\d+)", p)
    g = re.search(r"so (\d+) games fit where the unpruned pool holds (\d+)", ledger_text())
    return dict(speed_ctrl=float(sp["control"]), speed_prune=float(sp["prune384"]),
                pool_ctrl=int(un.group(2)), pool_prune=int(pr.group(2)), block=int(un.group(3)),
                games_prune=int(g.group(1)), games_ctrl=int(g.group(2)))


def agi2_funnels():
    e142 = json.loads(F["e142"].read_text(encoding="utf-8"))["decision"]
    e151 = json.loads(F["e151"].read_text(encoding="utf-8"))["decision"]
    e152 = json.loads(F["e152"].read_text(encoding="utf-8"))["c_rescuable"]
    v6 = json.loads(F["v6_verdict"].read_text(encoding="utf-8"))
    sel = json.loads(F["v6_selection"].read_text(encoding="utf-8"))["rows"]
    assert e151["H"] + e151["K"] == e142["n_rescuable"]
    a = [("Test outputs scored (4B, 4 views)", e142["n_units"], "E142"),
         ("Missed by the 4B model", e142["n_miss"], "E142"),
         ("Rescuable: answer within 1-3 cell switches", e142["n_rescuable"], "E142"),
         (f"Reachable in 16 views, not already found ({e151['H']} were)", e151["K"], "E151"),
         ("Run with the real decoder (CPU)", e152["completed"], "E152"),
         ("Truth found by the decoder", e152["truth_in_lds"], "E152")]
    n_tasks = len({r[0] for r in v6["numbers"]["per_output"]})
    b = [("Labelled public tasks run (smoke)", n_tasks, "E150"),
         ("Tasks gated to search after TTT", v6["n_p2_tasks"], "E150"),
         ("Complete contests (test outputs)", sum(1 for r in sel if r.get("contest")), "E150"),
         ("New correct answers (v6 right, v4 wrong)", v6["numbers"]["v6_only"], "E150")]
    overlap = len(set(e151["states"]) & set(v6["gated_tasks"]))
    return a, b, dict(hits=e151["H"], net=v6["numbers"]["net"], overlap=overlap, n_resc=len(e151["states"]),
                      n_gated=len(v6["gated_tasks"]))


# ---------------------------------------------------------------------------------------------------- figure 1
def fig1_data():
    draws = agi3_draws()
    e081 = [d["score"] for d in draws if d["sv"] == 357441618]
    e098 = [d["score"] for d in draws if d["sv"] in (357974936, 358364499)]
    assert len(e081) == 1 and len(e098) == 6
    t81 = e081_tok_s()
    t98a, t98b, inflight = e137_tok_s()
    el, el_lo, el_hi, _ = t3_elasticity()
    secs = DECODE_HOURS * 3600
    x81 = t81 * secs / N_HIDDEN
    x98lo, x98hi = t98a * secs / N_HIDDEN, t98b * secs / N_HIDDEN
    x98 = (x98lo + x98hi) / 2
    m98, sd98 = statistics.mean(e098), statistics.stdev(e098)
    pred = lambda x, e: e081[0] * (x / x81) ** e  # noqa: E731
    return dict(e081=e081[0], e098=e098, m98=m98, se98=sd98 / math.sqrt(len(e098)), t81=t81, t98=(t98a, t98b),
                inflight=inflight, x81=x81, x98=x98, x98lo=x98lo, x98hi=x98hi, el=el, el_lo=el_lo, el_hi=el_hi,
                pred=pred, p98=pred(x98, el), p98_lo=pred(x98, el_lo), p98_hi=pred(x98, el_hi))


def zero_above(d, x, knee=180e3):
    """[computed] alternative: no returns above the measured range's top (180k), elasticity 0.59 below it."""
    x = np.asarray(x, dtype=float)
    return np.where(x >= knee, d["e081"], d["e081"] * (np.minimum(x, knee) / knee) ** d["el"])


def fig1(out, d):
    z98 = float(zero_above(d, d["x98"]))
    fig = plt.figure(figsize=(WIDTH, 4.5))
    ax = fig.add_axes([0.10, 0.13, 0.86, 0.69])
    figtitle(fig, "Half the tokens did not cost score")
    fig.text(0.012, 0.925, "Hidden-set score against tokens per game [computed]; dashed and dotted lines are extrapolations, "
             "not fits", fontsize=9, color=INK2, va="top")
    fig.text(0.012, 0.885, "What bought the +%.1f is not isolated: reply cap, window, timeout fix and games in flight all "
             "changed at once" % (d["m98"] - d["e081"]), fontsize=9, color=INK2, va="top")
    ax.set_xscale("log")
    ax.set_xlim(108e3, 380e3)
    ax.set_ylim(0, 37)
    ticks = [125e3, 150e3, 200e3, 250e3, 300e3, 350e3]
    ax.set_xticks(ticks)
    ax.set_xticklabels([k(t) for t in ticks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_yticks([0, 10, 20, 30])
    grid(ax, "y")
    ax.set_xlabel(f"Tokens per game in the 9-hour run [computed: tok/s × {DECODE_HOURS:g} h ÷ {N_HIDDEN} games; log scale]")
    ax.set_ylabel("Leaderboard score (55 hidden games)")
    xs = np.geomspace(d["x98"], d["x81"], 80)
    ax.fill_between(xs, d["pred"](xs, d["el_hi"]), d["pred"](xs, d["el_lo"]), color=MUTED, alpha=0.13, lw=0, zorder=1)
    ax.plot(xs, d["pred"](xs, d["el"]), color=MUTED, lw=LW, ls=(0, (4, 3)), zorder=2)
    ax.plot(xs, zero_above(d, xs), color=MUTED, lw=LW, ls=(0, (1, 2.2)), zorder=2)
    ax.plot([d["x98"]], [d["p98"]], "o", ms=MS, mfc=SURF, mec=MUTED, mew=1.4, zorder=4)
    ax.plot([d["x98"]], [z98], "o", ms=MS, mfc=SURF, mec=MUTED, mew=1.4, zorder=4)
    ax.text(d["x98"] * 1.035, d["p98"] - 1.2,
            f"{d['p98']:.1f}: constant-elasticity extrapolation ({d['el']:.2f}, band {d['el_lo']:.2f}-{d['el_hi']:.2f})\n"
            "beyond the measured 140-180k range (public games); dashed, not a fit", fontsize=8.5, color=INK2, va="top")
    ax.text(178e3, float(zero_above(d, 200e3)) + 0.7, f"dotted: no returns\nabove 180k → {z98:.1f}", fontsize=8.5,
            color=INK2, va="bottom", ha="left")
    off = np.linspace(-0.012, 0.012, len(d["e098"]))
    ax.plot(d["x98"] * (1 + off), d["e098"], "o", ms=4.2, color=BLUE, alpha=0.35, mew=0, zorder=3)
    ax.plot([d["x98lo"], d["x98hi"]], [d["m98"]] * 2, color=BLUE, lw=LW, solid_capstyle="round", zorder=4)
    ax.errorbar([d["x98"]], [d["m98"]], yerr=[d["se98"]], fmt="o", ms=MS + 1, color=BLUE, mec=SURF, mew=RING,
                elinewidth=LW, capsize=0, zorder=5)
    ax.text(d["x98lo"] * 0.99, max(d["e098"]) + 0.9, f"E098: {d['m98']:.2f}, {len(d['e098'])} draws (mean ± SE)\n"
            f"{d['inflight']} games, 160k window, 32k replies;\n{d['t98'][0]:.0f}-{d['t98'][1]:.0f} tok/s",
            fontsize=8.5, color=INK, ha="left", va="bottom")
    ax.errorbar([d["x81"]], [d["e081"]], fmt="o", ms=MS + 1, color=BLUE, mec=SURF, mew=RING, zorder=5)
    ax.text(d["x81"] * 0.985, d["e081"] + 1.6, f"E081: {d['e081']:.2f}, 1 draw\n24 games, 92k window,\n"
            f"12k replies; {d['t81']:,.0f} tok/s", fontsize=8.5, color=INK, ha="right", va="bottom")
    xb = d["x98lo"] * 0.975
    ax.plot([xb, xb], [d["p98"], d["m98"]], color=INK2, lw=1.0)
    for yy in (d["m98"], d["p98"]):
        ax.plot([xb * 0.99, xb * 1.01], [yy, yy], color=INK2, lw=1.0)
    ax.text(xb * 0.985, (d["p98"] + d["m98"]) / 2, f"+{d['m98'] - d['p98']:.1f} above the\nextrapolation;\ncause not isolated",
            fontsize=8.5, color=INK2, ha="right", va="center")
    save(fig, out, "fig1_half_tokens_did_not_cost")
    write_csv(out, "fig1", ["what", "value", "label", "source"], [
        ["E081 score (1 draw)", d["e081"], "measured", "E081 via analysis/leaderboard_noise.py SUBS"],
        *[[f"E098 draw {i + 1}", v, "measured", "E098 via analysis/leaderboard_noise.py SUBS"] for i, v in enumerate(d["e098"])],
        ["E098 mean", round(d["m98"], 4), "computed", "E138"], ["E098 SE", round(d["se98"], 4), "computed", "E138"],
        ["E081 tok/s (60-min screen, 24 games)", d["t81"], "measured", "E081 run notebook"],
        ["E098 tok/s bench (MTP, repeat)", f"{d['t98'][0]};{d['t98'][1]}", "measured", "E137 summary.json"],
        ["decode hours of the scoring run", DECODE_HOURS, "guess", "9 h minus ~30 min server start"],
        ["E081 tokens per game", round(d["x81"]), "computed", "tok/s x hours / 110"],
        ["E098 tokens per game (lo, mid, hi)", f"{d['x98lo']:.0f};{d['x98']:.0f};{d['x98hi']:.0f}", "computed", "same"],
        ["elasticity 140-180k (90% band)", f"{d['el']};{d['el_lo']};{d['el_hi']}", "measured", "E072 t3_recheck_results.txt"],
        ["constant-elasticity extrapolation at E098 (band)", f"{d['p98']:.2f};{d['p98_hi']:.2f};{d['p98_lo']:.2f}",
         "computed", "E081 score x (x98/x81)^e; extrapolated beyond 140-180k; independent of the decode-hours guess"],
        ["no returns above 180k, elasticity below", round(z98, 2), "computed", "E081 score x (x98/180k)^0.59"]])
    return z98


def cover_alt(out, d):
    """the v3 cover (Figure 1 reduced), kept as an alternative; the title no longer claims the thesis."""
    fig = plt.figure(figsize=(12, 6.3), dpi=100)    # 1200 x 630 px; every text >= 22 pt = 30 px
    fig.text(0.04, 0.93, "Half the tokens did not cost score", fontsize=44, fontweight="semibold", color=INK, va="top")
    fig.text(0.04, 0.78, "Hidden-set score, 6 draws vs 1; what bought the gain is not isolated", fontsize=24, color=INK2,
             va="top")
    ax = fig.add_axes([0.10, 0.17, 0.86, 0.50])
    ax.set_xscale("log")
    ax.set_xlim(120e3, 380e3)
    ax.set_ylim(8, 34)
    ax.set_xticks([150e3, 300e3])
    ax.set_xticklabels(["150k", "300k"], fontsize=22)
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_yticks([10, 20, 30])
    ax.set_yticklabels(["10", "20", "30"], fontsize=22)
    ax.tick_params(axis="x", length=6, width=1.2)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(1.4)
    grid(ax, "y")
    ax.set_xlabel("tokens per game [computed]", fontsize=22, color=INK2, labelpad=4)
    ax.set_ylabel("hidden-set score", fontsize=22, color=INK2)
    xs = np.geomspace(d["x98"], d["x81"], 80)
    ax.fill_between(xs, d["pred"](xs, d["el_hi"]), d["pred"](xs, d["el_lo"]), color=MUTED, alpha=0.14, lw=0)
    ax.plot(xs, d["pred"](xs, d["el"]), color=MUTED, lw=4, ls=(0, (3, 2)))
    ax.plot([d["x98"]], [d["p98"]], "o", ms=20, mfc=SURF, mec=MUTED, mew=3.5, zorder=4)
    ax.text(d["x98"] * 1.035, d["p98"] - 3.4, f"extrapolated: {d['p98']:.0f}", fontsize=24, color=INK2, va="center")
    ax.plot([d["x98"]], [d["m98"]], "o", ms=26, color=BLUE, mec=SURF, mew=3, zorder=5)
    ax.plot([d["x81"]], [d["e081"]], "o", ms=26, color=BLUE, mec=SURF, mew=3, zorder=5)
    ax.text(d["x98"] * 1.06, d["m98"] + 0.6, f"E098 {d['m98']:.2f} (6 draws)", fontsize=26, fontweight="semibold",
            va="center")
    ax.text(d["x81"] * 0.94, d["e081"] + 3.2, f"E081 {d['e081']:.2f} (1 draw)", fontsize=26, fontweight="semibold",
            ha="right", va="center")
    fig.savefig(out / "cover_alt.png", dpi=100)
    plt.close(fig)


def cover(out, r2):
    """the cover: Figure 2 (the ban crossing) reduced. Every number on it is measured (E058, E055, E040, E061)."""
    b, sc, tmax, hv, cross = r2["budgets"], r2["sc"], r2["tmax"], r2["hv"], r2["cross"]
    fig = plt.figure(figsize=(12, 6.3), dpi=100)    # 1200 x 630 px; every text >= 22 pt = 30 px
    fig.text(0.04, 0.94, "Short tests reward the wrong agent", fontsize=42, fontweight="semibold", color=INK, va="top")
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
    fig.savefig(out / "cover.png", dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------- figure 2
def fig2(out):
    budgets = np.arange(0, 222e3, 1e3)
    cv = e058_curves(budgets)
    _, _, _, floor = t3_elasticity()
    games = sorted(cv["ctrl"])
    tmax = {a: max(cv[a][g][2] for g in games) for a in cv}
    sc = {a: np.array([statistics.mean(cv[a][g][0](b) for g in games) for b in budgets]) for a in cv}
    lv = {a: np.array([sum(cv[a][g][1](b) for g in games) for b in budgets]) for a in cv}
    diff = lv["ban"] - lv["ctrl"]

    def ahead_behind(b):
        a = sum(1 for g in games if cv["ban"][g][1](b) > cv["ctrl"][g][1](b))
        z = sum(1 for g in games if cv["ban"][g][1](b) < cv["ctrl"][g][1](b))
        return a, z
    b40 = 40e3
    i40 = int(b40 / 1e3)
    a40, z40 = ahead_behind(b40)
    aend, zend = ahead_behind(1e18)
    lend = (sum(cv["ban"][g][3] for g in games), sum(cv["ctrl"][g][3] for g in games))
    assert (diff[i40], a40, z40) == (4, 10, 2) and (lend[0] - lend[1], aend, zend) == (-20, 0, 11), \
        "E058 numbers no longer match the ledger's (+4 / 10 / 2 at 40k; -20 / 0 / 11 at the end)"
    draws = agi3_draws()
    hid = [("Ban\nE055", 356099442, ORANGE), ("No ban\nE040", 355734881, BLUE), ("No ban\nE061", 357018543, BLUE)]
    hv = [(lab, next(x["score"] for x in draws if x["sv"] == sv), c) for lab, sv, c in hid]

    fig = plt.figure(figsize=(WIDTH, 4.9))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[3.0, 1.3], height_ratios=[2.15, 1], left=0.09, right=0.985,
                  top=0.83, bottom=0.11, wspace=0.32, hspace=0.18)
    figtitle(fig, "Cutting thinking wins early and loses late")
    fig.text(0.012, 0.93, "Ban on the words that open self-checks (Hmm, Wait) vs no ban; 18 public games, 60 minutes each",
             fontsize=9, color=INK2, va="top")
    ax = fig.add_subplot(gs[0, 0])
    axd = fig.add_subplot(gs[1, 0], sharex=ax)
    axh = fig.add_subplot(gs[:, 1])
    for a, col, lab in (("ctrl", BLUE, "No ban (control)"), ("ban", ORANGE, "Ban")):
        m = budgets <= tmax[a]
        ax.plot(budgets[m] / 1e3, sc[a][m], color=col, lw=LW, solid_joinstyle="round", label=lab, zorder=3)
        ax.plot([tmax[a] / 1e3], [sc[a][m][-1]], "o", ms=MS, color=col, mec=SURF, mew=RING, zorder=4)
    ax.text(tmax["ctrl"] / 1e3 - 2, sc["ctrl"][budgets <= tmax["ctrl"]][-1] + 2.2, "No ban", ha="right", fontsize=8.5,
            color=INK)
    ax.text(tmax["ban"] / 1e3 + 3, sc["ban"][budgets <= tmax["ban"]][-1] - 2.8, "Ban", ha="left", fontsize=8.5, color=INK)
    ax.set_ylim(0, 56)
    ax.set_yticks([0, 20, 40])
    grid(ax, "y")
    ax.set_ylabel("Mean score per game\nwithin B tokens")
    ax.legend(loc="upper left", handlelength=1.6, borderaxespad=0.2)
    ax.axvline(b40 / 1e3, color=AXIS, lw=0.8, zorder=1)
    ax.text(3, 29.5, f"At {k(b40)} tokens (grey line) the\nban leads by {diff[i40]:+d} levels:\n{a40} games ahead, {z40} behind",
            fontsize=8.5, color=INK2, va="bottom")
    ax.text(150, 9, f"At 60 min: {lend[0]} vs {lend[1]} levels,\n{aend} games ahead, {zend} behind", fontsize=8.5,
            color=INK2, va="bottom")
    plt.setp(ax.get_xticklabels(), visible=False)
    # difference panel
    axd.fill_between([0, 222], [-floor] * 2, [floor] * 2, color=MUTED, alpha=0.13, lw=0, zorder=1)
    axd.axhline(0, color=AXIS, lw=0.8, zorder=2)
    axd.step(budgets / 1e3, diff, where="post", color=ORANGE, lw=LW, zorder=3)
    axd.set_xlim(0, 222)
    axd.set_ylim(-23, 9)
    axd.set_yticks([-20, -10, 0])
    axd.set_yticklabels(["−20", "−10", "0"])
    grid(axd, "y")
    axd.set_ylabel("Ban − no ban,\nlevels")
    axd.set_xticks([0, 50, 100, 150, 200])
    axd.set_xticklabels(["0", "50k", "100k", "150k", "200k"])
    axd.set_xlabel("Generated tokens per game (budget B)")
    axd.text(3, floor + 0.6, f"±{floor}-level noise floor at 60 min (E072)", fontsize=8.5, color=INK2, va="bottom")
    # hidden-set panel
    xs = np.arange(len(hv))
    for x, (lab, v, c) in zip(xs, hv):
        axh.plot([x], [v], "o", ms=MS + 1, color=c, mec=SURF, mew=RING, zorder=3)
        axh.text(x, v + 1.2, f"{v:.2f}", ha="center", fontsize=8.5, color=INK)
    axh.set_xlim(-0.6, len(hv) - 0.4)
    axh.set_ylim(0, 30)
    axh.set_yticks([0, 10, 20, 30])
    axh.set_xticks(xs)
    axh.set_xticklabels([h[0] for h in hv], fontsize=8)
    axh.tick_params(axis="x", length=0)
    grid(axh, "y")
    ptitle(axh, "Hidden games,\none draw each")
    axh.set_ylabel("Leaderboard score")
    save(fig, out, "fig2_horizon_reversal")
    rows = [[int(b), round(sc["ctrl"][i], 3), round(sc["ban"][i], 3), int(lv["ctrl"][i]), int(lv["ban"][i]), int(diff[i])]
            for i, b in enumerate(budgets) if b % 10e3 == 0]
    rows += [["hidden-set " + lab.replace("\n", " "), v, "", "", "", ""] for lab, v, _ in hv]
    write_csv(out, "fig2", ["budget_tokens_per_game", "score_control", "score_ban", "levels_control", "levels_ban",
                            "levels_ban_minus_control"], rows)
    ahead = [b for i, b in enumerate(budgets) if b > 0 and sc["ban"][i] > sc["ctrl"][i]]
    cross = max(ahead) + 1e3   # the ban's score lead ends here (it never leads again)
    return dict(d40=int(diff[i40]), a40=a40, z40=z40, lend=lend, aend=aend, zend=zend, floor=floor, hv=hv,
                budgets=budgets, sc=sc, tmax=tmax, cross=cross)


# ---------------------------------------------------------------------------------------------------- figure 3
AGI3_LABELS = {
    354982189: "Gambler v1 (other kernel)", 355150259: "E016 prompt guide (SGLang)",
    355467800: "Wrong notebook (never plays)", 355734881: "E040 (vLLM)", 356099442: "E055 word ban",
    357018543: "E060/E061 (vLLM)", 357121906: "E073 prune384", 357441618: "E081 hand-off",
    357974936: "E098 final version", 358364499: "E098 final version", 358728294: "Facts v1 (E118)",
    361111519: "E146 stub resave", 362115816: "E154 keep-alive", "E149v3": "E149 half-clock",
}
EXCLUDED = {355467800, 361111519, "E149v3"}   # instrument failures: no game was played as designed


def fig3(out):
    draws = agi3_draws()
    groups = {}
    for x in sorted(draws, key=lambda r: r["date"]):
        groups.setdefault(AGI3_LABELS[x["sv"]], dict(scores=[], excl=x["sv"] in EXCLUDED))["scores"].append(x["score"])
    agi2 = {}
    agi2_labels = {359703883: "v4: public 4B notebook", 359728161: "v5: E132 second pass"}
    for _, date, s, sv, _ in sorted(LBN2.SUBS, key=lambda r: r[1]):
        assert sv in LBN2.VERSIONS
        agi2.setdefault(agi2_labels[sv], dict(scores=[], excl=False))["scores"].append(s)
    e098 = groups["E098 final version"]["scores"]
    first5 = [x["score"] for x in sorted(draws, key=lambda r: r["date"]) if x["sv"] in (357974936, 358364499)][:5]
    fac = groups["Facts v1 (E118)"]["scores"]
    sd5, sd_pool, sd_old = LBN.sd(first5), LBN.pooled([e098, fac])[0], LBN.OLD_SD
    mde_plan = LBN.mde(sd_pool, 4, 10)
    n_eff = 2 / (1 / 4 + 1 / 10)
    n_plus1 = math.ceil(LBN.n_balanced(1.0, sd_pool, LBN.ALPHA1))

    rows = [("hdr", "ARC-AGI-3 (public half: 55 hidden games)")] + [("g", n, g) for n, g in groups.items()] +         [("gap", None)] + [("hdr", "ARC-AGI-2 (other competition)")] + [("g", n, g) for n, g in agi2.items()]
    fig = plt.figure(figsize=(WIDTH, 7.6))
    figtitle(fig, "What one draw can show", y=0.99)
    fig.text(0.012, 0.962, "Top: every leaderboard draw by saved version. Bottom: the smallest true difference a test "
             "with n draws per version detects", fontsize=9, color=INK2, va="top")
    ax = fig.add_axes([0.335, 0.375, 0.47, 0.522])
    axm = fig.add_axes([0.255, 0.065, 0.66, 0.205])
    ax.set_xlim(0, 35)
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.set_xticks([0, 10, 20, 30])
    grid(ax, "x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=11)
    ax.set_xlabel("Public-leaderboard score")
    ylab, ypos, csvrows, agi3_rows = [], [], [], []
    for y, r in enumerate(rows):
        if r[0] == "hdr":
            ax.text(-0.69, y + 0.1, r[1], transform=ax.get_yaxis_transform(), ha="left", va="center",
                    fontweight="semibold", fontsize=8.5, color=INK)
            continue
        if r[0] == "gap":
            continue
        name, g = r[1], r[2]
        sc = g["scores"]
        ylab.append(f"{name}  ({len(sc)})")
        ypos.append(y)
        if not agi2.get(name):
            agi3_rows.append(y)
        offs = [0, 0.17, -0.17, 0.09, -0.09, 0.0]
        order = np.argsort(sc)
        if len(sc) > 1:
            m, s = statistics.mean(sc), statistics.stdev(sc)
            ax.plot([m - s, m + s], [y, y], color=INK2, lw=1.6, zorder=3, solid_capstyle="butt")
            ax.plot([m, m], [y - 0.34, y + 0.34], color=INK, lw=2.0, zorder=5, solid_capstyle="butt")
            val = f"{m:.2f} ± {s:.2f}"
        elif g["excl"]:
            val = f"{sc[0]:.2f}, excluded"
        else:
            val = f"{sc[0]:.2f}"
        for j, i in enumerate(order):
            dy = offs[j % len(offs)] if len(sc) > 1 else 0
            if g["excl"]:
                ax.plot([sc[i]], [y + dy], "o", ms=MS, mfc=SURF, mec=MUTED, mew=1.4, zorder=4, clip_on=False)
            else:
                ax.plot([sc[i]], [y + dy], "o", ms=MS, color=BLUE, mec=SURF, mew=RING, zorder=4)
        ax.text(36.2, y, val, ha="left", va="center", fontsize=8, color=MUTED if g["excl"] else INK2, clip_on=False)
        for v in sc:
            csvrows.append([name, v, "excluded (instrument failure)" if g["excl"] else "draw"])
    ax.set_yticks(ypos)
    ax.set_yticklabels(ylab, fontsize=8.2)
    m098 = statistics.mean(e098)
    ax.plot([m098, m098], [min(agi3_rows) - 0.5, max(agi3_rows) + 0.5], color=AXIS, lw=0.8, zorder=1)
    ax.text(m098, min(agi3_rows) - 0.55, f"E098 mean {m098:.2f}", ha="center", va="bottom", fontsize=7.8, color=INK2)
    ax.text(36.2, -0.05, "mean ± SD\nor the draw", ha="left", va="center", fontsize=7.8, color=MUTED, clip_on=False)
    h = [Line2D([], [], marker="o", ls="", ms=MS, color=BLUE, mec=SURF, label="draw"),
         Line2D([], [], marker="o", ls="", ms=MS, mfc=SURF, mec=MUTED, mew=1.4, label="excluded: instrument failure"),
         Line2D([], [], color=INK2, lw=1.6, label="mean ± 1 SD (versions with 2+ draws)")]
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.006, 0.942), ncol=3, fontsize=8, handletextpad=0.3,
               columnspacing=1.2)
    # MDE panel: y axis on the right so the line labels sit in the left margin, at n = 1 where the lines are apart
    n = np.geomspace(1, 100, 200)
    for s, c, lab in ((sd_old, RAMP3[2], f"SD {sd_old:.1f} (borrowed)"), (sd_pool, RAMP3[1], f"SD {sd_pool:.2f} (ours, pooled)"),
                      (sd5, RAMP3[0], f"SD {sd5:.2f} (first 5 draws)")):
        axm.plot(n, [LBN.mde(s, x, x) for x in n], color=c, lw=LW, label=lab, solid_capstyle="round")
        axm.plot([1], [LBN.mde(s, 1, 1)], "o", ms=4.5, color=c, mec=SURF, mew=0.8, clip_on=False, zorder=5)
        axm.text(0.93, LBN.mde(s, 1, 1), lab, ha="right", va="center", fontsize=8, color=INK2, clip_on=False)
    axm.set_xscale("log")
    axm.set_xlim(1, 100)
    axm.set_ylim(0, 17)
    axm.set_xticks([1, 2, 5, 10, 20, 50, 100])
    axm.set_xticklabels(["1", "2", "5", "10", "20", "50", "100"])
    axm.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axm.yaxis.tick_right()
    axm.yaxis.set_label_position("right")
    axm.spines["left"].set_visible(False)
    axm.spines["right"].set_visible(True)
    axm.set_yticks([0, 5, 10, 15])
    grid(axm, "y")
    axm.axhline(1, color=AXIS, lw=0.8)
    axm.plot([n_eff], [mde_plan], "o", ms=MS, color=RAMP3[1], mec=SURF, mew=RING, zorder=5)
    axm.annotate(f"10 control + 4 new draws: {mde_plan:.2f}", xy=(n_eff, mde_plan), xytext=(11, 10.0), fontsize=8,
                 color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
    axm.plot([n_plus1], [LBN.mde(sd_pool, n_plus1, n_plus1)], "o", ms=MS, color=RAMP3[1], mec=SURF, mew=RING, zorder=5)
    axm.annotate(f"+1 point needs {n_plus1} draws per version", xy=(n_plus1, 1.0), xytext=(14, 6.2), fontsize=8,
                 color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
    axm.set_xlabel("Draws per version (both versions, log scale)")
    axm.set_ylabel("Points")
    ptitle(axm, "Minimum detectable effect (90% power, one-sided 0.05)")
    save(fig, out, "fig3_noise_floor")
    csvrows += [["MDE, 10 control + 4 candidate draws, SD %.2f" % sd_pool, round(mde_plan, 3), "computed"],
                ["draws per version for +1 at SD %.2f" % sd_pool, n_plus1, "computed"],
                ["SD first 5 E098 draws", round(sd5, 3), "computed"], ["SD pooled E098 x6 + facts x2", round(sd_pool, 3), "computed"],
                ["SD borrowed", sd_old, "read"]]
    write_csv(out, "fig3", ["version", "score", "kind"], csvrows)
    excl = [(n_, g["scores"][0]) for n_, g in groups.items() if g["excl"]]
    return dict(sd5=sd5, sd_pool=sd_pool, sd_old=sd_old, mde_plan=mde_plan, n_plus1=n_plus1, excl=excl,
                m098=m098, sd098=statistics.stdev(e098), n_versions=len(groups))


# ---------------------------------------------------------------------------------------------------- figure 4
def fig4(out):
    rows = t2a_rows()
    e73 = e073_numbers()
    labels = {"MTP (E098, check)": "MTP head (E098, served)", "DFlash2-3": "DFlash2 drafter, 3 drafts",
              "DFlash2-7": "DFlash2 drafter, 7 drafts"}
    fig = plt.figure(figsize=(WIDTH, 4.5))
    figtitle(fig, "Memory prices each lever")
    fig.text(0.012, 0.93, "A faster drafter's own cache halves the games that fit; pruning buys memory, not speed",
             fontsize=9, color=INK2, va="top")
    ax = fig.add_axes([0.25, 0.53, 0.62, 0.30])
    ax.set_xlim(0, 200)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xticks([0, 50, 100, 150, 200])
    grid(ax, "x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([labels.get(r["name"], r["name"]) for r in rows])
    ax.set_xlabel("KV-pool blocks one full-context game holds (163,840 tokens)")
    hh = h_in_data(ax, 0.20)
    gap = 1.4 * per_inch(ax)[0] / 72   # 1.4 pt surface gap between stacked segments
    for y, r in enumerate(rows):
        hbar(ax, y, 0, r["model"], hh, BLUE, round_end=r["drafter"] == 0)
        if r["drafter"]:
            hbar(ax, y, r["model"], r["drafter"], hh, ORANGE)
            ax.plot([r["model"], r["model"]], [y - hh / 2, y + hh / 2], color=SURF, lw=1.4, zorder=6,
                    solid_capstyle="butt")
            ax.text(r["model"] + r["drafter"] / 2, y, f"{r['drafter']}", ha="center", va="center", fontsize=8.5,
                    color=INK, zorder=7)
        ax.text(r["model"] / 2, y, f"{r['model']}", ha="center", va="center", fontsize=8.5, color=SURF, zorder=7)
        ax.text(r["total"] + 3, y, f"{r['total']} → {r['fit']} games fit", ha="left", va="center", fontsize=8.5,
                color=INK, fontweight="semibold" if r["fit"] < 17 else "normal")
    _ = gap
    h = [Line2D([], [], color=BLUE, lw=6, label="model's cache"), Line2D([], [], color=ORANGE, lw=6, label="drafter's cache")]
    ax.legend(handles=h, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize=8.5, borderaxespad=0.2)
    # pruning: pool and speed
    pairs = (("KV pool (blocks of %s tokens)" % f"{e73['block']:,}", e73["pool_ctrl"], e73["pool_prune"], 2000,
              [0, 1000, 2000], lambda v: f"{v:,.0f}", f"×{e73['pool_prune'] / e73['pool_ctrl']:.2f}"),
             ("Decode speed, 18 games busy (tok/s)", e73["speed_ctrl"], e73["speed_prune"], 1200,
              [0, 600, 1200], lambda v: f"{v:,.1f}", f"×{e73['speed_prune'] / e73['speed_ctrl']:.3f}"))
    for i, (title, v0, v1, xmax, xt, fmt, ratio) in enumerate(pairs):
        a2 = fig.add_axes([0.25 + i * 0.40, 0.11, 0.25, 0.20])
        a2.set_xlim(0, xmax)
        a2.set_ylim(1.6, -0.6)
        grid(a2, "x")
        a2.spines["left"].set_visible(False)
        a2.tick_params(axis="y", length=0)
        a2.set_yticks([0, 1])
        a2.set_yticklabels(["512 experts", "384 experts"] if i == 0 else ["", ""])
        a2.set_xticks(xt)
        a2.set_xticklabels([f"{x:,.0f}" for x in xt])
        hh2 = h_in_data(a2, 0.17)
        hbar(a2, 0, 0, v0, hh2, MUTED)
        hbar(a2, 1, 0, v1, hh2, BLUE)
        a2.text(v0 + xmax * 0.02, 0, fmt(v0), va="center", fontsize=8.5, color=INK)
        a2.text(v1 + xmax * 0.02, 1, f"{fmt(v1)}  ({ratio})", va="center", fontsize=8.5, color=INK)
        ptitle(a2, title)
    fig.text(0.25, 0.395, f"Pruning a quarter of the experts (E073): {e73['games_ctrl']} → {e73['games_prune']} games in "
             "flight at a 92k window", fontsize=9, color=INK, fontweight="semibold")
    save(fig, out, "fig4_memory_prices")
    write_csv(out, "fig4", ["what", "model_blocks", "drafter_blocks", "total_blocks", "games_fit", "label"],
              [[r["name"], r["model"], r["drafter"], r["total"], r["fit"], "computed (real allocator; MTP row = logged)"]
               for r in rows] +
              [["pool blocks 512 / 384 experts", e73["pool_ctrl"], "", e73["pool_prune"], "", "computed from measured GiB"],
               ["decode tok/s 512 / 384 experts (18 busy)", e73["speed_ctrl"], "", e73["speed_prune"], "", "measured"],
               ["games in flight 512 / 384 experts", e73["games_ctrl"], "", e73["games_prune"], "", "computed (E073)"]])
    return dict(rows=rows, e73=e73)


# ---------------------------------------------------------------------------------------------------- figure 5
def fig5(out):
    a, b, extra = agi2_funnels()
    fig = plt.figure(figsize=(WIDTH, 5.2))
    figtitle(fig, "Reachable is not found (ARC-AGI-2)")
    fig.text(0.012, 0.93, "Two separate task sets: the answer was within reach of a small decoder change, yet no search "
             "found it", fontsize=9, color=INK2, va="top")
    panels = ((a, RAMP5, "(a) Without test-time training: 4B model on CPU (E142 → E151 → E152)", [0.42, 0.47, 0.50, 0.35]),
              (b, RAMP3, "(b) With test-time training, on the competition box (E150)", [0.42, 0.115, 0.50, 0.22]))
    for stages, ramp, title, pos in panels:
        ax = fig.add_axes(pos)
        n = len(stages)
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_xlim(0, 56)
        ax.axis("off")
        hh = h_in_data(ax, 0.17)
        for y, (lab, v, eid) in enumerate(stages):
            if v > 0:
                hbar(ax, y, 0, v, hh, ramp[min(y, len(ramp) - 1)])
            ax.text(v + 0.8, y, f"{v}", va="center", ha="left", fontsize=9, color=INK,
                    fontweight="semibold" if v == 0 else "normal")
            ax.text(-1.2, y, f"{lab}", va="center", ha="right", fontsize=8.5, color=INK2)
        ax.plot([0, 0], [-0.4, n - 0.6], color=AXIS, lw=0.8)
        ax.text(-0.77 * 56, -1.0, title, fontsize=9.5, fontweight="semibold", color=INK, va="bottom", ha="left")
    fig.text(0.012, 0.06, f"Overlap: {extra['overlap']} of the {extra['n_resc']} rescuable tasks of (a) are among the "
             f"{extra['n_gated']} gated tasks of (b). NET correct answers vs the 4B notebook: {extra['net']}",
             fontsize=8.2, color=INK2, va="top")
    save(fig, out, "fig5_agi2_funnel")
    write_csv(out, "fig5", ["panel", "stage", "count", "source"],
              [["a", s, v, e] for s, v, e in a] + [["b", s, v, e] for s, v, e in b] +
              [["a", "already found by the floor's own 16 views (not plotted)", extra["hits"], "E151"],
               ["a&b", "rescuable tasks of (a) among the gated tasks of (b)", extra["overlap"], "E151 + E150"],
               ["b", "NET (v6 right & v4 wrong) - (v4 right & v6 wrong)", extra["net"], "E150"]])
    return dict(a=a, b=b, extra=extra)


# ---------------------------------------------------------------------------------------------------- captions
def ledger_line(pattern):
    """1-based line number in experiments/ledger.md of the first line matching pattern (the only record of a number);
    release: the number the 2026-10-27 run printed, stored with the line in data/ledger_excerpts.json."""
    for x in _ledger_lines():
        if re.search(pattern, x["text"]):
            return x["line_in_captions"]
    raise AssertionError(f"ledger line not found: {pattern}")


def captions(out, d1, r2, r3, r4, r5, z98):
    caps = [
        ("fig1_half_tokens_did_not_cost",
         "Hidden-set score, E081 one draw, E098 six ± SE [measured]. Tokens per game [computed: E081/E137 tok/s × "
         f"{DECODE_HOURS:g} h ÷ {N_HIDDEN}]. Dashed: elasticity {d1['el']:.2f} (E072) extrapolated, not fitted; dotted: zero "
         "returns above 180k. Confounded: reply cap, window, timeout fix, games in flight."),
        ("fig2_horizon_reversal",
         "Self-check word ban vs control, 18 public games × 60 min (E058) [measured]: score and level gap within B "
         f"tokens; floor ±{r2['floor']} levels (E072). Right: hidden-set draws, one each, not identical code "
         "(E055, E040, E061) [measured]."),
        ("fig3_noise_floor",
         "Every leaderboard draw by saved version, both competitions (E138, E154, E149) [measured]; mean ± 1 SD. "
         f"Hollow: instrument failures, excluded. Bottom: minimum detectable effect at SD {r3['sd5']:.2f} / "
         f"{r3['sd_pool']:.2f} / {r3['sd_old']:.1f}, 90% power, one-sided 0.05 [computed, E138]."),
        ("fig4_memory_prices",
         "Pool blocks per 163,840-token game and games that fit [computed with the server's allocator; MTP row "
         "reproduces its log; E137]. Bottom: pruning (E073): pool [computed from measured GiB], decode speed [measured], "
         f"{r4['e73']['games_ctrl']} → {r4['e73']['games_prune']} games [computed fit]."),
        ("fig5_agi2_funnel",
         "ARC-AGI-2 counts [measured]. (a) No test-time training: misses, rescuable (E142), reachable in 16 views "
         "(E151), real-decoder runs (E152). (b) Test-time training on the box (E150): gated tasks, complete contests, new "
         "correct answers. Different task sets."),
        ("cover",
         "Cover: fig2 reduced. Ban vs no ban, mean score within B tokens per game, 18 public games × 60 min (E058) "
         "[measured]; shaded: first 40k tokens (ban +4 levels). Right: hidden-set draws, one each, not identical code (E055, E040, "
         "E061) [measured]."),
        ("cover_alt",
         f"Alternative cover: fig1 reduced. Scores [measured; E081 one draw, E098 six]; tokens per game [computed, E081, "
         f"E137; {DECODE_HOURS:g} h a guess]; {d1['p98']:.0f} = constant-elasticity extrapolation [computed, E072]; the "
         "gain's cause is not isolated."),
    ]
    l149 = ledger_line(r"Result, board draw 1 \[measured 2026-10-26, ref 57669233")
    l150 = ledger_line(r"the truth appeared in 0 of 14 contests")
    l073 = ledger_line(r"so \d+ games fit where the unpruned pool holds \d+")
    lines = ["# Figure captions (one line each, <= 40 words; generated by analysis/make_paper_figures.py)", "",
             "Numbering in paper draft v4 (reports/paper_draft_v4_2026-10-27.md s.2): Figure 1 = `fig2_horizon_reversal` "
             "(also the cover), Figure 2 = `fig1_half_tokens_did_not_cost`; Figures 3-5 = `fig3`-`fig5`.", ""]
    for stem, c in caps:
        n = words(c)
        assert n <= 40, (stem, n)
        lines.append(f"- `{stem}` ({n} words): {c}")
    lines += ["", "## Numbers whose only record is a ledger line (no data file holds them)", "",
              f"- 0.31, E149 v3 half-clock draw 1 (fig3, hollow, excluded): experiments/ledger.md l.{l149}, E149 "
              "'Result, board draw 1' (ref 57669233, kernel version 190) [measured]. The E138 table in "
              "analysis/leaderboard_noise.py ends before it.",
              f"- 'Among the scored LDS finds the truth appeared in 0 of 14 contests' (fig5 b): experiments/ledger.md "
              f"l.{l150}, E150 Result line [measured]. The plotted last stage is the data file's own count "
              "(smoke_verdict.json numbers.v6_only = 0); selection.json carries no truth flags.",
              f"- 18 → 24 games in flight with pruning (fig4): experiments/ledger.md l.{l073}, E073 RESULT [computed fit "
              "from the measured pool].",
              f"- fig1's dotted line ends at {z98:.1f} with this script's E098 token midpoint ({d1['x98'] / 1e3:.0f}k); "
              "the v4 plan's ~21.5 uses 150k.",
              "", "Labels: [measured] read from a run log, submission list or summary file; [computed] arithmetic on "
              "measured inputs; [guess] an assumption. Plotted numbers per figure: `figN_data.csv`."]
    (out / "captions.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return caps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path.cwd() / "paper_figures_out"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d1 = fig1_data()
    z98 = fig1(out, d1)
    cover_alt(out, d1)
    r2 = fig2(out)
    cover(out, r2)
    r3 = fig3(out)
    r4 = fig4(out)
    r5 = fig5(out)
    caps = captions(out, d1, r2, r3, r4, r5, z98)
    print(f"Figure 1: E081 {d1['e081']:.2f} at {d1['x81']:,.0f} tok/game; E098 {d1['m98']:.2f} +- {d1['se98']:.2f} at "
          f"{d1['x98lo']:,.0f}-{d1['x98hi']:,.0f}; extrapolation {d1['p98']:.2f} (band {d1['p98_hi']:.2f}-{d1['p98_lo']:.2f}); "
          f"zero returns above 180k {z98:.2f}")
    print(f"Figure 2: 40k {r2['d40']:+d} levels ({r2['a40']} ahead, {r2['z40']} behind); end {r2['lend'][0]} vs {r2['lend'][1]} "
          f"({r2['aend']} ahead, {r2['zend']} behind); floor {r2['floor']}; score lead ends at {r2['cross']:,.0f}; "
          f"hidden {[(h[0].split(chr(10))[0], h[1]) for h in r2['hv']]}")
    print(f"Figure 3: SD first5 {r3['sd5']:.2f}, pooled {r3['sd_pool']:.2f}, borrowed {r3['sd_old']}; MDE 10+4 {r3['mde_plan']:.2f}; "
          f"+1 needs {r3['n_plus1']}/version; E098 {r3['m098']:.2f} SD {r3['sd098']:.2f}; excluded {r3['excl']}")
    print(f"Figure 4: {[(r['name'], r['model'], r['drafter'], r['fit']) for r in r4['rows']]}; E073 {r4['e73']}")
    print(f"Figure 5: a {[s[1] for s in r5['a']]}, b {[s[1] for s in r5['b']]}, {r5['extra']}")
    for stem, c in caps:
        print(f"  caption {stem}: {words(c)} words")
    return 0


if __name__ == "__main__":
    sys.exit(main())
