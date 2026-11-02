"""Figure 3 (noise floor), version 2: every leaderboard draw on record at 2026-11-01. Writes ONLY new files:
fig3_noise_floor_v2.png / .svg and fig3_data_v2.csv (the originals fig3_noise_floor.* / fig3_data.csv stay untouched),
and inserts one v2 line into captions.md if that file exists in --out.

What changed against fig3 (2026-10-27): 25 ARC-AGI-3 rows instead of 21 and 11 ARC-AGI-2 rows instead of 6.
  * E154 keep-alive now has 4 draws (31.00, 28.05, 27.07, 29.43): mean 28.89 +- SD 1.71. It is its own saved version and
    is NOT pooled with E098 (leaderboard_noise.py section 12).
  * E149 v3's 0.31 keeps its hollow "excluded" ring. E149 v4's 15.06 (kernel v191, ref 57806285) is drawn as an orange
    diamond: a half-clock DOSE draw of its own version, in band, never pooled (section 13). It is not an instrument failure.
  * ARC-AGI-2: v4 n = 6, v5 n = 5 (leaderboard_noise_agi2.py section 7).
  * The MDE panel is unchanged: the pooled SD is still E098 x6 + facts x2 (df 6) = 1.73, so 3.00 / 52 do not move.
The plotting code is fig3() of analysis/make_paper_figures.py with the new rows, labels and layout height (the only
reason it is copied rather than imported: that function is hard-wired to the 10-27 label table and a 7.6 in page).

Usage (repo root): ./.venv/Scripts/python.exe reports/paper_figures/make_fig3_v2.py [--out reports/paper_figures]
Release copy (tools/paper_figures/): python -I make_fig3_v2.py [--out DIR]; it then reads data/agi3_draws_v2.csv and
data/agi2_draws_v2.csv (frozen copies of the two leaderboard tables at 2026-11-01).
Export those two files from the project tables: ... make_fig3_v2.py --export-data DIR
Colours: the project's dataviz palette (blue slot 1, orange slot 2, blue ramp); validate_palette.js passes both slots on white.
"""
import argparse
import csv
import math
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RELEASE = (HERE / "make_paper_figures.py").exists()      # release layout: the figure script sits next to this one
sys.path.insert(0, str(HERE if RELEASE else HERE.parents[1] / "analysis"))
import make_paper_figures as mpf  # noqa: E402  (style, helpers and the original fig3 inputs)

np, plt, matplotlib, Line2D = mpf.np, mpf.plt, mpf.matplotlib, mpf.Line2D
SURF, INK, INK2, MUTED, AXIS = mpf.SURF, mpf.INK, mpf.INK2, mpf.MUTED, mpf.AXIS
BLUE, ORANGE, RAMP3, WIDTH, LW, MS, RING = mpf.BLUE, mpf.ORANGE, mpf.RAMP3, mpf.WIDTH, mpf.LW, mpf.MS, mpf.RING
LBN = mpf.LBN                                              # sd / pooled / mde / n_balanced / OLD_SD / ALPHA1

E098_SV = (357974936, 358364499)
E154_SV, E149V3_SV, E149V4_SV = 362115816, 362164942, 363866748
AGI3_LABELS = {
    354982189: "Gambler v1 (other kernel)", 355150259: "E016 prompt guide (SGLang)",
    355467800: "Wrong notebook (never plays)", 355734881: "E040 (vLLM)", 356099442: "E055 word ban",
    357018543: "E060/E061 (vLLM)", 357121906: "E073 prune384", 357441618: "E081 hand-off",
    357974936: "E098 final version", 358364499: "E098 final version", 358728294: "Facts v1 (E118)",
    361111519: "E146 stub resave", E154_SV: "E154 keep-alive", E149V3_SV: "E149 v3 half-clock",
    E149V4_SV: "E149 v4 half-clock (dose)",
}
EXCLUDED = {355467800, 361111519, E149V3_SV}   # instrument failures: no game was played as designed (hollow ring)
DOSE = {E149V4_SV}                              # own-dose draw of its own version: shown, never pooled (orange diamond)
AGI2_LABELS = {359703883: "v4: public 4B notebook", 359728161: "v5: E132 second pass"}


# ---------------------------------------------------------------------------------------------------- inputs
def read_csv_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return [(int(r["ref"]), r["date_utc"], float(r["score"]), int(r["script_version"])) for r in csv.DictReader(f)]


def load_tables():
    """(ref, UTC date, score, script version) rows of the two leaderboard tables."""
    if RELEASE:
        return read_csv_rows(HERE / "data" / "agi3_draws_v2.csv"), read_csv_rows(HERE / "data" / "agi2_draws_v2.csv")
    return ([(r, d, s, sv) for r, d, s, sv, _ in mpf.LBN.SUBS], [(r, d, s, sv) for r, d, s, sv, _ in mpf.LBN2.SUBS])


def export_data(dest):
    a3, a2 = load_tables()
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name, rows in (("agi3_draws_v2.csv", a3), ("agi2_draws_v2.csv", a2)):
        with open(dest / name, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["ref", "date_utc", "score", "script_version"])
            w.writerows(rows)
    print(f"wrote {len(a3)} + {len(a2)} rows to {dest}")


# ---------------------------------------------------------------------------------------------------- figure
def fig3_v2(out):
    a3, a2 = load_tables()
    draws = [dict(ref=r, date=d, score=s, sv=sv) for r, d, s, sv in a3]
    assert len(draws) == 25 and len(a2) == 11, "the 2026-11-01 tables hold 25 ARC-AGI-3 and 11 ARC-AGI-2 rows"
    groups = {}
    for x in sorted(draws, key=lambda r: r["date"]):
        g = groups.setdefault(AGI3_LABELS[x["sv"]], dict(scores=[], sv=x["sv"], excl=x["sv"] in EXCLUDED, dose=x["sv"] in DOSE))
        g["scores"].append(x["score"])
    agi2 = {}
    for _, date, s, sv in sorted(a2, key=lambda r: r[1]):
        agi2.setdefault(AGI2_LABELS[sv], dict(scores=[], sv=sv, excl=False, dose=False))["scores"].append(s)
    e098 = groups["E098 final version"]["scores"]
    e154 = groups["E154 keep-alive"]["scores"]
    fac = groups["Facts v1 (E118)"]["scores"]
    v4, v5 = agi2["v4: public 4B notebook"]["scores"], agi2["v5: E132 second pass"]["scores"]
    # the pooled SD must still be the registered one (E098 x6 + facts x2, df 6, E154 and the dose draw in no pool)
    assert (len(e098), len(e154), len(fac), len(v4), len(v5)) == (6, 4, 2, 6, 5), "unexpected group sizes"
    assert e154 == [31.00, 28.05, 27.07, 29.43], e154
    first5 = [x["score"] for x in sorted(draws, key=lambda r: r["date"]) if x["sv"] in E098_SV][:5]
    sd5, sd_pool, sd_old = LBN.sd(first5), LBN.pooled([e098, fac])[0], LBN.OLD_SD
    assert abs(sd_pool - 1.735) < 0.001 and abs(sd5 - 1.14) < 0.005, (sd_pool, sd5)
    mde_plan = LBN.mde(sd_pool, 4, 10)
    n_eff = 2 / (1 / 4 + 1 / 10)
    n_plus1 = math.ceil(LBN.n_balanced(1.0, sd_pool, LBN.ALPHA1))
    assert round(mde_plan, 2) == 3.00 and n_plus1 == 52, (mde_plan, n_plus1)
    sd_with_e154, df_with_e154 = LBN.pooled([e098, fac, e154])[:2]      # sensitivity only, never plotted

    rows = [("hdr", "ARC-AGI-3 (public half: 55 hidden games)")] + [("g", n, g) for n, g in groups.items()] + \
        [("gap", None)] + [("hdr", "ARC-AGI-2 (other competition)")] + [("g", n, g) for n, g in agi2.items()]
    H = 8.1                                       # inches (fig3: 7.6; two more rows at the same 0.233 in pitch)
    fig = plt.figure(figsize=(WIDTH, H))
    mpf.figtitle(fig, "What one draw can show", y=1 - 0.076 / H)
    fig.text(0.012, 1 - 0.289 / H, "Top: every draw to 2026-11-01, by saved version. Bottom: the smallest true "
             "difference n draws per version detect", fontsize=9, color=INK2, va="top")
    ax = fig.add_axes([0.335, 2.85 / H, 0.47, 4.43 / H])
    axm = fig.add_axes([0.255, 0.494 / H, 0.66, 1.558 / H])
    ax.set_xlim(0, 35)
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.set_xticks([0, 10, 20, 30])
    mpf.grid(ax, "x")
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
        if name not in agi2:
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
        elif g["dose"]:
            val = f"{sc[0]:.2f}, not pooled"
        else:
            val = f"{sc[0]:.2f}"
        for j, i in enumerate(order):
            dy = offs[j % len(offs)] if len(sc) > 1 else 0
            if g["excl"]:
                ax.plot([sc[i]], [y + dy], "o", ms=MS, mfc=SURF, mec=MUTED, mew=1.4, zorder=4, clip_on=False)
            elif g["dose"]:
                ax.plot([sc[i]], [y + dy], "D", ms=MS - 1.0, color=ORANGE, mec=SURF, mew=RING, zorder=4)
            else:
                ax.plot([sc[i]], [y + dy], "o", ms=MS, color=BLUE, mec=SURF, mew=RING, zorder=4)
        ax.text(36.2, y, val, ha="left", va="center", fontsize=8, color=MUTED if (g["excl"] or g["dose"]) else INK2, clip_on=False)
        for v in sc:
            kind = "excluded (instrument failure)" if g["excl"] else "excluded (own dose, never pooled)" if g["dose"] else "draw"
            csvrows.append([name, v, kind])
    ax.set_yticks(ypos)
    ax.set_yticklabels(ylab, fontsize=8.2)
    m098 = statistics.mean(e098)
    ax.plot([m098, m098], [min(agi3_rows) - 0.5, max(agi3_rows) + 0.5], color=AXIS, lw=0.8, zorder=1)
    ax.text(m098, min(agi3_rows) - 0.55, f"E098 mean {m098:.2f}", ha="center", va="bottom", fontsize=7.8, color=INK2)
    ax.text(36.2, -0.05, "mean ± SD\nor the draw", ha="left", va="center", fontsize=7.8, color=MUTED, clip_on=False)
    h = [Line2D([], [], marker="o", ls="", ms=MS, color=BLUE, mec=SURF, label="draw"),
         Line2D([], [], marker="o", ls="", ms=MS, mfc=SURF, mec=MUTED, mew=1.4, label="excluded: instrument failure"),
         Line2D([], [], marker="D", ls="", ms=MS - 1.0, color=ORANGE, mec=SURF, label="own-dose draw: not pooled"),
         Line2D([], [], color=INK2, lw=1.6, label="mean ± 1 SD (2+ draws)")]
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.006, 1 - 0.441 / H), ncol=4, fontsize=8, handletextpad=0.3,
               columnspacing=1.0)
    # MDE panel: identical to fig3 (the pooled SD, 3.00 and 52 are unchanged by the new draws)
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
    mpf.grid(axm, "y")
    axm.axhline(1, color=AXIS, lw=0.8)
    axm.plot([n_eff], [mde_plan], "o", ms=MS, color=RAMP3[1], mec=SURF, mew=RING, zorder=5)
    axm.annotate(f"10 control + 4 new draws: {mde_plan:.2f}", xy=(n_eff, mde_plan), xytext=(11, 10.0), fontsize=8,
                 color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
    axm.plot([n_plus1], [LBN.mde(sd_pool, n_plus1, n_plus1)], "o", ms=MS, color=RAMP3[1], mec=SURF, mew=RING, zorder=5)
    axm.annotate(f"+1 point needs {n_plus1} draws per version", xy=(n_plus1, 1.0), xytext=(14, 6.2), fontsize=8,
                 color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
    axm.set_xlabel("Draws per version (both versions, log scale)")
    axm.set_ylabel("Points")
    mpf.ptitle(axm, "Minimum detectable effect (90% power, one-sided 0.05)")
    mpf.save(fig, out, "fig3_noise_floor_v2")

    csvrows += [["MDE, 10 control + 4 candidate draws, SD %.2f" % sd_pool, round(mde_plan, 3), "computed"],
                ["draws per version for +1 at SD %.2f" % sd_pool, n_plus1, "computed"],
                ["SD first 5 E098 draws", round(sd5, 3), "computed"], ["SD pooled E098 x6 + facts x2 (df 6)", round(sd_pool, 3), "computed"],
                ["SD borrowed", sd_old, "read"],
                ["E154 own SD (n 4, not in the pool)", round(statistics.stdev(e154), 3), "computed"],
                ["sensitivity, not plotted: SD pooled E098 x6 + facts x2 + E154 x4 (df %d)" % df_with_e154, round(sd_with_e154, 3), "computed"],
                ["sensitivity, not plotted: MDE 10 + 4 draws at that SD", round(LBN.mde(sd_with_e154, 4, 10), 3), "computed"],
                ["sensitivity, not plotted: draws per version for +1 at that SD", math.ceil(LBN.n_balanced(1.0, sd_with_e154, LBN.ALPHA1)), "computed"]]
    with open(out / "fig3_data_v2.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["version", "score", "kind"])
        w.writerows(csvrows)
    return dict(sd5=sd5, sd_pool=sd_pool, sd_old=sd_old, mde_plan=mde_plan, n_plus1=n_plus1, m098=m098, sd098=statistics.stdev(e098),
                m154=statistics.mean(e154), sd154=statistics.stdev(e154), v4=(statistics.mean(v4), statistics.stdev(v4)),
                v5=(statistics.mean(v5), statistics.stdev(v5)), sd_with_e154=sd_with_e154, n_rows=(len(draws), len(a2)))


def caption_v2(r):
    c = ("Every leaderboard draw to 2026-11-01, by saved version, both competitions (E138, E154, E149) [measured]; mean ± 1 SD. "
         "Hollow: instrument failures, excluded. Diamond: E149 v4 half-clock, never pooled. "
         f"Bottom: minimum detectable effect at SD {r['sd5']:.2f} / {r['sd_pool']:.2f} / {r['sd_old']:.1f} [computed, E138].")
    n = mpf.words(c)
    assert n <= 40, n
    return f"- `fig3_noise_floor_v2` ({n} words): {c}"


def update_captions(out, line):
    """insert (or replace) the v2 caption line right after the fig3_noise_floor line; skip if captions.md is not in out."""
    p = out / "captions.md"
    if not p.exists():
        print("captions.md not in --out: caption line not written:\n" + line)
        return
    with open(p, newline="", encoding="utf-8") as f:        # newline="": keep the file's own line endings (captions.md is LF)
        text = f.read()
    eol = "\r\n" if "\r\n" in text else "\n"
    lines = [l for l in text.split(eol) if not l.startswith("- `fig3_noise_floor_v2`")]
    i = next(i for i, l in enumerate(lines) if l.startswith("- `fig3_noise_floor`"))
    lines.insert(i + 1, line)
    with open(p, "w", newline="", encoding="utf-8") as f:
        f.write(eol.join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE if not RELEASE else Path.cwd() / "paper_figures_out"))
    ap.add_argument("--export-data", default="", help="write data/agi3_draws_v2.csv and agi2_draws_v2.csv here and stop")
    a = ap.parse_args()
    if a.export_data:
        export_data(a.export_data)
        return 0
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    r = fig3_v2(out)
    update_captions(out, caption_v2(r))
    print(f"Figure 3 v2: rows {r['n_rows']}; SD first5 {r['sd5']:.2f}, pooled {r['sd_pool']:.3f}, borrowed {r['sd_old']}; "
          f"MDE 10+4 {r['mde_plan']:.2f}; +1 needs {r['n_plus1']}/version; E098 {r['m098']:.2f} SD {r['sd098']:.2f}; "
          f"E154 {r['m154']:.2f} SD {r['sd154']:.2f}; v4 {r['v4'][0]:.2f} SD {r['v4'][1]:.2f}; v5 {r['v5'][0]:.2f} SD {r['v5'][1]:.2f}; "
          f"sensitivity pooled with E154 {r['sd_with_e154']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
