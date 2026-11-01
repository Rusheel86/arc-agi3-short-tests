"""Leaderboard noise floor: how much can a few leaderboard draws tell you?

Plain words: a Kaggle rerun of the SAME notebook does not score the same twice (the agent samples, the platform's speed
varies). Give this tool every leaderboard score you have, labelled by saved version, and it tells you
  1. the spread of one draw (per-version SD with its 90% interval, and the pooled within-version SD),
  2. the smallest true difference a comparison with n draws per version can detect (minimum detectable effect, MDE),
  3. how many draws per version you need to detect a given difference, and
  4. optionally, whether two versions differ (Welch t, and a z-test at the pooled SD).
Generalised from the project's E138 analysis (one-sided alpha 0.05, power 0.90 by default: the decision is
"adopt or not").

Usage:
  python -I noise_floor.py DRAWS.csv [--pool A,B,...] [--compare CANDIDATE CONTROL] [--n 1,2,3,4,6,10,20,50]
                                     [--delta 1,2,3] [--alpha 0.05] [--power 0.90] [--sd S] [--json OUT.json]

DRAWS.csv: one row per leaderboard draw. Required columns `version` and `score`; optional `include` (0 / no / false =
keep the row in the table but out of every statistic: instrument failures, void runs), any other columns are ignored.
Example: examples/arc_agi3_public_leaderboard_draws.csv (our ARC-AGI-3 public-leaderboard draws, 2026-10-03..10-29).

Formulas (two-sample normal arithmetic, equal SD sigma in both versions):
  MDE(n_a, n_b)        = (z_{1-alpha} + z_power) * sigma * sqrt(1/n_a + 1/n_b)
  draws per version    = 2 sigma^2 (z_{1-alpha} + z_power)^2 / delta^2              (balanced design, rounded up)
  candidate draws n_c  = 1 / ((delta / (z_{1-alpha} + z_power))^2 / sigma^2 - 1/n_0)   (against n_0 existing control draws)
  90% interval of an SD with df degrees of freedom: s * sqrt(df / chi2_{0.95, df}) .. s * sqrt(df / chi2_{0.05, df})
Needs scipy (for the chi-square and t distributions). MIT-0.
"""
import argparse
import csv
import json
import math
import statistics
import sys

from scipy import stats

FALSE = {"0", "no", "false", "n", "excluded", "exclude"}


def z(p):
    return float(stats.norm.ppf(p))


def sd(x):
    return statistics.stdev(x) if len(x) > 1 else float("nan")


def sd_ci(s, df, level=0.90):
    lo = s * math.sqrt(df / stats.chi2.ppf(1 - (1 - level) / 2, df))
    hi = s * math.sqrt(df / stats.chi2.ppf((1 - level) / 2, df))
    return lo, hi


def pooled(groups):
    """(pooled SD, df) over groups with >= 2 draws"""
    groups = [g for g in groups if len(g) > 1]
    ss = sum(sum((v - statistics.mean(g)) ** 2 for v in g) for g in groups)
    df = sum(len(g) - 1 for g in groups)
    return (math.sqrt(ss / df), df) if df > 0 else (float("nan"), 0)


def mde(sigma, n_a, n_b, alpha=0.05, power=0.90):
    return (z(1 - alpha) + z(power)) * sigma * math.sqrt(1.0 / n_a + 1.0 / n_b)


def n_balanced(delta, sigma, alpha=0.05, power=0.90):
    return 2 * sigma ** 2 * (z(1 - alpha) + z(power)) ** 2 / delta ** 2


def n_candidate(delta, sigma, n0, alpha=0.05, power=0.90):
    r = (delta / (z(1 - alpha) + z(power))) ** 2 / sigma ** 2 - 1.0 / n0
    return math.inf if r <= 0 else 1.0 / r


def welch(a, b):
    """mean(a) - mean(b), Welch SE, t, Satterthwaite df, two-sided p"""
    va, vb = statistics.variance(a) / len(a), statistics.variance(b) / len(b)
    se = math.sqrt(va + vb)
    d = statistics.mean(a) - statistics.mean(b)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    t = d / se
    return dict(diff=d, se=se, t=t, df=df, p_two_sided=2 * float(stats.t.sf(abs(t), df)))


def known_sd_test(a, b, sigma):
    se = sigma * math.sqrt(1 / len(a) + 1 / len(b))
    d = statistics.mean(a) - statistics.mean(b)
    zz = d / se
    return dict(diff=d, se=se, z=zz, p_two_sided=2 * float(stats.norm.sf(abs(zz))))


def read_draws(path):
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for i, r in enumerate(csv.DictReader(f), 2):
            if "version" not in r or "score" not in r:
                raise SystemExit(f"{path}: the columns 'version' and 'score' are required")
            try:
                s = float(r["score"])
            except ValueError:
                raise SystemExit(f"{path} line {i}: score {r['score']!r} is not a number")
            inc = str(r.get("include", "1")).strip().lower() not in FALSE
            rows.append(dict(version=r["version"].strip(), score=s, include=inc))
    if not rows:
        raise SystemExit(f"{path}: no rows")
    return rows


def analyse(rows, pool=None, compare=None, ns=(1, 2, 3, 4, 6, 10, 20, 50), deltas=(1, 2, 3), alpha=0.05, power=0.90,
            sigma=None):
    groups = {}
    excluded = []
    for r in rows:
        (groups.setdefault(r["version"], []) if r["include"] else excluded).append(r["score"] if r["include"] else r)
    per = []
    for v, x in groups.items():
        s = sd(x)
        lo, hi = sd_ci(s, len(x) - 1) if len(x) > 1 else (float("nan"), float("nan"))
        per.append(dict(version=v, n=len(x), mean=statistics.mean(x), sd=s, sd_ci90=(lo, hi),
                        se_mean=s / math.sqrt(len(x)) if len(x) > 1 else float("nan")))
    names = list(groups) if not pool else pool
    missing = [v for v in names if v not in groups]
    if missing:
        raise SystemExit(f"--pool: unknown version(s) {missing}; known: {sorted(groups)}")
    sp, df = pooled([groups[v] for v in names])
    plo, phi = sd_ci(sp, df) if df > 0 else (float("nan"), float("nan"))
    sig = sigma if sigma is not None else sp
    if not (sig > 0):
        raise SystemExit("no SD: give at least one version with 2+ included draws, or --sd")
    res = dict(alpha_one_sided=alpha, power=power, versions=per, excluded=excluded,
               pooled=dict(versions=[v for v in names if len(groups[v]) > 1], sd=sp, df=df, sd_ci90=(plo, phi)),
               sigma_used=sig, sigma_source="--sd" if sigma is not None else "pooled",
               one_draw_90=dict(halfwidth=z(0.95) * sig),
               mde_balanced=[dict(n_per_version=n, mde=mde(sig, n, n, alpha, power)) for n in ns],
               draws_per_version=[dict(delta=d, n=math.ceil(n_balanced(d, sig, alpha, power))) for d in deltas])
    if compare:
        a, b = compare
        for v in (a, b):
            if v not in groups:
                raise SystemExit(f"--compare: unknown version {v!r}; known: {sorted(groups)}")
        xa, xb = groups[a], groups[b]
        c = dict(candidate=a, control=b, n_candidate=len(xa), n_control=len(xb), mean_candidate=statistics.mean(xa),
                 mean_control=statistics.mean(xb), mde=mde(sig, len(xa), len(xb), alpha, power),
                 known_sd=known_sd_test(xa, xb, sig),
                 candidate_draws_needed=[dict(delta=d, n=n_candidate(d, sig, len(xb), alpha, power)) for d in deltas])
        if len(xa) > 1 and len(xb) > 1:
            c["welch"] = welch(xa, xb)
        res["compare"] = c
    return res


def fmt(x, nd=2):
    return "n/a" if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))) else f"{x:.{nd}f}"


def report(res):
    out = []
    p = out.append
    p(f"Design: one-sided alpha {res['alpha_one_sided']}, power {res['power']} (z sum "
      f"{z(1 - res['alpha_one_sided']) + z(res['power']):.3f}).")
    p("")
    p("1. Per version (included draws only)")
    for v in res["versions"]:
        lo, hi = v["sd_ci90"]
        p(f"  {v['version'][:40]:40} n={v['n']:<3} mean {v['mean']:6.2f}  SD {fmt(v['sd'])}  90% CI of SD "
          f"{fmt(lo)}-{fmt(hi)}  SE of mean {fmt(v['se_mean'])}")
    if res["excluded"]:
        p(f"  excluded rows (include = 0): " + ", ".join(f"{r['version']} {r['score']:.2f}" for r in res["excluded"]))
    pl = res["pooled"]
    p("")
    p(f"2. Pooled within-version SD over {', '.join(pl['versions']) or 'no version'}: {fmt(pl['sd'])} "
      f"(df {pl['df']}, 90% CI {fmt(pl['sd_ci90'][0])}-{fmt(pl['sd_ci90'][1])})")
    p(f"   SD used below: {fmt(res['sigma_used'], 3)} ({res['sigma_source']}). One draw is uncertain by about "
      f"+/-{res['one_draw_90']['halfwidth']:.1f} points (90%).")
    p("")
    p("3. Minimum detectable effect, n draws of each version")
    p("   " + "  ".join(f"n={m['n_per_version']}: {m['mde']:.2f}" for m in res["mde_balanced"]))
    p("")
    p("4. Draws per version needed to detect a true difference (balanced)")
    p("   " + "  ".join(f"+{d['delta']:g}: {d['n']}" for d in res["draws_per_version"]))
    c = res.get("compare")
    if c:
        p("")
        p(f"5. {c['candidate']} (n={c['n_candidate']}, mean {c['mean_candidate']:.2f}) vs {c['control']} "
          f"(n={c['n_control']}, mean {c['mean_control']:.2f})")
        w = c.get("welch")
        if w:
            p(f"   difference {w['diff']:+.2f}; Welch SE {w['se']:.3f}, t {w['t']:+.2f}, df {w['df']:.2f}, two-sided p {w['p_two_sided']:.2f}")
        k = c["known_sd"]
        p(f"   at SD {res['sigma_used']:.2f} as known: SE {k['se']:.3f}, z {k['z']:+.2f}, two-sided p {k['p_two_sided']:.2f}")
        p(f"   minimum detectable effect of this design: {c['mde']:.2f}")
        p("   candidate draws needed against the existing control draws: " +
          "  ".join(f"+{d['delta']:g}: {'never' if math.isinf(d['n']) else math.ceil(d['n'])}" for d in c["candidate_draws_needed"]))
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csv")
    ap.add_argument("--pool", default="", help="comma-separated versions to pool for the SD (default: every version with 2+ draws)")
    ap.add_argument("--compare", nargs=2, metavar=("CANDIDATE", "CONTROL"))
    ap.add_argument("--n", default="1,2,3,4,6,10,20,50", help="draws per version for the MDE table")
    ap.add_argument("--delta", default="1,2,3", help="true differences (points) for the draws-needed table")
    ap.add_argument("--alpha", type=float, default=0.05, help="one-sided")
    ap.add_argument("--power", type=float, default=0.90)
    ap.add_argument("--sd", type=float, default=None, help="use this per-draw SD instead of the pooled one")
    ap.add_argument("--json", default="", help="also write every number to this JSON file")
    a = ap.parse_args(argv)
    rows = read_draws(a.csv)
    res = analyse(rows, pool=[v.strip() for v in a.pool.split(",") if v.strip()] or None, compare=a.compare,
                  ns=[int(x) for x in a.n.split(",")], deltas=[float(x) for x in a.delta.split(",")],
                  alpha=a.alpha, power=a.power, sigma=a.sd)
    print(report(res))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1, default=lambda o: None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
