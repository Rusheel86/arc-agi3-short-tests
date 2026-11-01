"""E155 (ledger, registered 2026-10-28 before the data existed): token elasticity at deployed depth on community games,
by truncating E110's E098 run at fixed token budgets per game. CPU only, read-only on every input.

Plain words: E110 let the deployed agent (E098) play 38 community-made games for ~60 minutes each. For every game the
harness logged how many tokens the model had generated when each level was finished. This script asks: had every game
been stopped after its first B generated tokens, how much of the game would have been finished? It compares B = 50k
with B = 100k (or 90k / 80k, by a rule fixed in advance) and turns the ratio into an elasticity e: how fast completion
grows with tokens (e = 1: doubling tokens doubles completion; e = 0: extra tokens buy nothing). The same is computed for
two public 60-minute controls (E058, E060-in-E061) as the reference e_pub, and the registered verdict is printed last.

Every rule is the one in the project ledger's entry E155 (the paper's pre-registration text + closed choices C1-C15,
copied in this folder's README.md); the code cites them as [Cn]. Nothing here is tuned on E110 data: the entry was
written before the zip was downloaded.
Release copy: identical to the project's analysis/e110_truncation.py except the file locations (REF_FILES, LOCAL_ENVS,
the default --out) and this docstring; the registered reference files are slim copies in data/reference/ (only the
fields this script reads).

  - Clock [C1]: arm_e110/benchmark.json per-action history; level k counts at budget B iff the cumulative generated
    tokens at its completing action (sum(actions_per_level[:k])) are <= B (E059/E072: game_curve.py, copied from the project's k_killtests.py).
  - Fallback [C2]: a game whose benchmark record is missing or corrupt is read from <gid>_p0_requests.jsonl (usage.
    completion_tokens) + artifacts/<gid>_p0_events.jsonl (level_completed) -- the same clock, cross-checked on every game
    that has both.
  - Metric [C7]: completion S(B) = mean over games of 100 x sum of finished level numbers / sum of all level numbers.
  - Window [C4]: B_lo = 50k; B_hi = the first of 100k / 90k / 80k that >= 30 valid games reached (total tokens >= B, or the
    game finished all its levels); none -> VOID.
  - e = ln(S(B_hi) / S(B_lo)) / ln(B_hi / B_lo); 90% CI: game-cluster bootstrap, 10,000 resamples, random.Random(110) [C8].
  - Verdict [C12]: VOID / HOLDS (e >= 0.33 and CI low > 0) / WEAKER (e < 0.33 and CI high < e_pub) / INCONCLUSIVE.

Usage (the results are data and are never imported or executed, so run with -I):
  PYTHONUTF8=1 python -I tools/e155_truncation/e110_truncation.py e110_results.zip [--out DIR]
      [--obtained YYYY-MM-DD] [--no-crosscheck] [--no-figure] [--boot N]
  src: the zip itself (read in place, nothing extracted) or the unzipped folder.
Writes <out>/e110_truncation_result.json, <out>/fig6_completion_vs_tokens.csv and .pdf/.png (default out:
./e155_out/ in the current folder). The last printed line is the registered verdict ("E155 VERDICT: ...").
Label every number "community games, deployed depth, one run" -- never "hidden" [C15].
"""
import argparse
import ast
import csv
import datetime as dt
import io
import json
import math
import os
import random
import re
import statistics
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:          # -I leaves the script's folder off sys.path; this folder is our own code, never the data
    sys.path.insert(0, HERE)
import e110_readout as R  # noqa: E402  (stem_of, source_of, request_action_ranges; unchanged)

# ------------------------------------------------------------------------------------------------ registered constants
ARM = "arm_e110"
B_LO = 50_000                                   # paper text
B_HI_CANDIDATES = (100_000, 90_000, 80_000)     # paper text: 100k, else the largest of {90k, 80k} that >= 30 games reached
MIN_GAMES = 30                                  # paper text (VOID (b) and the window rule) [C3, C5]
NBOOT = 10_000                                  # paper text
SEED = 110                                      # paper text
HOLDS_E = 0.33                                  # paper text (E072's interval lower end)
E110_LEVELS = 150                               # E110 headline [measured 10-14]
E110_TOKENS = 3_881_160                         # E110 headline [measured 10-14] (ledger: 3.88M)
REPRO_TOL = 0.01                                # paper text: within 1%
OBTAIN_BY = dt.date(2026, 11, 4)                # paper text: zip obtained by 11-04 [C11]
MAX_DROPPED = 0.01                              # [C8] more than 1% undefined resamples -> INCONCLUSIVE
CROSS_TOL = 0.01                                # [C2] two clocks agree within 1%
REF_FILES = {"E058": os.path.join(HERE, "data", "reference", "e058_arm_long_control_benchmark.json"),
             "E061": os.path.join(HERE, "data", "reference", "e061_arm_long_control_benchmark.json")}   # [C9]
# [C2] optional local copies of the community-game datasets (not shipped: their own licences); absent = only e110_envs/
LOCAL_ENVS = (os.path.join(HERE, "data", "dreamteam_games_dataset", "environment_files_generated"),
              os.path.join(HERE, "data", "arc_interactive_games_dataset", "environment_files"))
CURVE_STEP = 5_000                              # Figure 6 only [C13]
TOK_RE = re.compile(r"tokens=(\d+)")


def say(*a):
    print(*a, flush=True)


def q(v, f):
    """t3_recheck's percentile rule [C8]: sorted list, index round(f (n - 1))"""
    s = sorted(v)
    return s[int(round(f * (len(s) - 1)))]


def kfmt(b):
    return f"{b / 1e3:.0f}k"


# ------------------------------------------------------------------------------------------------ input (zip or folder)
class Source:
    """read-only view of the results: a zip read in place (nothing extracted) or a folder. Names use '/'."""

    def __init__(self, path, arm=ARM):
        self.path = os.path.abspath(path)
        self.arm = arm
        self.zip = None
        if os.path.isdir(self.path):
            names = []
            for d, _, fs in os.walk(self.path):
                for n in fs:
                    names.append(os.path.relpath(os.path.join(d, n), self.path).replace(os.sep, "/"))
            self.names = set(names)
        elif zipfile.is_zipfile(self.path):
            self.zip = zipfile.ZipFile(self.path)
            self.names = {n for n in self.zip.namelist() if not n.endswith("/")}
        else:
            raise ValueError(f"not a folder or a zip file: {path}")
        # the folder that holds arm_e110/ (the zip's root is /kaggle/working; tolerate one extra top folder)
        cands = sorted({n[:n.index(arm + "/")] for n in self.names if (arm + "/") in n and
                        (n.startswith(arm + "/") or ("/" + arm + "/") in n)}, key=len)
        self.prefix = cands[0] if cands else ""

    def has(self, rel):
        return (self.prefix + rel) in self.names

    def list(self, rel_dir):
        p = self.prefix + rel_dir.rstrip("/") + "/"
        return sorted(n[len(p):] for n in self.names if n.startswith(p) and "/" not in n[len(p):])

    def text(self, rel):
        name = self.prefix + rel
        if self.zip is not None:
            return io.TextIOWrapper(self.zip.open(name), encoding="utf-8", errors="strict")
        return open(os.path.join(self.path, *name.split("/")), encoding="utf-8")

    def json(self, rel):
        with self.text(rel) as f:
            return json.load(f)

    def obtained_date(self):
        """[C11] the zip's modification date (the download); a folder: the newest file inside it"""
        if self.zip is not None:
            ts = os.path.getmtime(self.path)
        else:
            ts = max([os.path.getmtime(os.path.join(self.path, *n.split("/"))) for n in self.names] or [0])
        return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).date()

    def close(self):
        if self.zip is not None:
            self.zip.close()


# ------------------------------------------------------------------------------------------------ per-game records
class Corrupt(ValueError):
    pass


def _int(x, what):
    """non-negative integer or Corrupt (12 and 12.0 are fine; True, -1, 'a', 1.5 are not)"""
    if isinstance(x, bool) or x is None:
        if x is None:
            return 0
        raise Corrupt(f"{what} is a bool")
    try:
        f = float(x)
    except (TypeError, ValueError):
        raise Corrupt(f"{what} not a number: {x!r}")
    if not math.isfinite(f) or f < 0 or f != int(f):
        raise Corrupt(f"{what} not a non-negative integer: {x!r}")
    return int(f)


def note_tokens(g):
    m = TOK_RE.search(str(g.get("solver_note") or ""))
    return int(m.group(1)) if m else None


def bench_totals(g):
    """(levels_completed, total tokens) from a benchmark record's fields, for the reproduction check [C10]; None if unreadable"""
    try:
        lc = _int(g.get("levels_completed"), "levels_completed")
        tok = note_tokens(g)
        if tok is None:
            h = g.get("history")
            h = ast.literal_eval(h) if isinstance(h, str) else h
            tok = sum(_int(x.get("generated_tokens"), "generated_tokens") for x in h)
        return lc, tok
    except Exception:
        return None


def game_from_bench(g):
    """[C1] the level costs of one benchmark.json record (E059/E072's rule); raises Corrupt"""
    h = g.get("history")
    if isinstance(h, str):
        try:
            h = ast.literal_eval(h)
        except Exception as exc:
            raise Corrupt(f"history not parseable ({type(exc).__name__})")
    if not isinstance(h, list) or any(not isinstance(x, dict) for x in h):
        raise Corrupt("history is not a list of action records")
    cum, c = [], 0
    for i, x in enumerate(h):
        c += _int(x.get("generated_tokens"), f"history[{i}].generated_tokens")
        cum.append(c)
    n = _int(g.get("number_of_levels"), "number_of_levels")
    lc = _int(g.get("levels_completed"), "levels_completed")
    apl = g.get("actions_per_level")
    if not isinstance(apl, list):
        raise Corrupt("actions_per_level is not a list")
    apl = [_int(a, "actions_per_level") for a in apl]
    if n < 1 or lc > n or len(apl) < lc:
        raise Corrupt(f"levels inconsistent (N {n}, completed {lc}, {len(apl)} per-level counts)")
    costs, idx = [], 0
    for k in range(lc):
        idx += apl[k]
        if idx > len(cum):
            raise Corrupt(f"history ({len(cum)} actions) shorter than level {k + 1}'s completing action {idx}")
        costs.append(cum[idx - 1] if idx > 0 else 0)
    if len(costs) != lc:              # [C10] internal check: the curve at B = infinity equals levels_completed
        raise Corrupt("curve at infinity differs from levels_completed")
    tot = note_tokens(g)
    return dict(N=n, costs=costs, total=tot if tot is not None else (cum[-1] if cum else 0),
                finished_all=lc == n, clock="benchmark", state=g.get("state"))


REQ_FIELDS = ("event", "action", "usage")                               # a request row also carries the whole prompt
EV_FIELDS = ("type", "action_num", "level_completed", "state")         # an events row also carries the board


def read_jsonl(src, rel, fields):
    """rows of a jsonl file, only the named fields kept (request logs hold whole prompts); Corrupt on any unparseable
    line or a row that is not an object [C2: conservative]"""
    rows = []
    with src.text(rel) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                raise Corrupt(f"{rel}: line {i + 1} not JSON")
            if not isinstance(r, dict):
                raise Corrupt(f"{rel}: line {i + 1} not an object")
            rows.append({k: r[k] for k in fields if k in r})
    return rows


def pairs_of(rows):
    """[(request, response or None)] in log order (e110_readout.load_pairs's pairing)"""
    out, i = [], 0
    while i < len(rows):
        if rows[i].get("event") == "request":
            resp = rows[i + 1] if i + 1 < len(rows) and rows[i + 1].get("event") == "response" else None
            out.append((rows[i], resp))
            i += 2 if resp is not None else 1
        else:
            i += 1
    return out


def game_from_logs(req_rows, ev_rows, n_levels):
    """[C2] the level costs from the request log (cumulative usage.completion_tokens through the request whose action range
    holds the completing action) and the events log (level_completed / WIN); raises Corrupt"""
    pairs = pairs_of(req_rows)
    if not pairs:
        raise Corrupt("request log has no requests")
    cum, c = [], 0
    for req, resp in pairs:
        c += _int(((resp or {}).get("usage") or {}).get("completion_tokens"), "usage.completion_tokens")
        cum.append(c)
    acts = [e for e in ev_rows if e.get("type") == "action"]
    nums = [_int(e.get("action_num"), "action_num") for e in acts]
    done = sorted(n for n, e in zip(nums, acts) if e.get("level_completed") or str(e.get("state")).upper() == "WIN")
    for req, _ in pairs:
        _int(req.get("action") or 1, "request.action")
    ranges = R.request_action_ranges(pairs, max(nums or [0]))
    costs = []
    for a in done:
        k = next((k for k, (lo, hi) in enumerate(ranges) if lo <= a <= hi), None)
        if k is None:
            raise Corrupt(f"completing action {a} is in no request's action range")
        costs.append(cum[k])
    if n_levels is None or n_levels < 1:
        raise Corrupt("level count unknown")
    if len(costs) > n_levels:
        raise Corrupt(f"{len(costs)} completions for {n_levels} levels")
    return dict(N=n_levels, costs=costs, total=cum[-1], finished_all=len(costs) == n_levels, clock="requests", state=None)


def levels_from_metadata(src, stem):
    """[C2] len(baseline_actions) from the game's metadata.json: e110_envs/ in the results, else the local datasets"""
    for n in sorted(src.names):
        rel = n[len(src.prefix):] if n.startswith(src.prefix) else n
        parts = rel.split("/")
        if len(parts) >= 3 and parts[0] == "e110_envs" and parts[1] == stem and parts[-1] == "metadata.json":
            try:
                return len(src.json(rel).get("baseline_actions") or []) or None
            except Exception:
                pass
    for base in LOCAL_ENVS:
        d = os.path.join(base, stem)
        if os.path.isdir(d):
            for ver in sorted(os.listdir(d)):
                p = os.path.join(d, ver, "metadata.json")
                if os.path.exists(p):
                    try:
                        return len(json.load(open(p, encoding="utf-8")).get("baseline_actions") or []) or None
                    except Exception:
                        pass
    return None


def costs_agree(a, b):
    if len(a) != len(b):
        return 0, max(len(a), len(b))
    ok = sum(1 for x, y in zip(a, b) if abs(x - y) <= max(1, CROSS_TOL * max(x, y)))
    return ok, len(a)


# ------------------------------------------------------------------------------------------------ loading E110
def load_e110(src, crosscheck=True):
    """valid game records [C1-C3] + an audit of what was missing / corrupt / cross-checked"""
    audit = dict(benchmark=None, manifest=None, listed=[], valid=0, fallback=[], missing=[], corrupt={},
                 unlisted=[], duplicates=[], crosscheck=dict(games=0, completions=0, agree=0, disagree_games=[]),
                 logs_missing=[], logs_corrupt={})
    arm = src.arm
    sources, listed = {}, None
    if src.has("e110/manifest.json"):
        try:
            st = src.json("e110/manifest.json").get("settings") or {}
            sources = {R.stem_of(k): v for k, v in (st.get("sources") or {}).items()}
            listed = list(dict.fromkeys(R.stem_of(x) for x in (st.get("games") or []))) or None
            audit["manifest"] = "ok"
        except Exception as exc:
            audit["manifest"] = f"unreadable ({type(exc).__name__})"
    else:
        audit["manifest"] = "missing"
    bench = {}
    if src.has(arm + "/benchmark.json"):
        try:
            runs = src.json(arm + "/benchmark.json").get("game_runs") or []
            for g in runs:
                gid = str(g.get("game_id"))
                if R.stem_of(gid) in {R.stem_of(k) for k in bench}:
                    audit["duplicates"].append(gid)            # [C3] first record wins
                    continue
                bench[gid] = g
            audit["benchmark"] = f"ok ({len(runs)} records)"
        except Exception as exc:
            audit["benchmark"] = f"unreadable ({type(exc).__name__})"
    else:
        audit["benchmark"] = "missing"
    req_files = {n[:-len("_p0_requests.jsonl")]: n for n in src.list(arm) if n.endswith("_p0_requests.jsonl")}
    ev_files = {n[:-len("_p0_events.jsonl")]: n for n in src.list(arm + "/artifacts") if n.endswith("_p0_events.jsonl")}
    gid_of = {}
    for gid in list(bench) + sorted(set(req_files) | set(ev_files)):
        gid_of.setdefault(R.stem_of(gid), gid)
    if listed is None:
        listed = sorted(gid_of)
    listed_set = set(listed)
    audit["listed"] = list(listed)
    audit["unlisted"] = sorted(s for s in gid_of if s not in listed_set)
    games, repro = [], dict(levels=0, tokens=0, from_benchmark=0, from_fallback=0, missing=0)
    for stem in listed:
        gid = gid_of.get(stem)
        rec, why = None, None
        g = bench.get(gid) if gid else None
        if g is not None:
            try:
                rec = game_from_bench(g)
            except Corrupt as exc:
                why = f"benchmark: {exc}"
        else:
            why = "no benchmark record"
        alt = None
        need_logs = rec is None or crosscheck
        if gid and need_logs:
            rq, ev = req_files.get(gid), ev_files.get(gid)
            if rq and ev:
                try:
                    n_lv = rec["N"] if rec else None
                    if n_lv is None and g is not None:
                        try:
                            n_lv = _int(g.get("number_of_levels"), "number_of_levels") or None
                        except Corrupt:
                            n_lv = None
                    if n_lv is None:
                        n_lv = levels_from_metadata(src, stem)
                    alt = game_from_logs(read_jsonl(src, arm + "/" + rq, REQ_FIELDS),
                                         read_jsonl(src, arm + "/artifacts/" + ev, EV_FIELDS), n_lv)
                except Corrupt as exc:
                    audit["logs_corrupt"][stem] = str(exc)
                except Exception as exc:
                    audit["logs_corrupt"][stem] = f"{type(exc).__name__}: {exc}"[:300]
            else:
                audit["logs_missing"].append(stem)
        if rec is None and alt is not None:
            rec = dict(alt)
            audit["fallback"].append(stem)
        elif rec is not None and alt is not None:
            ok, n = costs_agree(rec["costs"], alt["costs"])
            cc = audit["crosscheck"]
            cc["games"] += 1
            cc["completions"] += n
            cc["agree"] += ok
            if ok < n:
                cc["disagree_games"].append(stem)
        if rec is None:
            audit["missing"].append(stem)
            audit["corrupt"][stem] = why or "no record"
        else:
            if why:
                audit["corrupt"][stem] = why + " -> request-log fallback"
            rec.update(gid=gid, stem=stem, source=R.source_of(stem, sources), alt_costs=(alt or {}).get("costs")
                       if rec["clock"] == "benchmark" and alt is not None else None)
            games.append(rec)
        # [C10] reproduction sums: the benchmark record's own fields when readable, else the fallback record
        bt = bench_totals(g) if g is not None else None
        if bt is not None:
            repro["levels"] += bt[0]
            repro["tokens"] += bt[1]
            repro["from_benchmark"] += 1
        elif rec is not None:
            repro["levels"] += len(rec["costs"])
            repro["tokens"] += rec["total"]
            repro["from_fallback"] += 1
        else:
            repro["missing"] += 1
    audit["valid"] = len(games)
    games.sort(key=lambda g: g["stem"])     # [C8, amendment 1] the bootstrap's order: sorted stems, whatever the file order
    return games, audit, repro


# ------------------------------------------------------------------------------------------------ metric and estimator
def completion(game, b, costs=None):
    """[C6, C7] 100 x sum of the level numbers finished within b / sum of all level numbers"""
    n = game["N"]
    cs = game["costs"] if costs is None else costs
    return 100.0 * sum(k + 1 for k, c in enumerate(cs) if c <= b) / (n * (n + 1) / 2)


def levels_at(game, b, costs=None):
    cs = game["costs"] if costs is None else costs
    return float(sum(1 for c in cs if c <= b))


def elasticity(s_lo, s_hi, b_lo, b_hi):
    """[C8] None if undefined (both 0), +inf if only S(B_lo) = 0"""
    if s_lo <= 0:
        return math.inf if s_hi > 0 else None
    return math.log(s_hi / s_lo) / math.log(b_hi / b_lo)


def estimate(units, b_lo, b_hi, nboot=NBOOT, seed=SEED):
    """units: per-cluster (value at b_lo, value at b_hi) pairs (a game, or a public game's two runs pooled).
    Point e, 90% CI by cluster bootstrap [C8], and the count of dropped resamples."""
    n = len(units)
    if n == 0:
        return dict(n=0, S_lo=None, S_hi=None, e=None, lo=None, hi=None, dropped=0, boot=0)
    s_lo = statistics.fmean(u[0] for u in units)
    s_hi = statistics.fmean(u[1] for u in units)
    e = elasticity(s_lo, s_hi, b_lo, b_hi)
    rng = random.Random(seed)
    lo_v = [u[0] for u in units]
    hi_v = [u[1] for u in units]
    es, dropped = [], 0
    for _ in range(nboot):
        pick = [rng.randrange(n) for _ in range(n)]
        x = elasticity(sum(lo_v[i] for i in pick) / n, sum(hi_v[i] for i in pick) / n, b_lo, b_hi)
        if x is None:
            dropped += 1
        else:
            es.append(x)
    lo = q(es, 0.05) if es else None
    hi = q(es, 0.95) if es else None
    return dict(n=n, S_lo=s_lo, S_hi=s_hi, e=e, lo=lo, hi=hi, dropped=dropped, boot=nboot)


def reached(game, b, count_finished=True):
    """[C4] the game used >= b tokens, or (the closure) finished all its levels"""
    return game["total"] >= b or (count_finished and game["finished_all"])


def choose_window(games, count_finished=True):
    """[C4, C5] the first of 100k / 90k / 80k that >= 30 valid games reached; (B_hi or None, counts)"""
    counts = {b: sum(1 for g in games if reached(g, b, count_finished)) for b in B_HI_CANDIDATES}
    for b in B_HI_CANDIDATES:
        if counts[b] >= MIN_GAMES:
            return b, counts
    return None, counts


def units_of(games, b_lo, b_hi, metric=completion, costs_key=None):
    out = []
    for g in games:
        cs = g.get(costs_key) if costs_key else None
        out.append((metric(g, b_lo, cs), metric(g, b_hi, cs)))
    return out


# ------------------------------------------------------------------------------------------------ reference [C9]
def load_reference(paths=None):
    """{'games': {stem: [rec_E058, rec_E061]}, 'error': None} -- the two public 60-min controls, same 18 games"""
    paths = paths or REF_FILES
    runs = {}
    for name, p in paths.items():
        try:
            gr = json.load(open(p, encoding="utf-8"))["game_runs"]
            runs[name] = {R.stem_of(g["game_id"]): game_from_bench(g) for g in gr}
        except Exception as exc:
            return dict(games=None, error=f"{name} ({p}): {type(exc).__name__}: {exc}"[:300])
    sets = [set(v) for v in runs.values()]
    if len(sets) != 2 or sets[0] != sets[1]:
        return dict(games=None, error=f"the two controls do not hold the same games: {[sorted(s) for s in sets]}")
    names = list(runs)
    return dict(games={s: [runs[names[0]][s], runs[names[1]][s]] for s in sorted(sets[0])}, error=None, runs=names)


def ref_units(ref, b_lo, b_hi, metric=completion):
    """one cluster per public game: the mean of its two runs at each budget (t3_recheck's pooling)"""
    return [((metric(a, b_lo) + metric(b, b_lo)) / 2, (metric(a, b_hi) + metric(b, b_hi)) / 2)
            for a, b in ref["games"].values()]


# ------------------------------------------------------------------------------------------------ analysis
def verdict_of(void, est, ref_est):
    """[C12] first that applies: VOID, HOLDS, WEAKER, INCONCLUSIVE (with the reason)"""
    if void:
        return "VOID", "; ".join(void)
    e, lo, hi = est["e"], est["lo"], est["hi"]
    if e is None or est["S_lo"] is None or est["S_lo"] <= 0:
        return "INCONCLUSIVE", "S(B_lo) = 0: e undefined [C8]"
    if est["boot"] and est["dropped"] > MAX_DROPPED * est["boot"]:
        return "INCONCLUSIVE", f"{est['dropped']} of {est['boot']} resamples undefined (> 1%) [C8]"
    if e >= HOLDS_E and lo is not None and lo > 0:
        return "HOLDS", f"e {e:.3f} >= {HOLDS_E} and CI low {lo:.3f} > 0"
    e_pub = (ref_est or {}).get("e")
    if e < HOLDS_E and hi is not None and e_pub is not None and math.isfinite(e_pub) and hi < e_pub:
        return "WEAKER", f"e {e:.3f} < {HOLDS_E} and CI high {hi:.3f} < e_pub {e_pub:.3f}"
    why = (f"e {e:.3f}, CI {lo:.3f}..{hi:.3f}" if lo is not None else f"e {e:.3f}")
    why += f"; e_pub {e_pub:.3f}" if e_pub is not None else "; reference unavailable (WEAKER cannot apply)"
    return "INCONCLUSIVE", why


def fmt_est(x):
    if x["e"] is None:
        return f"n {x['n']}: e undefined"
    ci = f"{x['lo']:.3f}..{x['hi']:.3f}" if x["lo"] is not None else "n/a"
    return (f"n {x['n']}: S_lo {x['S_lo']:.2f}, S_hi {x['S_hi']:.2f}, e {x['e']:.3f} (90% CI {ci}"
            f"{', ' + str(x['dropped']) + ' resamples dropped' if x['dropped'] else ''})")


def analyse(src_path, out_dir=None, obtained=None, crosscheck=True, figure=True, nboot=NBOOT, ref_paths=None,
            expect_levels=E110_LEVELS, expect_tokens=E110_TOKENS, quiet=False, arm=ARM):
    """the registered analysis end to end; returns the result dict (also written to out_dir). `arm` exists for the
    tests' dress rehearsal on another run's zip (same layout); the CLI always reads arm_e110 (registered)."""
    log = (lambda *a: None) if quiet else say
    src = Source(src_path, arm)
    try:
        games, audit, repro = load_e110(src, crosscheck)
        date_used = obtained or src.obtained_date()
        date_how = "--obtained" if obtained else ("zip modification date" if src.zip is not None else "newest file in the folder")
    finally:
        src.close()
    log("=" * 110)
    log("E155: token elasticity at deployed depth on COMMUNITY games (E110's E098 run, truncated); one run; not hidden games")
    log("=" * 110)
    log(f"input {os.path.abspath(src_path)}; benchmark.json {audit['benchmark']}; manifest {audit['manifest']}; "
        f"obtained {date_used} ({date_how})")
    log(f"games listed {len(audit['listed'])}; valid {audit['valid']} (request-log fallback {len(audit['fallback'])}: "
        f"{audit['fallback'] or '-'}); MISSING {len(audit['missing'])}: {audit['missing'] or '-'}")
    for s, why in sorted(audit["corrupt"].items()):
        log(f"  corrupt/missing record {s}: {why}")
    if audit["unlisted"] or audit["duplicates"]:
        log(f"  ignored: unlisted {audit['unlisted'] or '-'}; duplicate records {audit['duplicates'] or '-'}")
    if audit["logs_missing"] or audit["logs_corrupt"]:
        log(f"  request/events logs missing for {len(audit['logs_missing'])} games {audit['logs_missing'] or ''}; "
            f"corrupt for {len(audit['logs_corrupt'])} {sorted(audit['logs_corrupt']) or ''}")
    cc = audit["crosscheck"]
    if cc["games"]:
        log(f"clock cross-check [C2]: {cc['games']} games with both clocks; {cc['agree']}/{cc['completions']} completing "
            f"actions agree within 1%; games with a disagreement: {cc['disagree_games'] or 'none'}")

    void = []
    # (a) date [C11]
    if date_used > OBTAIN_BY:
        void.append(f"(a) zip obtained {date_used}, after {OBTAIN_BY}")
    # (b) valid games [C3]
    if audit["valid"] < MIN_GAMES:
        void.append(f"(b) {audit['valid']} valid games < {MIN_GAMES}")
    # (d) reproduction [C10]
    dl = abs(repro["levels"] - expect_levels) / expect_levels
    dtk = abs(repro["tokens"] - expect_tokens) / expect_tokens
    repro.update(expect_levels=expect_levels, expect_tokens=expect_tokens, rel_diff_levels=dl, rel_diff_tokens=dtk,
                 ok=dl <= REPRO_TOL and dtk <= REPRO_TOL)
    log(f"reproduction [C10]: {repro['levels']} levels (E110: {expect_levels}, diff {100 * dl:.2f}%), {repro['tokens']:,} tokens "
        f"(E110: {expect_tokens:,}, diff {100 * dtk:.2f}%) from {repro['from_benchmark']} benchmark records + "
        f"{repro['from_fallback']} fallbacks, {repro['missing']} missing -> {'OK' if repro['ok'] else 'FAILS'}")
    if not repro["ok"]:
        void.append(f"(d) totals not reproduced within 1% (levels {100 * dl:.2f}%, tokens {100 * dtk:.2f}%)")
    # (c) window [C4, C5] -- sees only (total tokens, finished-all-levels) per game
    b_hi, counts = choose_window(games, True)
    b_hi_lit, counts_lit = choose_window(games, False)
    log(f"window [C4]: games reaching 100k / 90k / 80k (tokens >= B or all levels finished): "
        f"{counts[100_000]} / {counts[90_000]} / {counts[80_000]} of {len(games)} -> B_hi "
        f"{kfmt(b_hi) if b_hi else 'NONE'}; literal (tokens only): {counts_lit[100_000]} / {counts_lit[90_000]} / "
        f"{counts_lit[80_000]} -> {kfmt(b_hi_lit) if b_hi_lit else 'NONE'} (secondary)")
    if b_hi is None:
        void.append(f"(c) window rule fails: fewer than {MIN_GAMES} games reached 80k")

    ref = load_reference(ref_paths)
    win = b_hi or B_HI_CANDIDATES[0]          # the curve and the descriptive numbers still print on a VOID window
    est = estimate(units_of(games, B_LO, win), B_LO, win, nboot) if games else estimate([], B_LO, win)
    ref_est = (estimate(ref_units(ref, B_LO, win), B_LO, win, nboot) if ref["games"] else None)
    verdict, why = verdict_of(void, est, ref_est)

    # ---------------------------------------------------------------- secondary, reported only [C13]
    sec = {}
    ceil_games = [g for g in games if not (g["finished_all"] and g["costs"] and g["costs"][-1] <= win)]
    sec["ceiling_excluded"] = dict(left_out=len(games) - len(ceil_games),
                                   **estimate(units_of(ceil_games, B_LO, win), B_LO, win, nboot))
    sec["e_over_e_pub"] = (est["e"] / ref_est["e"] if ref_est and est["e"] is not None and ref_est["e"]
                           and math.isfinite(est["e"]) and math.isfinite(ref_est["e"]) else None)
    for s in ("dreamteam", "arc-interactive"):
        sg = [g for g in games if g["source"] == s]
        sec["source_" + s] = estimate(units_of(sg, B_LO, win), B_LO, win, nboot)
    sec["levels_only"] = estimate(units_of(games, B_LO, win, levels_at), B_LO, win, nboot)
    sec["levels_only_reference"] = (estimate(ref_units(ref, B_LO, win, levels_at), B_LO, win, nboot) if ref["games"] else None)
    if b_hi_lit != b_hi:
        sec["literal_window"] = dict(B_hi=b_hi_lit, **(estimate(units_of(games, B_LO, b_hi_lit), B_LO, b_hi_lit, nboot)
                                                       if b_hi_lit else {}))
        if b_hi_lit and ref["games"]:
            sec["literal_window"]["e_pub"] = estimate(ref_units(ref, B_LO, b_hi_lit), B_LO, b_hi_lit, nboot)["e"]
    else:
        sec["literal_window"] = "same window as the deciding one"
    both = [g for g in games if g.get("alt_costs") is not None]
    sec["request_log_clock"] = estimate(units_of(both, B_LO, win, completion, "alt_costs"), B_LO, win, nboot)
    sec["request_log_clock"]["same_games_benchmark_clock_e"] = estimate(units_of(both, B_LO, win), B_LO, win, 0)["e"]

    log("-" * 110)
    log(f"REGISTERED: window {kfmt(B_LO)} -> {kfmt(win)}{'' if b_hi else ' (window rule failed; descriptive only)'}")
    log(f"  E110 community games: {fmt_est(est)}")
    log(f"  reference e_pub (public E058 + E061 60-min controls, pooled): "
        f"{fmt_est(ref_est) if ref_est else 'UNAVAILABLE: ' + str(ref['error'])}")
    log("secondary (reported, not deciding) [C13]:")
    log(f"  (i) ceiling: {sec['ceiling_excluded']['left_out']} games that finished all levels by {kfmt(win)} left out: "
        f"{fmt_est(sec['ceiling_excluded'])}")
    log(f"  (ii) e / e_pub: {sec['e_over_e_pub']:.3f}" if sec["e_over_e_pub"] is not None else "  (ii) e / e_pub: n/a")
    log(f"  (iii) DreamTeam: {fmt_est(sec['source_dreamteam'])}")
    log(f"        ARC-Interactive: {fmt_est(sec['source_arc-interactive'])}")
    log(f"  (iv) levels-only: E110 {fmt_est(sec['levels_only'])}; reference "
        f"{fmt_est(sec['levels_only_reference']) if sec['levels_only_reference'] else 'n/a'}")
    lw = sec["literal_window"]
    log(f"  (v) literal window: {lw}" if isinstance(lw, str) else
        f"  (v) literal window B_hi {kfmt(lw['B_hi']) if lw.get('B_hi') else 'NONE (literal rule would VOID)'}: "
        f"{fmt_est(lw) if lw.get('B_hi') else ''}{'; e_pub ' + format(lw['e_pub'], '.3f') if lw.get('e_pub') is not None else ''}")
    log(f"  (vi) request-log clock ({len(both)} games with both clocks): {fmt_est(sec['request_log_clock'])}; "
        f"benchmark clock on the same games e "
        + (f"{sec['request_log_clock']['same_games_benchmark_clock_e']:.3f}"
           if sec['request_log_clock']['same_games_benchmark_clock_e'] is not None else "n/a"))

    # ---------------------------------------------------------------- Figure 6 data [C14]
    top = max([g["total"] for g in games] + [200_000] +
              [r["total"] for pair in (ref["games"] or {}).values() for r in pair])
    budgets = list(range(0, int(math.ceil(top / CURVE_STEP)) * CURVE_STEP + 1, CURVE_STEP))
    series = [("e110_community_all", games), ("e110_dreamteam", [g for g in games if g["source"] == "dreamteam"]),
              ("e110_arc_interactive", [g for g in games if g["source"] == "arc-interactive"])]
    rows = []
    for name, gs in series:
        if not gs:
            continue
        for b in budgets:
            rows.append(dict(series=name, budget_tokens=b, completion_S=round(statistics.fmean(completion(g, b) for g in gs), 4),
                             levels_per_game=round(statistics.fmean(levels_at(g, b) for g in gs), 4), games=len(gs),
                             games_reaching_budget=sum(1 for g in gs if reached(g, b))))
    if ref["games"]:
        rg = list(ref["games"].values())
        for b in budgets:
            rows.append(dict(series="public_reference_pooled", budget_tokens=b,
                             completion_S=round(statistics.fmean((completion(a, b) + completion(c, b)) / 2 for a, c in rg), 4),
                             levels_per_game=round(statistics.fmean((levels_at(a, b) + levels_at(c, b)) / 2 for a, c in rg), 4),
                             games=len(rg), games_reaching_budget=round(statistics.fmean(
                                 (reached(a, b) + reached(c, b)) / 2 for a, c in rg) * len(rg), 1)))

    result = dict(experiment="E155", label="community games, deployed depth, one run (not hidden games)",
                  input=os.path.abspath(src_path), obtained=str(date_used), obtained_how=date_how,
                  verdict=verdict, verdict_reason=why, void_reasons=void,
                  window=dict(B_lo=B_LO, B_hi=b_hi, counts={str(k): v for k, v in counts.items()},
                              literal_B_hi=b_hi_lit, literal_counts={str(k): v for k, v in counts_lit.items()}),
                  e110=est, reference=ref_est, reference_error=ref["error"], secondary=sec, reproduction=repro,
                  audit=audit, per_game=[dict(stem=g["stem"], gid=g["gid"], source=g["source"], N=g["N"],
                                              level_costs=g["costs"], total_tokens=g["total"], finished_all=g["finished_all"],
                                              clock=g["clock"], request_log_costs=g.get("alt_costs")) for g in games],
                  registered=dict(B_lo=B_LO, B_hi_candidates=B_HI_CANDIDATES, min_games=MIN_GAMES, nboot=NBOOT, seed=SEED,
                                  holds_e=HOLDS_E, e110_levels=E110_LEVELS, e110_tokens=E110_TOKENS, tol=REPRO_TOL,
                                  obtain_by=str(OBTAIN_BY)),
                  nboot_used=nboot)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "e110_truncation_result.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1, default=lambda x: None if isinstance(x, float) and not math.isfinite(x) else str(x))
        with open(os.path.join(out_dir, "fig6_completion_vs_tokens.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["series", "budget_tokens", "completion_S", "levels_per_game", "games",
                                              "games_reaching_budget"])
            w.writeheader()
            w.writerows(rows)
        if figure:
            try:
                make_figure(rows, result, out_dir)
            except Exception as exc:            # the numbers stand without the picture
                log(f"Figure 6 not drawn: {type(exc).__name__}: {exc}")
        log(f"wrote {out_dir}: e110_truncation_result.json, fig6_completion_vs_tokens.csv"
            f"{', fig6_completion_vs_tokens.pdf/.png' if figure else ''}")
    result["verdict_line"] = verdict_line(result)
    log(result["verdict_line"])
    return result


def verdict_line(r):
    est, ref = r["e110"], r["reference"]
    w = r["window"]
    e = (f"e = {est['e']:.3f} (90% CI {est['lo']:.3f}-{est['hi']:.3f})" if est.get("e") is not None and est.get("lo") is not None
         else "e undefined")
    pub = (f"e_pub = {ref['e']:.3f} (90% CI {ref['lo']:.3f}-{ref['hi']:.3f})" if ref and ref.get("e") is not None else "e_pub n/a")
    win = f"{kfmt(w['B_lo'])}-{kfmt(w['B_hi'])}" if w["B_hi"] else "window rule failed"
    return (f"E155 VERDICT: {r['verdict']} | {e}, window {win}, {r['audit']['valid']} of {len(r['audit']['listed'])} games; "
            f"{pub} | {r['verdict_reason']} | community games, deployed depth, one run (not hidden games)")


# ------------------------------------------------------------------------------------------------ Figure 6 [C14]
def make_figure(rows, result, out_dir):
    import logging
    import matplotlib
    matplotlib.use("Agg")
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    # the paper figures' palette and type (tools/paper_figures/make_paper_figures.py; the dataviz reference palette)
    SURF, INK, INK2, GRID, AXIS = "#ffffff", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
    BLUE, ORANGE, LIGHT, DARK = "#2a78d6", "#eb6834", "#86b6ef", "#104281"
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"], "font.size": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2,
        "ytick.color": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "ytick.major.size": 0,
        "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
        "savefig.facecolor": SURF, "legend.frameon": False, "legend.fontsize": 8.5, "pdf.fonttype": 42})
    by = {}
    for r in rows:
        by.setdefault(r["series"], []).append(r)
    style = {"e110_community_all": ("Community games, all (E110, deployed depth)", BLUE, "-", 1.9),
             "e110_dreamteam": ("DreamTeam", LIGHT, "--", 1.2), "e110_arc_interactive": ("ARC-Interactive", DARK, ":", 1.4),
             "public_reference_pooled": ("Public games, two 60-min controls (shallower depth)", ORANGE, "-", 1.6)}
    fig = plt.figure(figsize=(7.0, 4.6))
    gs = GridSpec(2, 1, height_ratios=[3.2, 1], hspace=0.12, left=0.09, right=0.98, top=0.86, bottom=0.11)
    ax, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    w = result["window"]
    for a in (ax, ax2):
        a.grid(True, axis="y", color=GRID, linewidth=0.6)
        a.set_axisbelow(True)
        for b in (w["B_lo"], w["B_hi"] or B_HI_CANDIDATES[0]):
            a.axvline(b / 1e3, color=AXIS, lw=0.9, ls=(0, (2, 2)), zorder=1)
    for name, (lab, col, ls, lw) in style.items():
        rs = by.get(name)
        if not rs:
            continue
        x = [r["budget_tokens"] / 1e3 for r in rs]
        ax.plot(x, [r["completion_S"] for r in rs], color=col, ls=ls, lw=lw, label=f"{lab} (n {rs[0]['games']})", zorder=3)
        if name in ("e110_community_all", "public_reference_pooled"):
            ax2.plot(x, [100 * r["games_reaching_budget"] / r["games"] for r in rs], color=col, ls=ls, lw=lw, zorder=3)
    est, ref = result["e110"], result["reference"]
    txt = (f"e = {est['e']:.2f} (90% CI {est['lo']:.2f}-{est['hi']:.2f})" if est.get("e") is not None and est.get("lo") is not None
           else "e undefined")
    if ref and ref.get("e") is not None:
        txt += f"   public e_pub = {ref['e']:.2f} ({ref['lo']:.2f}-{ref['hi']:.2f})"
    fig.text(0.012, 0.985, "Completion vs tokens per game at deployed depth (community games, one run)", fontsize=11,
             fontweight="semibold", color=INK, va="top")
    fig.text(0.012, 0.935, f"E155 {result['verdict']}: {txt}; window {w['B_lo'] / 1e3:.0f}k-"
             f"{(w['B_hi'] or B_HI_CANDIDATES[0]) / 1e3:.0f}k", fontsize=8.8, color=INK2, va="top")
    ax.set_ylabel("completion within B\n(level-weighted, moves not scored)")
    ax.set_ylim(0, 100)
    ax.tick_params(labelbottom=False)
    ax.legend(loc="upper left", fontsize=8)
    ax2.set_ylabel("games reaching B\nor finished (%)")
    ax2.set_ylim(0, 105)
    ax2.set_xlabel("generated tokens per game, B (thousands)")
    xmax = max(r["budget_tokens"] for r in rows) / 1e3
    for a in (ax, ax2):
        a.set_xlim(0, xmax)
    fig.savefig(os.path.join(out_dir, "fig6_completion_vs_tokens.pdf"))
    fig.savefig(os.path.join(out_dir, "fig6_completion_vs_tokens.png"), dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src", help="e110_results.zip or the unzipped folder")
    ap.add_argument("--out", default=os.path.join(os.getcwd(), "e155_out"))
    ap.add_argument("--obtained", default="", help="download date YYYY-MM-DD (default: the zip's modification date) [C11]")
    ap.add_argument("--no-crosscheck", action="store_true", help="skip the request-log clock on games with a valid record")
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--boot", type=int, default=NBOOT, help="bootstrap resamples (registered: 10,000)")
    a = ap.parse_args(argv)
    if not os.path.exists(a.src):
        say(f"E155: {a.src} not found. Download e110_results.zip first (tools/e155_truncation/README.md, runbook).")
        return 2
    if a.boot != NBOOT:
        say(f"NOTE: --boot {a.boot} differs from the registered {NBOOT}; this run is not the registered analysis.")
    obtained = dt.date.fromisoformat(a.obtained) if a.obtained else None
    r = analyse(a.src, a.out, obtained, not a.no_crosscheck, not a.no_figure, a.boot)
    return 0 if r["verdict"] in ("HOLDS", "WEAKER", "INCONCLUSIVE", "VOID") else 1


if __name__ == "__main__":
    sys.exit(main())
