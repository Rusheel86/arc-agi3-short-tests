"""CPU tests for e110_truncation.py (ledger E155). No GPU, no network, no E110 data (the zip is not public).

Run from anywhere (own runner; exit 0 = all pass):
  PYTHONUTF8=1 python -I tools/e155_truncation/test_e110_truncation.py [--skip-dress]
Release copy of the project's analysis/test_e110_truncation.py; the only changes are the file locations (the E154 smoke
data is a slim copy in data/e154_smoke/, the E081 dress-rehearsal zip is optional: set E155_DRESS_ZIP to its path),
the interpreter (sys.executable) and game_curve.py in place of k_killtests.py (the same function, copied unchanged).

Synthetic results are written in the real layout of e110_results.zip (analysis/build_e110_nb.py's results cell zips
/kaggle/working: arm_e110/benchmark.json, arm_e110/<gid>_p0_requests.jsonl, arm_e110/artifacts/<gid>_p0_events.jsonl,
e110/manifest.json, e110_envs/<stem>/<ver>/metadata.json), with the harness's token booking (a reply's generated tokens
on its first action, tool-only replies rolling over to the next action, tokens after the last action only in the
solver_note total). Test 1 proves the layout by reading the fixtures with analysis/e110_readout.py itself.

  1. layout      fixtures read by e110_readout (files_of, summarize_run): same games, levels, tokens
  2. metric      completion = game_curve.game_curve with the efficiency factor 1 on real E058 records; the cut rule
  3. known e     an exact design (e = log2 1.5) and power-law fixtures (e = 0.3 / 0.6 / 1.0) are recovered
  4. estimator   bootstrap determinism, collapse, +inf and dropped resamples, t3_recheck's percentile rule
  5. window      100k / 90k / 80k / VOID; finished games count (C4); the literal rule as a secondary line
  6. robustness  zip = folder (read in place), extra top folder, repr-string history, corrupt / missing benchmark
                 records and files -> request-log fallback, missing logs, corrupt lines, unlisted / duplicate games
  7. verdicts    HOLDS, WEAKER, INCONCLUSIVE (three ways), VOID (a)-(d)
  8. outputs     result JSON, Figure 6 CSV / PDF / PNG; the CLI with -I (exit code, last line)
  9. real data   the real reference files load (18 + 18 public games); the E154 smoke outputs' two clocks agree;
                 dress rehearsal on the real E081 results zip (same layout, read in place; --skip-dress skips it)
"""
import contextlib
import hashlib
import io
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import e110_readout as R  # noqa: E402
import e110_truncation as T  # noqa: E402

PY = sys.executable
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  -- {str(detail)[:700]}"), flush=True)


def close(a, b, tol=1e-9):
    return a is not None and b is not None and abs(a - b) <= tol


# ================================================================================================ fixture builder
def spec(stem, N, costs, total=None, source=None, state=None):
    """one game: level completion costs (cumulative generated tokens at the completing action), total tokens"""
    costs = list(costs)
    assert costs == sorted(costs) and len(costs) <= N
    total = total if total is not None else (costs[-1] if costs else 0)
    assert total >= (costs[-1] if costs else 0)
    return dict(stem=stem, N=N, costs=costs, total=total,
                source=source or ("dreamteam" if len(stem) == 6 else "arc-interactive"), state=state)


def build_game(s):
    """requests (tokens, n_actions, which action completes a level) -> history, request log, events, benchmark record"""
    reqs, prev = [], 0
    for k, c in enumerate(s["costs"]):
        d = c - prev
        prev = c
        if d >= 3:
            reqs.append((d // 3, 0, None))                 # a tool-only reply: its tokens roll over to the next action
            reqs.append((d - d // 3, 2, k + 1))            # the reply whose 2nd action completes level k + 1
        else:
            reqs.append((d, 1, k + 1))
    rest = s["total"] - prev
    finished = len(s["costs"]) == s["N"]
    if not finished:
        reqs.append((rest // 2, 3, None))                  # actions on the unfinished level
        reqs.append((rest - rest // 2, 0, None))           # tokens after the last action (only in the solver_note total)
    elif rest:
        reqs.append((rest, 0, None))
    history, req_rows, events, apl = [], [], [], []
    pending, n_act, lvl_actions = 0, 0, 0
    events.append(dict(type="initial", action_num=0, level=1, state="NOT_FINISHED", board=[[0]], action_display="RESET",
                       score=0, analysis_step=None))
    level = 1
    for i, (tok, na, completes) in enumerate(reqs):
        req_rows.append(dict(event="request", action=n_act + 1, analysis_step=i + 1, request_index_within_turn=1,
                             tool_choice="auto", chat_template_kwargs={"preserve_thinking": True},
                             messages=[{"role": "user", "content": f"step {i}"}]))
        req_rows.append(dict(event="response", action=n_act + 1, analysis_step=i + 1, request_index_within_turn=1,
                             usage=dict(prompt_tokens=1000, completion_tokens=tok, total_tokens=1000 + tok),
                             finish_reason="tool_calls"))
        pending += tok
        for j in range(na):
            n_act += 1
            lvl_actions += 1
            history.append({"action": {"id": "ACTION1", "data": {}}, "generated_tokens": pending if j == 0 else 0,
                            "uncached_input_tokens": 0, "wallclock_seconds": 1.0})
            if j == 0:
                pending = 0
            done = completes is not None and j == na - 1
            win = done and completes == s["N"]
            if done:
                apl.append(lvl_actions)
                lvl_actions = 0
                level += 1
            events.append(dict(type="action", action_num=n_act, analysis_step=i + 1, action_name="ACTION1",
                               action_display="UP", level=min(level, s["N"]), level_completed=done, game_over=False,
                               state="WIN" if win else "NOT_FINISHED", board=[[n_act % 7]], reward=0.0))
    if not finished:
        apl.append(lvl_actions)
    apl += [0] * (s["N"] - len(apl))
    gid = f"{s['stem']}-{hashlib.md5(s['stem'].encode()).hexdigest()[:8]}"
    tot = sum(t for t, _, _ in reqs)
    assert tot == s["total"]
    rec = dict(game_id=gid, number_of_levels=s["N"], base_actions_per_level=[10] * s["N"], hint=None,
               state=s["state"] or ("won" if finished else "gave_up"), history=history, record_intermediate_states=True,
               actions_per_level=apl, levels_completed=len(s["costs"]), final_score=0.0, solver_note=f"tokens={tot}",
               solver_analysis_html=f"solver_analysis/{gid}_p0.html", final_generated_tokens=0,
               final_uncached_input_tokens=0, final_wallclock_seconds=3600.0, started_at="2026-10-14T17:40:00")
    return gid, rec, req_rows, events


def write_results(root, specs, history_as_str=False, manifest=True, benchmark=True, envs=True, extra_bench=(),
                  dup_bench=(), drop_requests=(), drop_events=(), corrupt_history=(), corrupt_request_line=(),
                  bench_text=None, top=None):
    """a results folder in e110_results.zip's layout; returns its path"""
    base = os.path.join(root, top) if top else root
    arm = os.path.join(base, "arm_e110")
    os.makedirs(os.path.join(arm, "artifacts"), exist_ok=True)
    runs = []
    for s in list(specs) + list(extra_bench):
        gid, rec, req_rows, events = build_game(s)
        if s in extra_bench:
            runs.append(rec)
            continue
        if s["stem"] in corrupt_history:
            rec = dict(rec, history="[{'action': {'id': 'ACTION1'}, 'generated_tokens': 5")   # truncated repr
        elif history_as_str:
            rec = dict(rec, history=repr(rec["history"]))
        runs.append(rec)
        if s["stem"] in dup_bench:
            runs.append(dict(rec, levels_completed=0, actions_per_level=[0] * s["N"]))      # a later duplicate: ignored
        if s["stem"] not in drop_requests:
            with open(os.path.join(arm, f"{gid}_p0_requests.jsonl"), "w", encoding="utf-8") as f:
                for i, r in enumerate(req_rows):
                    f.write(json.dumps(r) + "\n")
                    if s["stem"] in corrupt_request_line and i == 1:
                        f.write('{"event": "response", "usage": {"completion_tok\n')
        if s["stem"] not in drop_events:
            with open(os.path.join(arm, "artifacts", f"{gid}_p0_events.jsonl"), "w", encoding="utf-8") as f:
                for e in events:
                    f.write(json.dumps(e) + "\n")
        if envs:
            d = os.path.join(base, "e110_envs", s["stem"], "0000abcd")
            os.makedirs(d, exist_ok=True)
            json.dump(dict(game_id=gid, baseline_actions=[10] * s["N"]), open(os.path.join(d, "metadata.json"), "w"))
    if benchmark:
        text = bench_text if bench_text is not None else json.dumps(dict(
            label="duck-harness-kaggle", n_passes=1, solver_label="duck-harness", start_time="2026-10-14T17:40:00",
            end_time="2026-10-14T19:30:00", game_weights=None, periodic_save_interval_s=600.0, game_runs=runs))
        open(os.path.join(arm, "benchmark.json"), "w", encoding="utf-8").write(text)
    if manifest:
        os.makedirs(os.path.join(base, "e110"), exist_ok=True)
        json.dump(dict(experiment="E110", settings=dict(games=[s["stem"] for s in specs],
                                                        sources={s["stem"]: s["source"] for s in specs})),
                  open(os.path.join(base, "e110", "manifest.json"), "w"))
    return root


def zip_folder(folder, zpath):
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for d, _, fs in os.walk(folder):
            for n in fs:
                p = os.path.join(d, n)
                z.write(p, os.path.relpath(p, folder).replace(os.sep, "/"))
    return zpath


def write_ref(root, specs_a, specs_b=None):
    """two synthetic 'public control' benchmark.json files (same stems) -> {name: path}"""
    out = {}
    for name, ss in (("E058", specs_a), ("E061", specs_b or specs_a)):
        p = os.path.join(root, f"ref_{name}.json")
        json.dump(dict(game_runs=[build_game(s)[1] for s in ss]), open(p, "w"))
        out[name] = p
    return out


def totals(specs):
    return sum(len(s["costs"]) for s in specs), sum(s["total"] for s in specs)


def run(folder, specs, ref, **kw):
    lv, tok = totals(specs)
    kw.setdefault("expect_levels", lv)
    kw.setdefault("expect_tokens", tok)
    kw.setdefault("nboot", 2000)
    kw.setdefault("figure", False)
    kw.setdefault("obtained", T.dt.date(2026, 10, 29))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = T.analyse(folder, kw.pop("out", None), ref_paths=ref, **kw)
    r["_log"] = buf.getvalue()
    return r


def realistic(seed=7, n_dt=25, n_ai=13):
    """38 games like E110's (8-level DreamTeam, 5-6-level ARC-Interactive), mixed difficulty; some finish early"""
    rng = random.Random(seed)
    out = []
    for i in range(n_dt + n_ai):
        dt_ = i < n_dt
        stem = f"dt{i:04d}" if dt_ else f"a{i:03d}"
        N = 8 if dt_ else rng.choice((5, 6))
        rate = rng.uniform(4_000, 40_000)          # tokens per level, rising with depth
        costs, c = [], 0
        for k in range(N):
            c += int(rate * (1 + 0.35 * k) * rng.uniform(0.5, 1.6))
            if c > 125_000:
                break
            costs.append(c)
        total = costs[-1] if len(costs) == N else rng.randint(max(costs[-1] if costs else 0, 101_000), 135_000)
        out.append(spec(stem, N, costs, total, "dreamteam" if dt_ else "arc-interactive"))
    return out


def independent_e(specs, b_lo, b_hi):
    """the registered estimator written out from the ledger text, without the script"""
    def comp(s, b):
        return 100 * sum(k + 1 for k, c in enumerate(s["costs"]) if c <= b) / (s["N"] * (s["N"] + 1) / 2)
    lo = sum(comp(s, b_lo) for s in specs) / len(specs)
    hi = sum(comp(s, b_hi) for s in specs) / len(specs)
    return math.log(hi / lo) / math.log(b_hi / b_lo)


def ref_power(e, n=18, seed=1):
    return power_law(e, n, prefix="r", seed=seed)


def power_law(e, n, prefix="p", N=8, total=120_000, seed=0):
    """S(B) proportional to B^e in expectation for B <= B_hi: level k costs s_j * w(k)^(1/e), s_j = B_max u_j^(1/e),
    u_j uniform quantiles, B_max chosen so that (B_hi / B_max)^e = 0.8 / W (every completion probability < 1)"""
    W = N * (N + 1) / 2
    bmax = 100_000 * (W / 0.8) ** (1 / e)
    out = []
    for j in range(n):
        u = (j + 0.5) / n
        s = bmax * u ** (1 / e)
        costs = [int(round(s * (k * (k + 1) / (N * (N + 1))) ** (1 / e))) for k in range(1, N + 1)]
        costs = [c for c in costs if c <= total]
        out.append(spec(f"{prefix}{j:05d}", N, costs, total, "dreamteam"))
    random.Random(seed).shuffle(out)
    return out


# ================================================================================================ 1. layout
def t_layout(tmp):
    specs = realistic()
    root = write_results(os.path.join(tmp, "layout"), specs)
    arm = os.path.join(root, "arm_e110")
    files = R.files_of(arm)
    summ = R.summarize_run(arm, {s["stem"]: s["source"] for s in specs})["totals"]["all"]
    lv, tok = totals(specs)
    check("1 layout: e110_readout.files_of finds every game's events + requests in the fixture",
          len(files) == 38 and all(f.get("events") and f.get("requests") for f in files.values()), len(files))
    check("1 layout: e110_readout.summarize_run reads the fixture: 38 games, its levels and tokens",
          summ["games"] == 38 and summ["levels"] == lv and summ["tokens"] == tok, (summ["games"], summ["levels"], lv, summ["tokens"], tok))
    pairs = R.load_pairs(os.path.join(arm, sorted(n for n in os.listdir(arm) if n.endswith("_requests.jsonl"))[0]))
    check("1 layout: request log pairs request/response rows like the real logs (usage.completion_tokens)",
          pairs and all(r is not None and "completion_tokens" in r["usage"] for _, r in pairs))
    return specs


# ================================================================================================ 2. metric
def t_metric(tmp):
    import game_curve as kk   # the project's k_killtests.game_curve, copied unchanged
    bad, n = [], 0
    for p in T.REF_FILES.values():
        for g in json.load(open(p, encoding="utf-8"))["game_runs"]:
            g2 = dict(g, base_actions_per_level=[max(1, a) for a in g["actions_per_level"]])   # efficiency factor 1
            score = kk.game_curve(g2)[0]
            rec = T.game_from_bench(g)
            for b in (20e3, 50e3, 80e3, 90e3, 100e3, 140e3, 1e18):
                n += 1
                if abs(score(b) - T.completion(rec, b)) > 1e-9:
                    bad.append((g["game_id"], b, score(b), T.completion(rec, b)))
    check(f"2 metric: completion = E072's k_killtests.game_curve with the move-efficiency factor 1 ({n} game-budgets, "
          "real E058 + E061 records)", n > 0 and not bad, bad[:3])
    g = dict(N=3, costs=[10, 50_000, 50_001])
    check("2 metric: a level counts at B iff its cost <= B (ties count); finished after the cut = 0 [C6]",
          close(T.completion(g, 50_000), 50.0) and close(T.completion(g, 50_001), 100.0) and close(T.completion(g, 9), 0.0))
    g = dict(N=8, costs=[20_000])
    check("2 metric: a game that ended before B keeps its final value [C6]",
          close(T.completion(g, 50_000), T.completion(g, 100_000)) and close(T.completion(g, 100_000), 100 / 36))
    check("2 metric: levels-only counts levels", T.levels_at(dict(N=6, costs=[5, 7, 60_000]), 50_000) == 2)


# ================================================================================================ 3. known elasticity
def t_known(tmp):
    # exact design: 20 games 2 -> 3 levels, 10 games 3 -> 4, 8 games stuck at 4 (8-level games): S_hi / S_lo = 300 / 200
    specs = []
    for i in range(38):
        if i < 20:
            costs = [10_000, 30_000, 80_000]
        elif i < 30:
            costs = [10_000, 20_000, 40_000, 95_000]
        else:
            costs = [5_000, 10_000, 20_000, 45_000]
        specs.append(spec(f"x{i:05d}", 8, costs, 120_000))
    random.Random(3).shuffle(specs)
    root = write_results(os.path.join(tmp, "exact"), specs)
    ref = write_ref(tmp, power_law(1.0, 18, "r"))
    r = run(root, specs, ref)
    want = math.log(1.5) / math.log(2)
    check(f"3 known e: the exact design gives e = log2(1.5) = {want:.6f} to 1e-12",
          close(r["e110"]["e"], want, 1e-12), r["e110"]["e"])
    check("3 known e: the independent re-implementation agrees on the realistic fixture",
          close(run(write_results(os.path.join(tmp, "real1"), realistic()), realistic(), ref)["e110"]["e"],
                independent_e(realistic(), 50_000, 100_000), 1e-12))
    for e_true in (0.3, 0.6, 1.0):
        specs = power_law(e_true, 400)
        root = write_results(os.path.join(tmp, f"pow{e_true}"), specs)
        r = run(root, specs, ref, nboot=500, crosscheck=False)
        x = r["e110"]
        check(f"3 known e: power-law fixture (400 games) e_true {e_true} recovered within 0.03 "
              f"(got {x['e']:.4f}, 90% CI {x['lo']:.3f}..{x['hi']:.3f}), CI covers it",
              abs(x["e"] - e_true) < 0.03 and x["lo"] <= e_true <= x["hi"], x)


# ================================================================================================ 4. estimator
def t_estimator(tmp):
    u = [(1.0 + (i % 5), 2.0 + 2 * (i % 5) + (i % 3)) for i in range(38)]
    a, b = T.estimate(u, 50_000, 100_000), T.estimate(u, 50_000, 100_000)
    check("4 estimator: bootstrap is deterministic (random.Random(110), 10,000 resamples)",
          a == b and a["boot"] == 10_000 and a["lo"] <= a["e"] <= a["hi"], (a, b))
    c = T.estimate([(3.0, 6.0)] * 38, 50_000, 100_000)
    check("4 estimator: identical games -> the CI collapses on the point (e = 1)", close(c["e"], 1) and close(c["lo"], 1) and close(c["hi"], 1), c)
    d = T.estimate([(0.0, 0.0)] * 37 + [(1.0, 10.0)], 50_000, 100_000, nboot=4000)
    p_miss = (37 / 38) ** 38
    check(f"4 estimator: resamples with S*(B_lo) = 0 = S*(B_hi) are dropped and counted (~{p_miss:.0%} expected)",
          abs(d["dropped"] / 4000 - p_miss) < 0.03, d)
    f = T.estimate([(0.0, 0.0)] * 36 + [(0.0, 5.0), (1.0, 2.0)], 50_000, 100_000, nboot=4000)
    check("4 estimator: a resample with S*(B_lo) = 0 < S*(B_hi) gives e* = +inf (kept)", f["hi"] == math.inf, f)
    check("4 estimator: elasticity() edges (None when both 0, inf when only S_lo is 0)",
          T.elasticity(0, 0, 1, 2) is None and T.elasticity(0, 1, 1, 2) == math.inf and close(T.elasticity(1, 2, 1, 2), 1))
    check("4 estimator: t3_recheck's percentile rule (index round(f (n - 1)))",
          T.q(list(range(101)), 0.05) == 5 and T.q(list(range(101)), 0.95) == 95 and T.q([3, 1, 2], 0.5) == 2)


# ================================================================================================ 5. window
def t_window(tmp):
    def g(total, finished=False):
        return dict(total=total, finished_all=finished)
    # the records carry only (total tokens, finished-all-levels): the window cannot see any level cost [C4]
    cases = [("all >= 100k -> 100k (records with total tokens and finished flag only)", [g(120_000)] * 38, 100_000),
             ("29 at 100k, 30 at 90k -> 90k", [g(120_000)] * 29 + [g(95_000)] * 1 + [g(60_000)] * 8, 90_000),
             ("29 at 90k, 30 at 80k -> 80k", [g(120_000)] * 29 + [g(85_000)] + [g(60_000)] * 8, 80_000),
             ("29 at 80k -> VOID", [g(120_000)] * 29 + [g(79_999)] * 9, None),
             ("exactly 30 of 30 valid at 100k -> 100k [C5]", [g(100_000)] * 30, 100_000)]
    for name, gs, want in cases:
        got, counts = T.choose_window(gs)
        check(f"5 window: {name}", got == want, (got, counts))
    gs = [g(120_000)] * 25 + [g(60_000, True)] * 13
    got, _ = T.choose_window(gs, True)
    lit, _ = T.choose_window(gs, False)
    check("5 window: a game that finished all its levels counts as reaching every budget [C4]; the literal rule would VOID",
          got == 100_000 and lit is None, (got, lit))
    # end to end: the literal variant is reported as a secondary line
    specs = [spec(f"w{i:05d}", 8, [10_000, 40_000, 90_000], 120_000) for i in range(25)] + \
            [spec(f"f{i:03d}", 5, [5_000, 10_000, 20_000, 30_000, 60_000], None, "arc-interactive") for i in range(13)]
    root = write_results(os.path.join(tmp, "win"), specs)
    r = run(root, specs, write_ref(tmp, power_law(1.0, 18, "r")))
    check("5 window: end to end: deciding window 100k, literal window NONE reported as secondary (v), not VOID",
          r["window"]["B_hi"] == 100_000 and r["window"]["literal_B_hi"] is None and r["verdict"] != "VOID"
          and isinstance(r["secondary"]["literal_window"], dict), (r["window"], r["verdict"]))


# ================================================================================================ 6. robustness
def t_robust(tmp, base_specs):
    ref = write_ref(tmp, power_law(1.0, 18, "r"))
    clean_dir = write_results(os.path.join(tmp, "clean"), base_specs)
    clean = run(clean_dir, base_specs, ref)
    e0 = clean["e110"]["e"]
    check("6 robust: clean fixture: 38 valid, 0 fallback, clocks agree on every completion",
          clean["audit"]["valid"] == 38 and not clean["audit"]["fallback"]
          and clean["audit"]["crosscheck"]["agree"] == clean["audit"]["crosscheck"]["completions"] > 0, clean["audit"]["crosscheck"])
    zp = zip_folder(clean_dir, os.path.join(tmp, "clean.zip"))
    before = set(os.listdir(tmp))
    rz = run(zp, base_specs, ref)
    check("6 robust: the zip read in place gives the same numbers as the folder, and extracts nothing",
          rz["e110"] == clean["e110"] and rz["audit"]["valid"] == 38 and set(os.listdir(tmp)) == before, rz["e110"])
    topd = write_results(os.path.join(tmp, "top"), base_specs, top="e110_results")
    rt = run(zip_folder(topd, os.path.join(tmp, "top.zip")), base_specs, ref)
    check("6 robust: a zip with one extra top folder is read the same", rt["e110"] == clean["e110"], rt["audit"]["benchmark"])
    rs = run(write_results(os.path.join(tmp, "repr"), base_specs, history_as_str=True), base_specs, ref)
    check("6 robust: history stored as a repr string (as e110_readout allows) gives the same numbers", rs["e110"] == clean["e110"])
    bad = [s["stem"] for s in base_specs[:5]]
    rc = run(write_results(os.path.join(tmp, "corrupt"), base_specs, corrupt_history=bad), base_specs, ref)
    check("6 robust: 5 corrupt benchmark histories -> request-log fallback for those 5, same e, reported [C2]",
          sorted(rc["audit"]["fallback"]) == sorted(bad) and close(rc["e110"]["e"], e0, 1e-12)
          and all(s in rc["audit"]["corrupt"] for s in bad) and "fallback" in rc["_log"], rc["audit"]["fallback"])
    rn = run(write_results(os.path.join(tmp, "nobench"), base_specs, benchmark=False), base_specs, ref)
    check("6 robust: benchmark.json missing -> all 38 from the logs, level counts from e110_envs metadata, same e; "
          "reproduction from the fallbacks", rn["audit"]["benchmark"] == "missing" and len(rn["audit"]["fallback"]) == 38
          and close(rn["e110"]["e"], e0, 1e-12) and rn["reproduction"]["ok"], (rn["audit"]["benchmark"], rn["reproduction"]))
    rb = run(write_results(os.path.join(tmp, "badjson"), base_specs, bench_text='{"game_runs": [{"game_id": "x'), base_specs, ref)
    check("6 robust: benchmark.json truncated -> 'unreadable', fallback for all, same e",
          rb["audit"]["benchmark"].startswith("unreadable") and close(rb["e110"]["e"], e0, 1e-12), rb["audit"]["benchmark"])
    rne = run(write_results(os.path.join(tmp, "nobench_noenv"), base_specs, benchmark=False, envs=False), base_specs, ref)
    check("6 robust: no benchmark.json and no metadata -> level counts unknown -> games MISSING, VOID (b)",
          rne["audit"]["valid"] == 0 and len(rne["audit"]["missing"]) == 38 and rne["verdict"] == "VOID"
          and any(v.startswith("(b)") for v in rne["void_reasons"]), rne["void_reasons"])
    drop = [s["stem"] for s in base_specs[5:9]]
    rd = run(write_results(os.path.join(tmp, "nolog"), base_specs, drop_requests=drop[:2], drop_events=drop[2:]), base_specs, ref)
    check("6 robust: missing request or events logs for 4 games with valid records -> still valid, listed as logs missing",
          rd["audit"]["valid"] == 38 and sorted(rd["audit"]["logs_missing"]) == sorted(drop) and rd["e110"] == clean["e110"],
          rd["audit"]["logs_missing"])
    rl = run(write_results(os.path.join(tmp, "badline"), base_specs, corrupt_request_line=[base_specs[9]["stem"]]), base_specs, ref)
    check("6 robust: a corrupt line in a request log -> that log rejected and reported; the game stays valid via benchmark",
          base_specs[9]["stem"] in rl["audit"]["logs_corrupt"] and rl["audit"]["valid"] == 38 and rl["e110"] == clean["e110"])
    both = base_specs[10]["stem"]
    rm = run(write_results(os.path.join(tmp, "both"), base_specs, corrupt_history=[both], drop_requests=[both]), base_specs, ref)
    check("6 robust: corrupt record AND missing log -> the game is MISSING (37 valid), counted and printed",
          rm["audit"]["missing"] == [both] and rm["audit"]["valid"] == 37 and "MISSING 1" in rm["_log"], rm["audit"]["missing"])
    extra = [spec("zz9999", 6, [1000], 2000)]
    ru = run(write_results(os.path.join(tmp, "extra"), base_specs, extra_bench=extra, dup_bench=[base_specs[0]["stem"]]),
             base_specs, ref)
    check("6 robust: an unlisted game is ignored, a duplicated record uses the first one [C3]",
          ru["audit"]["unlisted"] == ["zz9999"] and len(ru["audit"]["duplicates"]) == 1 and ru["e110"] == clean["e110"],
          (ru["audit"]["unlisted"], ru["audit"]["duplicates"]))
    zeros = base_specs + [spec(f"z{i:05d}", 8, [], 110_000) for i in range(6)]
    rz0 = run(write_results(os.path.join(tmp, "zeros"), zeros), zeros, ref)
    check("6 robust: games with 0 levels stay in at 0 and leave e unchanged (both budgets scale alike) [C3]",
          close(rz0["e110"]["e"], e0, 1e-12) and rz0["e110"]["n"] == 44)
    rnm = run(write_results(os.path.join(tmp, "nomani"), base_specs, manifest=False), base_specs, ref)
    check("6 robust: no manifest -> every game in the results is listed; sources from the stem rule",
          rnm["audit"]["manifest"] == "missing" and rnm["audit"]["valid"] == 38 and rnm["e110"] == clean["e110"])
    return clean, ref


# ================================================================================================ 7. verdicts
def t_verdicts(tmp):
    ref1 = write_ref(tmp, power_law(1.0, 18, "r"))
    sp = power_law(0.6, 120, "h")
    r = run(write_results(os.path.join(tmp, "holds"), sp), sp, ref1)
    check(f"7 verdict: HOLDS when e >= 0.33 and CI low > 0 (e {r['e110']['e']:.3f}, low {r['e110']['lo']:.3f})",
          r["verdict"] == "HOLDS", (r["verdict"], r["verdict_reason"]))
    flat = [spec(f"q{i:05d}", 8, [5_000, 10_000, 20_000] + ([70_000] if i < 6 else []), 120_000) for i in range(38)]
    root = write_results(os.path.join(tmp, "flat"), flat)
    rw = run(root, flat, ref1)
    check(f"7 verdict: WEAKER when e < 0.33 and CI high < e_pub (e {rw['e110']['e']:.3f}, high {rw['e110']['hi']:.3f}, "
          f"e_pub {rw['reference']['e']:.3f})", rw["verdict"] == "WEAKER", (rw["verdict"], rw["verdict_reason"]))
    flat_ref = [spec(f"s{i:05d}", 8, [5_000, 10_000, 20_000] + ([70_000] if i < 3 else []), 200_000) for i in range(18)]
    ri = run(root, flat, write_ref(tmp, flat_ref))
    check("7 verdict: INCONCLUSIVE when e < 0.33 but CI high >= e_pub",
          ri["verdict"] == "INCONCLUSIVE" and ri["e110"]["hi"] >= ri["reference"]["e"], (ri["verdict"], ri["verdict_reason"]))
    one = [spec(f"o{i:05d}", 8, [20_000] + ([30_000, 40_000, 50_000, 60_000, 70_000, 80_000, 90_000] if i == 0 else []), 120_000)
           for i in range(38)]
    ro = run(write_results(os.path.join(tmp, "one"), one), one, ref1)
    check(f"7 verdict: INCONCLUSIVE when e >= 0.33 but CI low = 0 (one game carries it; e {ro['e110']['e']:.3f})",
          ro["verdict"] == "INCONCLUSIVE" and ro["e110"]["e"] >= 0.33 and ro["e110"]["lo"] <= 0, (ro["verdict"], ro["e110"]))
    late = [spec(f"l{i:05d}", 8, [60_000, 90_000], 120_000) for i in range(38)]
    rl = run(write_results(os.path.join(tmp, "late"), late), late, ref1)
    check("7 verdict: INCONCLUSIVE when S(B_lo) = 0 (e undefined) [C8]",
          rl["verdict"] == "INCONCLUSIVE" and "S(B_lo) = 0" in rl["verdict_reason"], rl["verdict_reason"])
    sparse = [spec(f"d{i:05d}", 8, ([20_000, 60_000] if i == 0 else []), 120_000) for i in range(38)]
    rs = run(write_results(os.path.join(tmp, "sparse"), sparse), sparse, ref1)
    check("7 verdict: INCONCLUSIVE when more than 1% of resamples are undefined [C8]",
          rs["verdict"] == "INCONCLUSIVE" and "resamples undefined" in rs["verdict_reason"], rs["verdict_reason"])
    rnr = run(root, flat, {"E058": os.path.join(tmp, "nope.json"), "E061": os.path.join(tmp, "nope.json")})
    check("7 verdict: reference unreadable -> WEAKER cannot apply -> INCONCLUSIVE, error reported [C9]",
          rnr["verdict"] == "INCONCLUSIVE" and rnr["reference"] is None and rnr["reference_error"], rnr["verdict_reason"])
    rh = run(write_results(os.path.join(tmp, "holds2"), sp), sp, {"E058": os.path.join(tmp, "nope.json"), "E061": ref1["E061"]})
    check("7 verdict: HOLDS needs no reference", rh["verdict"] == "HOLDS")
    # VOID (a)-(d)
    ra = run(os.path.join(tmp, "holds"), sp, ref1, obtained=T.dt.date(2026, 11, 5))
    check("7 verdict: VOID (a) when the zip was obtained after 2026-11-04 [C11]",
          ra["verdict"] == "VOID" and ra["void_reasons"][0].startswith("(a)"), ra["void_reasons"])
    r4 = run(os.path.join(tmp, "holds"), sp, ref1, obtained=T.dt.date(2026, 11, 4))
    check("7 verdict: 2026-11-04 itself is in time", r4["verdict"] == "HOLDS")
    few = sp[:29]
    rf = run(write_results(os.path.join(tmp, "few"), few), few, ref1)
    check("7 verdict: VOID (b) with 29 valid games", rf["verdict"] == "VOID" and any(v.startswith("(b)") for v in rf["void_reasons"]),
          rf["void_reasons"])
    thirty = sp[:30]
    r30 = run(write_results(os.path.join(tmp, "thirty"), thirty), thirty, ref1)
    check("7 verdict: 30 valid games pass (b)", not any(v.startswith("(b)") for v in r30["void_reasons"]), r30["void_reasons"])
    short = [spec(f"c{i:05d}", 8, [10_000, 30_000], 70_000) for i in range(38)]
    rc = run(write_results(os.path.join(tmp, "short"), short), short, ref1)
    check("7 verdict: VOID (c) when fewer than 30 games reach 80k",
          rc["verdict"] == "VOID" and any(v.startswith("(c)") for v in rc["void_reasons"]), rc["void_reasons"])
    lv, tok = totals(sp)
    rd = run(os.path.join(tmp, "holds"), sp, ref1, expect_tokens=int(tok / 1.012))
    rd_ok = run(os.path.join(tmp, "holds"), sp, ref1, expect_tokens=int(tok / 1.009))
    check("7 verdict: VOID (d) when the totals miss by > 1% (tokens +1.2%); +0.9% passes",
          rd["verdict"] == "VOID" and any(v.startswith("(d)") for v in rd["void_reasons"]) and rd_ok["verdict"] == "HOLDS",
          (rd["void_reasons"], rd_ok["verdict"]))
    rlv = run(os.path.join(tmp, "holds"), sp, ref1, expect_levels=lv + max(2, int(0.011 * lv) + 1))
    check("7 verdict: VOID (d) when the level total misses by > 1%", rlv["verdict"] == "VOID")
    check("7 verdict: the registered constants are the ledger's (150 levels, 3,881,160 tokens, 50k, 100/90/80k, 30, "
          "10,000, seed 110, 0.33, 1%, 2026-11-04)",
          (T.E110_LEVELS, T.E110_TOKENS, T.B_LO, T.B_HI_CANDIDATES, T.MIN_GAMES, T.NBOOT, T.SEED, T.HOLDS_E, T.REPRO_TOL,
           str(T.OBTAIN_BY)) == (150, 3_881_160, 50_000, (100_000, 90_000, 80_000), 30, 10_000, 110, 0.33, 0.01, "2026-11-04"))


# ================================================================================================ 8. outputs and CLI
def t_outputs(tmp, base_specs, ref):
    out = os.path.join(tmp, "out")
    root = os.path.join(tmp, "clean")
    r = run(root, base_specs, ref, out=out, figure=True)
    files = sorted(os.listdir(out))
    check("8 outputs: result JSON, Figure 6 CSV, PDF and PNG written",
          files == ["e110_truncation_result.json", "fig6_completion_vs_tokens.csv", "fig6_completion_vs_tokens.pdf",
                    "fig6_completion_vs_tokens.png"], files)
    j = json.load(open(os.path.join(out, "e110_truncation_result.json"), encoding="utf-8"))
    check("8 outputs: the JSON carries the verdict, window, e, e_pub, secondary, audit and per-game costs",
          j["verdict"] == r["verdict"] and j["window"]["B_hi"] == 100_000 and close(j["e110"]["e"], r["e110"]["e"])
          and j["reference"]["e"] is not None and len(j["per_game"]) == 38 and "community" in j["label"]
          and "hidden" in j["label"] and "not hidden" in j["label"])
    rows = list(__import__("csv").DictReader(open(os.path.join(out, "fig6_completion_vs_tokens.csv"), encoding="utf-8")))
    series = {x["series"] for x in rows}
    at = {(x["series"], int(x["budget_tokens"])): float(x["completion_S"]) for x in rows}
    check("8 outputs: the CSV has the four series on a 5k grid and matches S(B_lo), S(B_hi)",
          series == {"e110_community_all", "e110_dreamteam", "e110_arc_interactive", "public_reference_pooled"}
          and abs(at[("e110_community_all", 50_000)] - r["e110"]["S_lo"]) < 1e-3
          and abs(at[("e110_community_all", 100_000)] - r["e110"]["S_hi"]) < 1e-3, sorted(series))
    check("8 outputs: the PDF is a PDF", open(os.path.join(out, "fig6_completion_vs_tokens.pdf"), "rb").read(5) == b"%PDF-")
    check("8 outputs: the verdict line names the result as community games, not hidden",
          r["verdict_line"].startswith("E155 VERDICT: ") and "community games" in r["verdict_line"]
          and "(not hidden games)" in r["verdict_line"], r["verdict_line"])
    zp = os.path.join(tmp, "clean.zip")
    out2 = os.path.join(tmp, "cli_out")
    env = dict(os.environ, PYTHONUTF8="1")
    p = subprocess.run([PY, "-I", os.path.join(HERE, "e110_truncation.py"), zp, "--out", out2, "--boot", "300",
                        "--obtained", "2026-10-30"], capture_output=True, text=True, env=env, cwd=tmp, timeout=600)
    last = (p.stdout.strip().splitlines() or [""])[-1]
    check("8 CLI: python -I runs from another folder on the zip, exit 0, last line = the verdict line, outputs written",
          p.returncode == 0 and last.startswith("E155 VERDICT: ") and os.path.exists(os.path.join(out2, "fig6_completion_vs_tokens.pdf"))
          and "not the registered analysis" in p.stdout, (p.returncode, last, p.stderr[-500:]))
    p2 = subprocess.run([PY, "-I", os.path.join(HERE, "e110_truncation.py"), os.path.join(tmp, "missing.zip")],
                        capture_output=True, text=True, env=env, cwd=tmp, timeout=120)
    check("8 CLI: a missing zip -> exit 2 and a pointer to the runbook", p2.returncode == 2 and "runbook" in p2.stdout, p2.stdout)


# ================================================================================================ 9. real data
def t_real(tmp, skip_dress):
    ref = T.load_reference()
    ok = ref["games"] is not None and len(ref["games"]) == 18 and ref["error"] is None
    check("9 real: the registered reference files load: the same 18 public games in E058 and E061 [C9]", ok, ref.get("error"))
    if ok:
        x = T.estimate(T.ref_units(ref, 50_000, 100_000), 50_000, 100_000, nboot=200)
        check("9 real: e_pub is computable on the real reference (value not asserted)",
              x["e"] is not None and math.isfinite(x["e"]) and x["S_lo"] > 0)
    e154 = os.path.join(HERE, "data", "e154_smoke")   # slim copy: only the fields read_jsonl / game_from_bench keep
    if os.path.isdir(os.path.join(e154, "arm_prune384")):
        src = T.Source(e154, "arm_prune384")
        runs = src.json("arm_prune384/benchmark.json")["game_runs"]
        agree = n = 0
        for g in runs:
            a = T.game_from_bench(g)
            b = T.game_from_logs(T.read_jsonl(src, f"arm_prune384/{g['game_id']}_p0_requests.jsonl", T.REQ_FIELDS),
                                 T.read_jsonl(src, f"arm_prune384/artifacts/{g['game_id']}_p0_events.jsonl", T.EV_FIELDS),
                                 g["number_of_levels"])
            ok_, nn = T.costs_agree(a["costs"], b["costs"])
            agree += ok_
            n += nn
        check(f"9 real: E154 smoke outputs (E098 harness): benchmark clock = request-log clock at {agree}/{n} completing "
              "actions (ledger C1: 10/10)", n == 10 and agree == 10, (agree, n))
    else:
        check("9 real: E154 clock check SKIPPED (data/e154_smoke not on disk)", True)
    zp = os.environ.get("E155_DRESS_ZIP", os.path.join(HERE, "data", "e081_results.zip"))   # 0.69 GB, not shipped
    if skip_dress or not os.path.exists(zp):
        check("9 real: E081 dress rehearsal SKIPPED" + (" (--skip-dress)" if skip_dress else " (zip not on disk)"), True)
        return
    t0 = time.time()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = T.analyse(zp, None, obtained=T.dt.date(2026, 10, 12), nboot=200, figure=False, arm="arm_v2")
    a = r["audit"]
    check(f"9 real: dress rehearsal on the real E081 results zip (0.69 GB, read in place, {time.time() - t0:.0f} s): 24 games "
          f"valid, no fallback, clocks agree at {a['crosscheck']['agree']}/{a['crosscheck']['completions']} completions",
          a["valid"] == 24 and not a["fallback"] and not a["missing"] and a["crosscheck"]["games"] == 24
          and a["crosscheck"]["agree"] == a["crosscheck"]["completions"] > 0, a)
    check("9 real: the E081 rehearsal is correctly not E110: reproduction fails and the 24-game window rule fails -> VOID",
          r["verdict"] == "VOID" and not r["reproduction"]["ok"], r["void_reasons"])


def main():
    tmp = tempfile.mkdtemp(prefix="e155t_")
    t0 = time.time()
    try:
        base = t_layout(tmp)
        t_metric(tmp)
        t_known(tmp)
        t_estimator(tmp)
        t_window(tmp)
        clean, ref = t_robust(tmp, base)
        t_verdicts(tmp)
        t_outputs(tmp, base, ref)
        t_real(tmp, "--skip-dress" in sys.argv)
    finally:
        if os.environ.get("E155_KEEP_TMP"):
            print("kept", tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    n_ok = sum(ok for _, ok in RESULTS)
    print(f"\n{n_ok}/{len(RESULTS)} passed ({time.time() - t0:.0f} s)")
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
