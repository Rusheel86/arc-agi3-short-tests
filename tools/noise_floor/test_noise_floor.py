"""Tests for noise_floor.py. Each expected number is one the project's ledger printed (E138, E154) from the same draws.

Run either way, from anywhere:
  python -I tools/noise_floor/test_noise_floor.py        (own runner; exit 0 = all pass)
  python -m pytest tools/noise_floor                      (if pytest is installed)
"""
import io
import math
import os
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:      # -I leaves the script's folder off sys.path
    sys.path.insert(0, HERE)
import noise_floor as N  # noqa: E402

EXAMPLE = os.path.join(HERE, "examples", "arc_agi3_public_leaderboard_draws.csv")
E098 = [27.27, 30.07, 28.24, 28.37, 29.68, 25.01]
FACTS = [24.42, 26.06]
E154_3 = [31.00, 28.05, 27.07]


def near(a, b, tol):
    return abs(a - b) <= tol


def test_e098_six_draws():
    """ledger E154 / E138: E098 mean 28.11, SD 1.83, 90% CI of the SD 1.23-3.82, SE of the mean 0.75"""
    res = N.analyse([dict(version="E098", score=s, include=True) for s in E098])
    v = res["versions"][0]
    assert near(v["mean"], 28.11, 0.005) and near(v["sd"], 1.83, 0.005)
    assert near(v["sd_ci90"][0], 1.23, 0.005) and near(v["sd_ci90"][1], 3.82, 0.005)
    assert near(v["se_mean"], 0.75, 0.005)


def test_first_five_sd():
    """E138: SD of the first five E098 draws 1.14"""
    assert near(N.sd(E098[:5]), 1.14, 0.005)


def test_pooled_stack_sd_and_draws_for_plus_one():
    """E138 / fig3: pooled SD E098 x6 + facts x2 = 1.735 (df 6); +1 point needs 52 draws per version; 10 + 4 draws MDE 3.00"""
    sp, df = N.pooled([E098, FACTS])
    assert near(sp, 1.735, 0.0005) and df == 6
    assert math.ceil(N.n_balanced(1.0, sp)) == 52
    assert near(N.mde(sp, 4, 10), 3.003, 0.0005)


def test_mde_one_two_three_against_six():
    """ledger E154: one / two / three draws against six at SD 1.83 detect about 5.8 / 4.4 / 3.8"""
    s = N.sd(E098)
    assert near(N.mde(s, 1, 6), 5.8, 0.05) and near(N.mde(s, 2, 6), 4.4, 0.05) and near(N.mde(s, 3, 6), 3.8, 0.05)


def test_welch_e154_three_draws():
    """ledger E154 draw 3: +0.60, Welch SE 1.397, t +0.43, Satterthwaite df 3.68, two-sided p 0.69"""
    w = N.welch(E154_3, E098)
    assert near(w["diff"], 0.60, 0.005) and near(w["se"], 1.397, 0.0005) and near(w["t"], 0.43, 0.005)
    assert near(w["df"], 3.68, 0.005) and near(w["p_two_sided"], 0.69, 0.005)
    k = N.known_sd_test(E154_3, E098, N.sd(E098))
    assert near(k["se"], 1.293, 0.0005) and near(k["z"], 0.46, 0.005) and near(k["p_two_sided"], 0.64, 0.005)


def test_example_csv_reads_and_excludes():
    rows = N.read_draws(EXAMPLE)
    res = N.analyse(rows, pool=["E098", "facts v1 (E118)"], compare=("E154", "E098"))
    by = {v["version"]: v for v in res["versions"]}
    assert by["E098"]["n"] == 6 and by["facts v1 (E118)"]["n"] == 2 and by["E154"]["n"] >= 3
    excl = {r["version"] for r in res["excluded"]}
    assert {"wrong notebook (never plays)", "E146 stub resave", "E149 v3 half-clock"} <= excl
    assert near(res["pooled"]["sd"], 1.735, 0.0005)
    assert "welch" in res["compare"] and res["compare"]["n_control"] == 6


def test_candidate_draws_never_when_control_alone_is_too_noisy():
    """with 6 control draws at SD 1.83 the control mean alone has SE 0.75, so +1 can never reach 90% power"""
    assert math.isinf(N.n_candidate(1.0, 1.83, 6))
    assert N.n_candidate(5.0, 1.83, 6) < 2


def test_bad_input():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.csv")
        open(p, "w", encoding="utf-8").write("name,value\na,1\n")
        try:
            N.read_draws(p)
            assert False, "missing columns accepted"
        except SystemExit as e:
            assert "required" in str(e)
        open(p, "w", encoding="utf-8").write("version,score\na,abc\n")
        try:
            N.read_draws(p)
            assert False, "non-numeric score accepted"
        except SystemExit as e:
            assert "not a number" in str(e)


def test_cli_isolated():
    """the CLI runs with -I from another folder and prints every section"""
    with tempfile.TemporaryDirectory() as d:
        js = os.path.join(d, "out.json")
        r = subprocess.run([sys.executable, "-I", os.path.join(HERE, "noise_floor.py"), EXAMPLE, "--pool", "E098,facts v1 (E118)",
                            "--compare", "E154", "E098", "--json", js], capture_output=True, text=True, cwd=d, timeout=120)
        assert r.returncode == 0, r.stderr[-500:]
        for s in ("1. Per version", "2. Pooled", "3. Minimum detectable", "4. Draws per version", "5. E154"):
            assert s in r.stdout, s
        assert "+1: 52" in r.stdout and os.path.exists(js)


def main():
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    ok = 0
    for name, fn in tests:
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                fn()
            print("PASS", name)
            ok += 1
        except Exception as e:   # noqa: BLE001
            print("FAIL", name, "--", type(e).__name__, str(e)[:300])
    print(f"\n{ok}/{len(tests)} passed")
    return 0 if ok == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
