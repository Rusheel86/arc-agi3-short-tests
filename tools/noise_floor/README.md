# Leaderboard noise floor

## In plain words

On Kaggle a rerun of the same notebook does not score the same twice. Before you believe that version B beats version
A, you need to know how far one draw wanders. Give this tool every leaderboard score you have, one row per draw, labelled
by version. It tells you how noisy one draw is, the smallest true difference your number of draws can detect, and how
many draws you would need.

For our entry the answer was sobering: one draw is uncertain by about ±3 points at 90%, one draw against six can only
detect about 5.8 points, and a true +1 point needs about 52 draws per version (E138). Most changes we tried were far
smaller than that, so we decided on cheaper commit-run measurements and used the leaderboard only to confirm.

## Use

    python -I tools/noise_floor/noise_floor.py tools/noise_floor/examples/arc_agi3_public_leaderboard_draws.csv \
        --pool "E098,facts v1 (E118)" --compare E154 E098

Input CSV: columns `version` and `score` are required. `include` = 0 keeps a row in the table but out of every statistic
(instrument failures, void runs). Other columns are ignored. Options: `--pool` (versions whose draws give the pooled SD;
default all with 2+ draws), `--compare CANDIDATE CONTROL` (Welch t, a z-test at the pooled SD, the design's MDE, the
candidate draws needed), `--n`, `--delta`, `--alpha` (one-sided, default 0.05), `--power` (default 0.90), `--sd`
(use a given SD), `--json OUT`.

Output, in order: per-version n, mean, SD with its 90% chi-square interval and SE of the mean; the pooled within-version
SD; the ±90% half-width of one draw; the MDE for n draws of each version; the draws per version needed for +1 / +2 / +3;
and the comparison.

## Example data

`examples/arc_agi3_public_leaderboard_draws.csv`: all 24 of our ARC-AGI-3 public-leaderboard submissions from 2026-10-03
to 2026-10-29 [measured: Kaggle API], with the Kaggle submission ref, the date and the notebook's script version id.
E098's six draws come from two saved notebooks whose competition path is the same (the project ledger records them as
byte-identical, E102/E104); they are one version here. Excluded rows: a run of another kernel, a wrong notebook that never
played, an E146 stub resave (0.00) and the void E149 half-clock draw (0.31).

What the example gives [computed, 2026-10-30]: E098 mean 28.11, SD 1.83 (90% CI 1.23-3.82); pooled SD over E098 and
facts v1 1.73 (df 6); E154 (4 draws) mean 28.89, SD 1.71; E154 minus E098 +0.78, Welch t +0.69, df 6.9, two-sided p
0.51; minimum detectable effect of that design 3.28 at SD 1.73.

## Tests

    python -I tools/noise_floor/test_noise_floor.py      # own runner
    python -m pytest tools/noise_floor                    # if pytest is installed

The tests check the numbers the project's ledger printed from the same draws (E138; E154 draws 1-3): SD 1.83 and its
interval, the first-five SD 1.14, the pooled 1.735, 52 draws for +1, the MDE 3.00 for 10 + 4 draws, 5.8 / 4.4 / 3.8 for
1 / 2 / 3 draws against six, and the E154 n = 3 Welch test (+0.60, SE 1.397, t 0.43, df 3.68, p 0.69).

Assumptions: draws are independent and normal with the same SD in both versions; the MDE is for a one-sided test.
Leaderboard draws of one version share the same 55 hidden games, so the SD here is rerun noise, not game sampling.
