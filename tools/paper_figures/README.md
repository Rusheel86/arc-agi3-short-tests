# Paper figures

## In plain words

Every figure in the writeup is made by one script from data files; no number is typed in by hand. This folder holds the
script, the data it reads, the published figures, and a check that regenerating them gives the same files.

    python -I tools/paper_figures/check_figures.py          # regenerate into a temp folder and compare with expected/
    python -I tools/paper_figures/make_paper_figures.py --out DIR
    python -I tools/paper_figures/make_cover_v6.py --out DIR   # the writeup's cover: "Short Tests Pick the Wrong Agent"

Needs numpy, scipy, matplotlib (and Pillow for make_cover_v6.py). About 40 s on a CPU in all.

## Result of the check [measured 2026-10-30]

All 19 files match the published ones (`expected/` = the project's `reports/paper_figures/`: 18 files made 2026-10-27,
commit 25e3f9b, and `cover_v6.png`, made 2026-10-30 by `make_cover_v6.py`): the 8 PNGs (fig1-fig5, cover, cover_alt,
cover_v6) are **byte-identical (sha256)**, the 5 CSVs and `captions.md` are
byte-identical, and the 5 SVGs are identical after removing the creation date and renumbering the random ids matplotlib
writes into every SVG. Measured with Python 3.13, matplotlib 3.11.2, numpy 2.5.3, scipy 1.18.1 on Windows with the
Segoe UI font. Another matplotlib version or a machine without Segoe UI (the script then falls back to DejaVu Sans) draws
the same numbers but the PNG pixels differ.

## The figures

| File | Writeup figure | What it shows | Ledger ids |
|---|---|---|---|
| `cover_v6.png` | the cover of writeup v6 | `cover.png` with the title "Short Tests Pick the Wrong Agent" and the note "one draw each, different code" under the right panel; same data and drawing code (its script first redraws `cover.png` and checks 0 pixels differ) | E058, E055, E040, E061 |
| `fig2_horizon_reversal` (+ `cover.png`) | Figure 1 and the earlier cover | banning the words that open self-checks leads early and loses at 60 minutes, 18 public games; hidden-set draws 11.28 vs 21.24 and 21.87 | E055, E058, E072, E040, E061 |
| `fig1_half_tokens_did_not_cost` (+ `cover_alt.png`) | Figure 2 | hidden-set score against tokens per game; E098 got about half E081's tokens per game and scored 4.1 more; the cause is not isolated | E081, E098, E137, E072 |
| `fig3_noise_floor` | Figure 3 | every leaderboard draw by saved version (as of 2026-10-27), and the minimum detectable effect against draws per version | E138, E154, E149, E146 |
| `fig3_noise_floor_v2` (+ `fig3_data_v2.csv`, `make_fig3_v2.py`) | Figure 3, newer version | the same figure with every draw on record at 2026-11-02: 25 ARC-AGI-3 rows (E154 now 4 draws, 28.89 ± 1.71, never pooled with E098; E149 v4's 15.06 as an own-dose draw, never pooled) and 12 ARC-AGI-2 rows (v4 n = 6, v5 n = 6). The pooled SD 1.73, the 3.00 and the 52 do not move. `fig3_noise_floor` above is left as published | E138, E154, E149, E146 |
| `fig4_memory_prices` | Figure 4 | KV-pool blocks per full-context game with a drafter's own cache; expert pruning buys memory, not speed | E137, E073 |
| `fig5_agi2_funnel` | Figure 5 | ARC-AGI-2: answers within reach of a small decoder change, yet none found | E142, E150-E152 |

`captions.md` has one caption per figure (40 words or fewer); `figN_data.csv` the plotted numbers. The captions and the
script's docstrings name the project's file paths and ledger line numbers as they were on 2026-10-27; those files are
not part of this release, and `data/` holds what was read from them.

## Data (`data/`), frozen at the figures' commit

| File | What it is (the project file it was taken from) |
|---|---|
| `agi3_draws.csv`, `agi2_draws.csv`, `agi2_versions.json` | the leaderboard draw tables of `analysis/leaderboard_noise.py` and `leaderboard_noise_agi2.py` at commit 25e3f9b (20 ARC-AGI-3 rows, through E154 draw 1; 6 ARC-AGI-2 rows) |
| `e058_arm_long_control_benchmark.json`, `e058_arm_long_nohmm_benchmark.json` | the E058 60-minute screen (control and word ban, 18 public games), slim: the fields the scoring rule reads |
| `t3_recheck_results_excerpt.txt` | 3 lines: the E072 elasticity 0.59, its band, the ±6-level noise floor |
| `e081_run_excerpt.txt`, `e073_run_excerpt.txt` | the throughput lines printed by the E081 and E073 run notebooks |
| `e137_summary_subset.json`, `t2a_result_excerpt.md`, `throughput_arithmetic_excerpt.txt` | the drafter bench and allocator numbers, the KV pool sizes |
| `ledger_excerpts.json` | the 3 ledger lines that are the only record of three plotted numbers (E149's 0.31, E150's 0 of 14, E073's 18 -> 24 games), with the line numbers the captions print |
| `agi2/` | the ARC-AGI-2 counts (E142, E151, E152, the v6 smoke): only the fields read |
| `agi3_draws_v2.csv`, `agi2_draws_v2.csv` | the same two leaderboard tables at 2026-11-02 (25 ARC-AGI-3 rows, 12 ARC-AGI-2 rows), read only by `make_fig3_v2.py` (exported from the project tables with `make_fig3_v2.py --export-data DIR`) |

To redraw Figure 3 with later draws (E154 has 4 draws by 2026-10-29), append rows to `data/agi3_draws.csv` (the labels
come from `AGI3_LABELS` in the script); the published figure then no longer matches, by design.

`python -I tools/paper_figures/make_fig3_v2.py --out DIR` redraws the 2026-11-01 version; `check_figures.py` does not cover it (it compares `expected/` only). Run here it reproduced `fig3_noise_floor_v2.png` byte for byte.

## Changes against the project's script

Only the inputs: the paths now point into `data/`, the leaderboard tables are read from CSV instead of imported from
`leaderboard_noise.py` (whose five helper functions are inlined with the same formulas), `game_curve` comes from
`game_curve.py` (copied unchanged from `k_killtests.py`), the ledger lines come from `ledger_excerpts.json`, and the
default output folder is `./paper_figures_out`. Every plotting line is unchanged.
