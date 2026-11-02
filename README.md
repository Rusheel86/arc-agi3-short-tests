# Short Tests Pick the Wrong Agent: our ARC-AGI-3 entry (E154), its patches and the measurement tools

This is the open-source release of our entry to the ARC Prize 2026 ARC-AGI-3 competition on Kaggle and of the code
and experiment ledger behind its Paper Track writeup, "Short Tests Pick the Wrong Agent" (PDF: `paper/paper_long.pdf`, the current version; dated copies alongside it).

## In plain words

ARC-AGI-3 is a set of small video-game-like puzzles. The agent sees a grid, chooses moves, and has to work out the
rules of each game by itself; every move counts against it, thinking is free. On Kaggle a small open model plays 110
hidden games on one GPU in nine hours, offline, so each game's thinking is bounded.

The writeup reports three measurements, all in `ledger/`:
1. **Short tests pick the wrong agent.** Banning the words that open the model's self-checks led early in four
   comparisons (two inside noise), then lost 20 levels at full length on 18 public games (noise floor 6) and about
   10 points on the hidden games (E055, E058).
2. **Halving tokens per game to buy depth did not cost score.** E098 got about half its predecessor's tokens per game
   and scored 4.1 points more (six draws against one; five changes moved together, so the cause is not isolated;
   E098, E081).
3. **One leaderboard draw is uncertain by about ±3 points** (E138), so most single-draw comparisons say little.

Our agent is not new from scratch. It is a chain of public work with our changes on top:

1. **Tufa Labs' Duck harness** (Jeroen Cottaar): the game loop, the Python tool sandbox and the prompts.
2. **Daniel Franzen's Milestone 2 solution**: a patch to that harness (harness, prompts, scheduler) and the notebook we
   started from, using the open-weights model Qwen3.8-Flash-Next.
3. **Lord Han Solo's serving stack**: a vLLM build and a mixed NVFP4/FP8 copy of the model that fits the GPU.
4. **Our changes** (each one an experiment in the ledger, numbered E0xx/E1xx):
   - keep 384 of the model's 512 experts per layer, chosen from the agent's own traffic, so more games fit in memory
     (E060, E073);
   - hand the game state to the model at a constant cost per move instead of rewriting the whole history (E077);
   - store the model's attention memory with 8-bit keys and 4-bit values (E083, `k8v4_port/`) and spend the saved
     memory on depth: 19 games at once with a 160,768-token window and 32,768-token replies (E091, E098);
   - keep the agent's saved helper functions when a tool call times out (E097, the "timeout fix");
   - survive slow model loads on Kaggle: while the model server is still loading, one game repeats its RESET every
     7 minutes so the run is not closed as idle, and the start timers wait for a load that is still making progress
     (E154, this version).

E154 is E098 plus the keep-alive and start timers. Its four draws (31.00, 28.05, 27.07, 29.43; mean 28.89) show that
this version scores normally. They do not show the keep-alive's own effect: Kaggle's scored reruns return no log, so
whether a slow load happened, and whether the keep-alive acted, is not known.

## Results (public leaderboard = the 55-game public half of the hidden set)

Every number below is a Kaggle public-leaderboard score of the same saved notebook, re-run by Kaggle. The private half
(the other 55 hidden games) is scored only after the deadline. A rerun of identical code does not score the same twice.

| Version | Kaggle saved version | Draws [measured] | Mean | SD per draw |
|---|---|---|---|---|
| **E154** = E098 + keep-alive (this release) | kernel version 189, script version id 362115816 | 31.00, 28.05, 27.07, 29.43 | 28.89 | 1.71 |
| Public release copy of E154 (this notebook: kernel `arc-agi-3-short-tests-pick-the-wrong-agent-e154` version 1, code byte-identical to version 189) | ref 57877052 | 28.74 | 28.74 | (1 draw; reported separately from version 189, not pooled) |
| E098 (depth + K8V4 + timeout fix) | script version ids 357974936 (1 draw) and 358364499 (5 draws), the same competition path | 27.27, 30.07, 28.24, 28.37, 29.68, 25.01 | 28.11 | 1.83 |
| E081 (the version before E098) | script version id 357441618 | 23.98 | | |

[computed with `tools/noise_floor`] E154 minus E098: +0.78 points; Welch t +0.69 (df 6.9), two-sided p 0.51. Four E154
draws against six E098 draws can only detect a true difference of about 3.5 points (one-sided 0.05, power 0.90, SD 1.83),
so the two versions are not separable. The inherited Milestone 2 notebook scored about 27.8 on the same board (E001).
Two draws of related versions are excluded from every pool as instrument failures: an E146 stub resave (0.00) and the
first E149 half-clock draw (version 190, 0.31, void, cause not found). The registered half-clock dose test was re-run
as E149 v4 (kernel version 191, its game clocks counted from the first model reply): draw 1 (ref 57806285, 2026-10-30)
scored 15.06 [measured], inside its registered primary band 13.5-24.8. For the registered reading of that number see
ledger E149; it is not interpreted here.

Other kinds of numbers in the writeup, so they are not mixed up with the board:
- **public-set numbers**: runs we made on the 25 public games (Kaggle commit runs, 60 minutes per game), for example the
  token elasticity 0.59 (E072) and the self-check word ban (E055, E058);
- **community-game numbers**: E110, 38 community-made games the agent was never tuned on (E155 analyses them);
- **[computed]** numbers are arithmetic on measured ones, **[guess]** marks an assumption.

## Reproduce on Kaggle

The notebook runs on one RTX PRO 6000, internet off, inside the competition's 9-hour limit. The save (commit) is a
real GPU smoke test (17 to 50 minutes so far, depending on how fast Kaggle loads the model); the scored rerun runs on
Kaggle's side after you submit.

1. Read `notebook/README.md`: it lists the 11 inputs (competition data, 6 datasets, 1 notebook, 3 models). The run needs
   our dataset `rusheelsharma/arc3-e061-tools`; `rusheelsharma/arc3-probe-set-e058` is used only by optional stages.
2. Create a Kaggle notebook from `notebook/arc3-e154-keepalive.ipynb`, attach the inputs, choose the RTX PRO 6000
   accelerator and turn internet off (`notebook/kernel-metadata.json` holds the exact settings, including the pinned
   Docker image, for `kaggle kernels push`; fill in the placeholders first).
3. Save a version. The first save is a real commit: it starts the model server, plays 4 public games for 5 minutes each
   and prints `E098 SMOKE: PASS | SUBMIT THIS VERSION: yes` only if every check passes.
4. Submit that version to `arc-prize-2026-arc-agi-3`. Kaggle reruns it on the hidden games.

Check that the notebook is the one that scored: `python -I notebook/verify_notebook.py` (removes the licence header cell
and compares the bytes with the file pushed for version 189). Rebuild the exact harness files it runs:
`python -I patches/apply_patches.py --harness <copy of the Duck harness> --out <new folder> --check` (the rebuilt
solver.py matches the sha256 that version 189's own log printed).

## What is in this folder

| Path | What it is |
|---|---|
| `notebook/` | the E154 notebook (version 189) with an MIT-0 licence header cell in front; its inputs; a byte-identity check; a `kernel-metadata.json` template |
| `ledger/` | the experiment ledger: every experiment's question, prediction and kill rule, written before the run, and its result appended after; lightly scrubbed (see `ledger/README.md`) |
| `patches/harness/` | our eight harness patches as readable diffs, in the order the notebook applies them, each with a header saying what it does |
| `patches/e154/` | the E154 change against the E098 notebook, cell by cell (keep-alive patch and its apply cell, start timers, server cell) |
| `patches/e149/` | the E149 half-clock patches (a research tool: give every game half the time while the model still reads the full clock; v3 ran once and was void; v4 ran as version 191 and drew 15.06, see ledger E149) |
| `patches/apply_patches.py` | rebuilds the patched harness from the inherited one and checks the hashes the Kaggle runs logged |
| `k8v4_port/` | the K8V4 KV-cache port, byte-identical to the copy the notebook writes; **Apache-2.0** (it contains andreasgru's code), with its licence |
| `tools/noise_floor/` | a small CLI: give it a CSV of leaderboard draws per version, get the per-draw SD, the smallest detectable difference and the draws needed; with tests |
| `tools/e155_truncation/` | the pre-registered E155 analysis (token elasticity at deployed depth on community games, by truncating one run at fixed token budgets), with its 67-check test suite |
| `tools/paper_figures/` | the scripts and data that regenerate the paper's five figures and both covers, and a check against the published files |
| `paper/` | the long paper as PDF (`paper_long_v7.pdf`, 21 pages, 2026-11-02); the Kaggle writeup link is added once it is submitted (by 2026-11-08) |
| `LICENSE` | MIT-0, for our code |
| `LICENSES/Apache-2.0.txt`, `NOTICE` | the Apache-2.0 text and the notices for the Apache-2.0 code shipped here |
| `THIRD_PARTY.md` | every inherited component and model, with its licence, source and how sure we are |

All tools run on a CPU. Tested with Python 3.13 (numpy 2.5.3, scipy 1.18.1, matplotlib 3.11.2, Pillow 12.3.0); the
noise-floor tool also with scipy 1.17.1. Run them with `python -I` from this folder; each test file is its own runner
(exit code 0 = all pass), and the noise-floor tests also run under pytest.

## Licence

Our code: MIT-0 (`LICENSE`). Inherited code keeps its own licence: the K8V4 port and the Apache-2.0 parts inside the
notebook (`NOTICE`, `LICENSES/Apache-2.0.txt`); the Tufa Labs harness, which the notebook loads from a Kaggle dataset,
has no confirmed licence (`THIRD_PARTY.md`). The model weights are not in this folder; they are Kaggle Model inputs under
the Qwen Community License 1.0. Whether that licence meets the prize's "open source weights" wording has not been
answered by the organisers.

## Links

- Paper (PDF): https://github.com/Rusheel86/arc-agi3-short-tests/blob/main/paper/paper_long.pdf (current version; dated copies alongside it)
- Paper Track writeup: to be submitted on Kaggle by 2026-11-08 (link added then)
- Public notebook: Kaggle kernel `rusheelsharma/arc-agi-3-short-tests-pick-the-wrong-agent-e154`, version 1 (code byte-identical to the private kernel version 189); leaderboard submission ref 57877052 (public score 28.74, 2026-11-02)
- This repository: https://github.com/Rusheel86/arc-agi3-short-tests
