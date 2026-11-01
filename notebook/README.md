# The E154 notebook

`arc3-e154-keepalive.ipynb` is the notebook of Kaggle kernel `rusheelsharma/arc-agi-3-e109`, **version 189** (script
version id **362115816**), the version whose four public-leaderboard draws scored 31.00, 28.05, 27.07 and 29.43. One
markdown cell was added in front: the MIT-0 licence statement and the third-party notice. Nothing else changed.

## Is this really the notebook that scored?

Kaggle refuses to hand out old versions of a notebook (HTTP 403 for every versioned pull, even to the owner), so the
server's copy of version 189 cannot be compared byte for byte. What can be checked:

| Check | Result |
|---|---|
| `python -I notebook/verify_notebook.py`: remove the header cell, write the notebook back in Kaggle's format, compare | identical to the file pushed for version 189: 989,125 bytes, sha256 `6cf982cdc56207671303e8fff71708aeda5a887d972d00b23c2fb1f40e153044` (the hash the project ledger recorded for E154 at 11:38 UTC on 2026-10-25, before the commit ran); 39 of 39 cells match `original_cells.json` |
| `python -I patches/apply_patches.py --harness <Duck harness copy> --out <dir> --check`: rebuild the harness from the notebook's patch cells | the rebuilt `solver.py` has sha256 `0f36330aea89e29227e8ab1bfb923789ad0d57df5a3d5111eca67c9d799e419b`, the hash that version 189's own run log printed ("E154 keep-alive patch applied") |
| version 189's run log against the notebook | the log writes the nine patch files in the notebook's cell order, prints the E146 stub line of the force-full commit, and the E154 settings line with the notebook's constants (keep-alive 7 min, stall 15 min, hard cap 100 min) |
| timing | the pushed file was saved 2026-10-25 11:03 UTC; version 189's commit ran 11:50-12:17 UTC; the next version (190) was pushed from a different folder at 14:37 UTC |

So the code cells are the ones that ran in version 189 as far as the run's own log can show; a cell that leaves no
trace in the log (a comment, for example) cannot be proved identical to the server's copy.

## Inputs it needs (from `kernel-metadata.json`)

| Input | Kind | Owner | Public today? |
|---|---|---|---|
| `arc-prize-2026-arc-agi-3` | competition data and wheels | ARC Prize Foundation | for competition participants |
| `rusheelsharma/arc3-e061-tools` | dataset: census scripts, decoding bench, quality probes, the E062 KV port, and the expert census of 2026-10-10 that the E073 cell turns into the 384-of-512 pruning plan (licence field apache-2.0). **Required**: without it the commit run stops at the E073 cell | us | **no, private**: making it public is the user's decision |
| `rusheelsharma/arc3-probe-set-e058` | dataset: model requests sampled from public-game play-throughs (`probe_set.jsonl`, 15 MB; licence field CC0-1.0). Used only by the E061 stages; if it is missing the preflight marks them skipped | us | **no, private**: making it public is the user's decision |
| `lordhansolo/vllm-main-e975732-arc3` | dataset: the vLLM build | Lord Han Solo | yes |
| `lordhansolo/taaf-kaggle-source` | dataset: source bundle with the 32k draft vocabulary | Lord Han Solo | yes |
| `dfranzen/pennyroyal-v253` | dataset: SGLang wheelhouse | Daniel Franzen | yes |
| `dfranzen/taaf-kaggle-source-bundle-copy` | dataset: the Duck harness source bundle | Daniel Franzen (copy of Tufa Labs' bundle) | yes |
| `lordhansolo/built-on-tufa-labs-duck-harness-milestone-2` | notebook (kernel source) | Lord Han Solo | yes |
| `lordhansolo/qwen3-8-flash-next-mixed-nvfp4-fp8` (PyTorch, hf-mixed-mtp-nvfp4, v1) | model: the served weights | primitive-ai build, mirrored by Lord Han Solo | yes |
| `dfranzen/intel-qwen3.8-flash-next-w4a16-autoround` (Transformers, v1) | model | Intel build, mirrored by Daniel Franzen | yes |
| `dfranzen/albucino-qwen3-8-flash-next-drafter` (Transformers, v1) | model: MTP draft head | albucino, mirrored by Daniel Franzen | yes |

"Public today" for the third-party inputs is as checked on 2026-10-28 (the pages loaded without signing in). In a
commit run the notebook's preflight cell checks every input (pinned sha256 values) before anything long starts and stops
on a missing required input; a competition rerun skips the preflight.

## Settings

`kernel-metadata.json` is version 189's metadata with the kernel id and title replaced by placeholders and the numeric
kernel id removed: GPU on (machine shape `NvidiaRtxPro6000`), internet **off**, the Docker image pinned by digest,
`is_private: true`. Fill `{{KAGGLE_USERNAME}}`, `{{KERNEL_SLUG}}` and `{{KERNEL_TITLE}}` before using it with
`kaggle kernels push` (which saves a new version and starts a GPU run).

## What a run does

- **Save (commit)**: the stub guard sees a normal commit and runs the full notebook (this version's first save was a
  deliberate full commit). It starts the K8V4 pruned vLLM server, plays 4 public games for 5 minutes each, and the
  E098 smoke check prints `SUBMIT THIS VERSION: yes` only if every check passes (K8V4 live, block 2,128, 163,840-token
  context, no preemptions, 32,768-token reply cap, the patches applied, the pool holds 19 games).
- **Scored rerun** (after a submission): the same notebook with the competition flag set; 19 games at once, about
  532 minutes per game, the keep-alive armed until the model server answers.

The header cell links this repository (https://github.com/Rusheel86/arc-agi3-short-tests). The submission ID is named in the writeup, because this
copy gets its own submission after it is saved.
