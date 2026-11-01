# Our changes, as readable diffs

## In plain words

The notebook does not ship a modified copy of the harness. It starts from the inherited Duck harness, applies Daniel
Franzen's Milestone 2 patch, then applies ours, one patch per experiment. The same patches are here as plain files, so
you can read what each change does without opening the notebook.

## Harness patches (`harness/`), in the order the notebook applies them

| File | Ledger | What it does | In E154 |
|---|---|---|---|
| `01-e040-context-window.patch` | E040 | the context budget is read per game | applied |
| `02-arm-reasoning-effort-switch.patch` | arm switch | a reasoning-effort switch per arm, kept for parity | unused |
| `03-e055-bad-words-switch.patch` | E055 | an optional ban on the words that open self-checks | off (E058: the ban loses at length) |
| `04-e058-temperature-per-request.patch` | E058 | the sampling temperature can be set per request | applied |
| `05-e077-state-handoff.patch` | E077 | constant-cost game-state hand-off instead of rewriting the whole history every move | on (`ARC3_STATE_HANDOFF=incremental`) |
| `06-e091-max-output.patch` | E091 | the reply cap is read per game (32,768 tokens in the competition arm) | on |
| `07-e097-harness-v3.patch` | E097 | four harness-computed fact lines (off: no measured gain) and the retained-functions timeout fix (on) | timeout fix only |
| `08-e154-keepalive.patch` | E154 | while the model server is not answering, the warm-up game repeats its RESET every 7 minutes | on |

Each file starts with a comment header (what it does, which files, the source cell, the sha256 of the body); `git apply`
skips the header. `index.json` lists the files each patch touches.

Rebuild the patched harness and check it against the Kaggle run logs:

    python -I patches/apply_patches.py --harness <dir with ARC3-Inference/> --out <new dir> --check [--lf]
    python -I patches/apply_patches.py --harness <dir> --out <new dir> --e149 v3 --check

`--harness` is a copy of the Duck harness source, for example the `src/` folder of the Kaggle dataset
`dfranzen/taaf-kaggle-source-bundle-copy` (`--lf` converts a Windows checkout's line endings first). Measured on
2026-10-30 against a local copy of the Duck harness: every patch applies, and the rebuilt files have exactly the sha256
values the Kaggle runs printed: `solver.py` 0f36330a... (kernel version 189, E154) and, with `--e149 v3`,
`tool_agent.py` 9cdd1cef... (version 190). With `--e149 v4` both files match the post-patch hashes (16-digit prefixes)
that the v4 apply cell pins and that kernel version 191's commit log printed.

## E154 notebook changes (`e154/`)

`e154-notebook-cells.diff`: the E154 notebook against the E098 notebook, cell by cell. Four cells changed:
- the E146 v2 stub guard (inserted; with `force_full_default '1'` the first save is a real commit),
- the keep-alive patch cell (= `harness/08-e154-keepalive.patch`),
- the E154 settings cell: applies the patch fail-closed, sets the keep-alive (`ARC3_KEEPALIVE_MIN` 7,
  `ARC3_KEEPALIVE_HEALTH_URL`), and in a competition rerun the start timers: the first-request grace
  (`E073_HTTP_GRACE_S` and `ARC3_HTTP_RETRY_INITIAL_SECONDS`, 9,000 s and 3,000 s -> 21,600 s) and the launcher's ready
  deadline (`ARC3_VLLM_READY_TIMEOUT_S`, 3,600 s -> 6,600 s); it also defines the start-cap helpers;
- the server cell: the launcher reads its ready deadline from `ARC3_VLLM_READY_TIMEOUT_S`, and the 45-minute start cap
  replaces a loading server only if its start has shown no progress for 15 minutes, or at a hard cap of 100 minutes.

## E149 half-clock (`e149/`), a research tool

E149 asks how much score the agent loses when every game gets half the time, while the model still reads the full
clock, so that only the token dose changes. It is kept here as a tool, not as a result.
- `e149-v3-display-clock.patch` + `e149-v3-notebook-cells.diff`: v3 (kernel version 190) halves each game's limit
  (532 -> 266 minutes) and doubles the two clock fields the model reads. Its one draw scored 0.31: **void** by the
  registered rule, cause not found (one candidate: the game clocks start before the model loads, so a very late load
  leaves the half-clock games almost no time).
- `e149-v4-clock-origin.patch` + `e149-v4-notebook-cells.diff`: v4 counts every game's clock from the run's first model
  reply instead. It ran as kernel version 191: the smoke commit passed on 2026-10-30 and draw 1 (ref 57806285) scored
  15.06, inside the registered primary band 13.5-24.8; for the registered reading see ledger E149.

All patches here are ours, MIT-0. Franzen's `harness-changes.patch` (Apache-2.0) stays in the notebook.
