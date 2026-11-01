# The experiment ledger

`ledger.md` is the project's experiment log: 161 entries (E001 to E155, with sub-entries), not in strict numeric
order. Each entry was opened before its run with the question, the change, a prediction and a kill rule, and the result
and verdict were appended afterwards; amendments are dated and say what they change. The writeup cites experiments by
these numbers (E055, E058, E072, E098, E138, E149, E154, E155, ...). Every pre-registration the writeup cites is inside
this file: E155's registered text, first drafted in a paper draft, is pasted into its entry unchanged.

## What was changed from the project's copy

Source: `experiments/ledger.md` of the project repository as of 2026-10-30 (commit d2c3d67; 3,742 lines, 1,322,418
bytes, sha256 `01774d56...1776`). This copy: 1,322,452 bytes, the same 3,742 lines. Made by a script that fails if a new
name appears that nobody has reviewed.

| Kind | Found | Done |
|---|---|---|
| e-mail addresses | 0 | (would be replaced by "[e-mail removed]") |
| local paths with a user name, the Windows user name, the GitHub handle | 0 | |
| tokens, keys, passwords (the release scanner's patterns) | 0 | |
| other people named as the source of a claim who are not a public team name | 3 | replaced by a neutral phrase: one private person's real name ("a forum post, thread 750546"), one participant's user name ("another participant lost 14.6 ..."), one team member's user name attached to a score ("another participant's copy"). The forum thread numbers stay, so each claim can still be checked |

Kept on purpose:
- **public team names and their public leaderboard scores** (for example Tufa Labs, Yi-Chia Chen, NVARC3; 10 names);
- **owners of the public notebooks, datasets, models and repositories the ledger cites** (for example
  `sorokin/qwen3_4b_grids15_sft139`, `koushikrudra/failed-in-aimo`; 10 owners): the slug names the work and is needed to
  credit it and to find it;
- Kaggle forum thread numbers, Kaggle staff and organiser statements as cited.

## Reading it

- Labels: [measured] read from a log or a submission list; [computed] arithmetic on measured inputs; [guess] an
  assumption; [read], [claim] taken from a source we did not check ourselves.
- File names in the entries (`analysis/...`, `ARC-AGI3/...`, `experiments/...`, `reports/...`) are files of the project
  repository; most are not part of this release. The tools that the writeup's numbers depend on are in `../tools/`.
- Some characters show as replacement marks where the original was written with a different encoding; they are kept as
  they are in the project's copy.
