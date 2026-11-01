# E155: token elasticity at deployed depth, by truncating one run

## In plain words

Our writeup argues that on ARC-AGI-3 the score follows how many tokens each game gets to think with. We measured that
on the public games at an older, shallower setting. E155 asks whether it still holds at the depth we actually deployed
(E098: 160,768-token window, 32,768-token replies, 19 games at once), on a second set of games the agent was never tuned
on (E110: 38 community-made games).

It needs no new GPU run. E110 logged, for every game, how many tokens the model had generated when each level was
finished. Cut every game at B tokens and count what was finished by then: S(B). Compare B = 50k with B = 100k (or 90k /
80k, by a rule fixed in advance). The elasticity e = ln(S(B_hi) / S(B_lo)) / ln(B_hi / B_lo) says how fast completion
grows with tokens (e = 1: double the tokens, double the completion; e = 0: extra tokens buy nothing). The same script
computes the public 60-minute controls of E058 and E061 as a reference (`data/reference/`).

The rules were registered in the project ledger on 2026-10-28, before the E110 results archive was downloaded and before
any E110 truncation number existed. They are copied at the end of this file. **Result: pending** (the archive,
`e110_results.zip`, 0.67 GB, is a Kaggle notebook output that only the owner can download). Every result must be
labelled "community games, deployed depth, one run", never "hidden".

## Files

| File | What it is |
|---|---|
| `e110_truncation.py` | the analysis (CLI). Identical to the project's `analysis/e110_truncation.py` except file locations and the docstring |
| `e110_readout.py` | the E110 readout it imports (helpers: game ids, sources, request-to-action ranges), unchanged |
| `game_curve.py` | the E059/E072 truncation scoring rule (`game_curve`, `load_runs`), copied unchanged from the project's `k_killtests.py` |
| `test_e110_truncation.py` | 67 checks in 9 groups: layout, metric, known elasticities, estimator, window rule, robustness, verdicts, outputs, real data |
| `data/reference/` | slim copies of the two registered reference runs (E058 and E061 60-minute controls on 18 public games): only the fields the script reads |
| `data/e154_smoke/` | slim copy of the E154 commit smoke's logs (4 public games), for the check that the two token clocks agree |

## Run

    PYTHONUTF8=1 python -I tools/e155_truncation/e110_truncation.py e110_results.zip --out e155_out [--obtained YYYY-MM-DD]

The archive is read in place (nothing is extracted or executed). The last printed line is the registered verdict
(`E155 VERDICT: ...`); `e155_out/` gets `e110_truncation_result.json` and Figure 6 (`fig6_completion_vs_tokens.csv`,
`.pdf`, `.png`). `--boot N` other than 10,000 marks the run as not the registered analysis.

## Tests

    PYTHONUTF8=1 python -I tools/e155_truncation/test_e110_truncation.py [--skip-dress]

Synthetic results are written in the archive's real layout; the real-data group reads the shipped reference and E154
files. The dress rehearsal on the real E081 results archive (0.69 GB, same layout; not shipped) runs only when
`E155_DRESS_ZIP` points to it; otherwise it is reported as skipped. Measured on 2026-10-30 from this folder: 66 of 66
checks pass with the rehearsal skipped (63 s), 67 of 67 with it (82 s).

Slim data check [measured 2026-10-30]: on the full and the slim reference files the script gives the same e_pub, the
same S(50k), S(100k) and the same bootstrap interval (10,000 resamples), number for number.

## The registered entry (verbatim from the project ledger, E155)

### E155 — Token elasticity at deployed depth on community games: truncate E110's E098 run at fixed token budgets (CPU only)            (opened 2026-10-28, pre-registered BEFORE e110_results.zip is on disk and before any E110 truncation number has been computed; CPU only: no GPU, no quota, no push, no submission, no commit; no score claim)
Status at registration: e110_results.zip (0.67 GB, 443 files) is still on Kaggle (kernel rusheelsharma/arc-agi-3-e109, the version that ran 2026-10-14 17:21-19:32 UTC); only the user can download it. Script analysis/e110_truncation.py and its tests analysis/test_e110_truncation.py are built AFTER this entry; runbook experiments/e110_truncation_STATUS.md.
What was read before writing this entry (so nothing here was tuned to E110's truncation): analysis/e110_readout.py, analysis/build_e110_nb.py (zip layout), analysis/k_killtests.py game_curve and analysis/t3_recheck.py (E059/E072's truncation rule), E154's smoke outputs ARC-AGI3/e154_out/arm_prune384 (same harness, used for the format and the clock check in C1), the per-game token totals and actions_per_level of the two reference controls (format check; e_pub NOT computed), and the E110 HEADLINE line of the saved notebook ARC-AGI3/arc-agi-3-e110_run.ipynb (totals only: 150 levels, 3,881,160 tokens, the ledger's 3.88M). The saved notebook's per-game E110 lines (tokens and levels per game) were NOT read.

Registered text, pasted unchanged from reports/paper_draft_v4_2026-10-27.md section 4 (only the placeholder "E1xx" became E155):
> **E155 — Token elasticity at deployed depth on community games: truncate E110's E098 run at fixed token budgets
> (CPU only)** (opened before `e110_results.zip` is opened; no GPU, no push, no submission)
> Question: at E098's deployed depth (160,768-token window, 32,768-token replies, 19 games at once), does completion
> on games the agent was never tuned on still rise with tokens per game, as on the public games at shallower depth?
> This tests the token leg of "memory sets the budget" at deployed depth on a second game set. It is community games,
> not hidden games, and it says nothing about what bought E098's +4.1.
> Data: E110's results zip (38 community games: 25 DreamTeam, 13 ARC-Interactive; E098 version unchanged; ~60 min per
> game, 19 at once; 3.88M generated tokens, ~102k per game; 150 levels) [measured in E110]. Per request: generated
> tokens and the game's level count after the action the reply chose.
> Outcome metric: completion score per game within the first B generated tokens = (sum of the level numbers of the
> levels finished within B) / (sum of the level numbers of all the game's levels), i.e. the scoring rule's level
> weights with the move-efficiency factor set to 1 (community games have no human action counts on file). S(B) =
> mean over games. A game that ended before using B tokens keeps its final value.
> Window: B_lo = 50k, B_hi = 100k tokens per game. Rule fixed now: if fewer than 30 of the 38 games used >= 100k
> tokens, B_hi = the largest of {90k, 80k} that at least 30 games reached, decided from token counts alone before any
> level is read; if neither, VOID.
> Estimator: e = ln(S(B_hi) / S(B_lo)) / ln(B_hi / B_lo); 90% CI by game-cluster bootstrap (10,000 resamples, seed 110).
> Reference (same script, same metric, same window): the two public 60-minute controls of E058 and E061 (18 games
> each, the shallower E040/E051-era depth), pooled, e_pub with its own bootstrap CI. For orientation only, the score
> elasticity of those controls between 45k and 100k is about 1.05 [computed from existing files:
> `analysis/t3_recheck_results.txt` pooled scores 12.3 at 45k and 28.5 at 100k]; the levels-only elasticity in the same
> range is about 0.76 [computed from existing files: pooled levels 64 at 45k and 117 at 100k]. These are not the
> registered reference, which must be recomputed with the completion metric.
> Outcomes (first that applies):
> - VOID: the zip is not obtained by 11-04, fewer than 30 games have per-request token and level logs, the window
>   rule above fails, or the script cannot reproduce E110's own totals (150 levels, 3.88M tokens) within 1%.
> - HOLDS: e >= 0.33 and its CI lower bound > 0. Reading: at deployed depth, tokens still pay on a second game set;
>   the paper may say "the token leg holds at depth on community games" with the number.
> - WEAKER: e < 0.33 and its CI upper bound < e_pub. Reading: on community games, deeper settings make extra tokens
>   matter less ("depth makes tokens matter less" gains support, on community games only).
> - INCONCLUSIVE: anything else; reported with its CI, no claim.
> Secondary (reported, not deciding): the same with games that finished all their levels before B_hi excluded (the
> ceiling: community levels are x1.69 cheaper per token, so a ceiling pulls e down); e / e_pub; DreamTeam and
> ARC-Interactive separately; levels-only elasticity.
> Prediction [guess]: e about 0.6 (80% interval 0.25-1.0); HOLDS about 65%, WEAKER 10%, INCONCLUSIVE 15%, VOID 10%.
> Kill threshold: none (a measurement). Cost: a few CPU-hours on the laptop; 0 quota; 0 submissions.
> Caveats, stated now: one run, one session; community games are easier per token than public ones (E110); the
> reference controls ran at a different depth, so e vs e_pub mixes depth and game set; truncation reads a single run's
> trajectory, not independent runs at each budget.

Closed choices (added 2026-10-28 by the builder, in writing, before the script exists and before any E110 truncation number; C1-C15 close what the text above leaves open; C4 is the ONE departure from a literal reading and is flagged; everything else is an interpretation):
- C1 Data unit and token clock. A game's "per-request token and level log" = its record in arm_e110/benchmark.json (game_runs): the per-action `history` with `generated_tokens` (the harness books each reply's generated tokens on the reply's first action; tokens of replies that executed no action fall on the next action), `actions_per_level`, `levels_completed`, `number_of_levels`. Level k (1-based) is finished at action number sum(actions_per_level[:k]); its cost = the cumulative history tokens through that action (E059/E072's rule: analysis/k_killtests.py game_curve, t3_recheck section 3). The reference controls have only this clock (no request logs), so both sides use it. Check of the clock [measured 2026-10-28, ARC-AGI3/e154_out/arm_prune384, E098 harness, 4 games]: the request-log clock (cumulative usage.completion_tokens in *_p0_requests.jsonl through the request whose action range holds the completing action; completions from artifacts/*_p0_events.jsonl) gives the same cost at all 10 completing actions.
- C2 Fallback. A game whose benchmark record is missing or corrupt (history not parseable, shorter than its last completing action, levels_completed > number_of_levels, negative or non-integer tokens) is read from its request log + events log with the C1 check's clock; its level count from the benchmark record when that field is readable, else len(baseline_actions) of its metadata.json (e110_envs/ in the zip, else analysis/dreamteam_games_dataset + analysis/arc_interactive_games_dataset); its total tokens = the request log's sum. A game with neither is MISSING. Every game with both clocks is cross-checked (share of completing actions whose two costs agree within 1%): reported, not deciding.
- C3 Which games count. The session's list (e110/manifest.json settings.games; else every game in benchmark.json). Every listed game with a valid record (C1 or C2) counts in S(B) with weight 1, including games with 0 levels, 0 actions or a harness error (they stay in at 0). Unlisted games are ignored and reported; a duplicated game id uses its first record. VOID (b) = fewer than 30 valid games.
- C4 "Used >= B" (window rule) = the game's total generated tokens (benchmark solver_note "tokens=N", the rule of E110's own summary; else the history sum; C2 games: the request-log sum) >= B. DEPARTURE, flagged: a game that finished all its levels (levels_completed == number_of_levels) also counts as reaching every budget, because the clock did not cut it and its value at every B is exact, which is what the rule protects. The window decision sees only (total tokens, finished-all-levels) per game, never a truncated value. The literal count (tokens only) and the window it would choose are printed as a secondary line; the deciding window is this one.
- C5 "At least 30 games" counts among the valid games; the threshold stays 30 when fewer than 38 are valid.
- C6 A level finished after the cut counts 0 at that budget (no partial credit); it counts at B iff its cost <= B (ties count). A game that ended before using B tokens keeps its final value (text above).
- C7 S(B) = plain mean over valid games of 100 x sum_{finished k} k / sum_{k=1..N} k (0-100 scale; the ratio e does not depend on the scale).
- C8 Estimator edges. S(B_lo) = 0 -> e undefined -> INCONCLUSIVE. Bootstrap: resample the valid games with replacement (n = number of valid games), 10,000 resamples, random.Random(110); a resample with S*(B_lo) = 0 and S*(B_hi) > 0 gives e* = +inf, one with both 0 is dropped and counted; more than 1% dropped -> INCONCLUSIVE. 90% CI = the 5th and 95th percentiles by t3_recheck's q() (sorted list, index round(f x (n - 1))). Comparisons use unrounded values; printing rounds to 3 decimals.
- C9 Reference. ARC-AGI3/screen_e058/arm_long_control/benchmark.json (E058, 2026-10-07) and ARC-AGI3/screen_e061/arm_long_control/benchmark.json (the second 60-min control, run inside the E061 session 2026-10-10), the two files of t3_recheck; the same 18 public games. Pooled as t3_recheck pools: S_pub(B) = mean over the 18 games of the two runs' mean completion; e_pub at E110's chosen window; bootstrap = resample the 18 games with both runs of a game together, 10,000, random.Random(110), the same percentile rule. WEAKER compares E110's CI upper bound with the point value of e_pub. Same script and same functions as E110. If the reference cannot be read, WEAKER cannot apply (HOLDS still can); otherwise INCONCLUSIVE.
- C10 Reproduction (VOID (d)): over every listed game record that benchmark.json carries (valid or not), the sums of levels_completed and total tokens (C4's count) must be within 1% of 150 levels and 3,881,160 tokens (the session's own headline [measured 10-14]); without a readable benchmark.json the sums come from the C2 fallbacks and the same 1% rule applies. Internal check: for each C1 game, the curve at B = infinity must equal levels_completed, else the game goes to C2.
- C11 Date rule (VOID (a)): "obtained by 11-04" = the zip's file modification date (the download) <= 2026-11-04 UTC, or the date given with --obtained; the script prints the date it used. A folder input without --obtained uses the newest file date inside it.
- C12 Outcome order as written: VOID, HOLDS, WEAKER, INCONCLUSIVE; the first that applies.
- C13 Secondary, reported only: (i) ceiling: games whose last level was finished at a cost <= B_hi left out of both budgets; (ii) e / e_pub; (iii) DreamTeam and ARC-Interactive separately (source from manifest settings.sources, else stem length 6 = DreamTeam, 4 = ARC-Interactive, e110_readout's rule), each with its bootstrap CI; (iv) levels-only elasticity (S = mean number of levels finished by B, unweighted), E110 and the reference; (v) the literal-window variant of C4, if its window differs; (vi) e on the request-log clock for games with both clocks. No other budget pair is computed for any claim; the S(B) curve at 5k steps exists only for Figure 6.
- C14 Outputs: e110_truncation_result.json (every number and the verdict line), fig6_completion_vs_tokens.csv (series, budget, S, levels per game, games, games reaching B) and fig6_completion_vs_tokens.pdf (+ .png): S(B) for E110 all / DreamTeam / ARC-Interactive and the pooled public reference, B_lo and B_hi marked, labelled "community games".
- C15 Naming: the result is "community games, deployed depth, one run"; never "hidden". A HOLDS reading supports only the token leg at depth; it says nothing about what bought E098's +4.1.
Amendment 1 (2026-10-28, during the build, still before any E110 data exists on disk): two implementation details the choices above left implicit. (1) The bootstrap draws from the valid games in sorted stem order (the reference: its 18 stems sorted), so the CI does not depend on file or manifest order. (2) A request or events log with any unparseable line is rejected as a whole for the C2 fallback and cross-check (reported); a C2 cross-check "agrees" within max(1 token, 1%).
Build [measured 2026-10-28, CPU]: analysis/e110_truncation.py + analysis/test_e110_truncation.py (synthetic fixtures in the zip's real layout, read by e110_readout itself; known-elasticity recovery; edge cases; real reference files; E154 clock check; a dress rehearsal on the real E081 results zip, same layout, read in place in ~60 s: 24/24 games valid, the two clocks agree at 77/77 completing actions). Test count and the reference value computed after this registration: experiments/e110_truncation_STATUS.md.
Result: PENDING (zip not downloaded).
Verdict: PENDING.
Paper: Theory (the token leg of "memory sets the budget" at deployed depth; skeptic v3's cheapest Theory fix), Universality (second game set); community games only.
