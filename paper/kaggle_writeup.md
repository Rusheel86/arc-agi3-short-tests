**Summary.** On ARC-AGI-3 an agent learns unseen games where every move is scored; on Kaggle a small open model plays 110 of them on one GPU in nine hours, so thinking is bounded per game. Lesson: evaluate thinking changes at the deployment budget. Three measurements. (1) Banning self-check words led within 40k tokens in four comparisons (two inside noise), then lost 20 levels at full length (noise about 6; E072) and about 10 points on single hidden-game draws (E055, E058). (2) Halving tokens per game, a side effect of deeper turns, did not cost score: the public slope predicted a fall from 23.98 to about 16 [computed]; E098 scored 28.11 (six draws against one; five changes, not isolated). (3) One leaderboard draw is uncertain by ±3 points (E138). Prompts, generated facts, light fine-tuning and ARC-AGI-2 program search did not help. Public-leaderboard score: 28.74 (one draw; version 189's four: 28.89). Code and ledger are public.

## 1. Why short tests pick the wrong agent

Moves are scored, thinking is free, compute is fixed, and a level scores only when finished. Thinking's cost is immediate: each turn's tokens come out of the game's budget. Its payoff arrives later, on the hard levels that checking finishes. A test shorter than deployment counts the cost and misses the payoff, so it favours agents that think less. Meta-optimisation has the same short-horizon bias: learning rates tuned over short unrolls come out too small (Wu et al., 2018, arXiv 1803.02021).

The ban's 60-minute runs show both halves (E058): on levels both arms solved the banned agent needed 11% fewer tokens, but of levels still open after 45k tokens it finished 2 of 22 by 60k against 8 of 21 (registered band). Unbanned, most "Hmm", "Wait" and "Actually" sentences check a rule against cells.

The reversal was observed, not predicted: E058's registered prediction gave a loss of 12 or more levels 25% (no effect 55%), and the argument was stated after it. The two agree: the ban cut thinking per turn and lost; E098's deeper turns halved its tokens per game without loss, though at elasticity 0.59 halving predicted a score of about 16, so depth paid at least the predicted loss [inference; not isolated]. The argument also needs tokens per game to keep paying at deployment length (the token leg); E149 v4 (hidden games) and E155 (community games), registered before their data, test that leg, not the principle itself. E149 v4 passed (a down-dose); E155 was void. No test of the principle itself succeeded: E157, a "think-less" post-trained variant registered to score below version 189, failed its registered mechanism gate (thinking 0.93x ours in a short smoke, inside our own smoke-to-smoke spread); its one draw, 8.64, is not evidence either way and its cause is not isolated [inference].

**What bounds tokens per game.** On one GPU the KV-cache pool (PagedAttention; Kwon et al., 2023) decides how many games think at once and how much context each keeps. E098 makes 531-575 tokens per second at 19 games (E137 replay bench; ~540 in play, E110 [computed]), half of E081's 1,112 at 24: about 148-160k tokens per game against ~309k [computed]. Running 19 games instead of 24 predicts x0.89 (E040's step model, fitted at short contexts), not the measured x0.49: depth is paid for mainly in attention per token at long context [inference].

## 2. Method

Inherited: Tufa Labs' Duck harness; Daniel Franzen's Milestone-2 harness, prompts and scheduler; Lord Han Solo's vLLM runtime and mixed NVFP4/FP8 checkpoint of Qwen 3.8 Flash Next; andreasgru's 4-bit KV-cache patch; for ARC-AGI-2, koushikrudra's public 4B notebook, NVARC's 4B grid model and icecuber's DSL solver. No submitted weights were trained. Our additions, each pre-registered with a kill rule:
1. Expert pruning calibrated on the agent's own traffic: keep 384 of 512 experts per layer (E060, E073).
2. A constant-cost state hand-off: the harness rewrote the history every move (2-26 s CPU per step); the fix changes the model's input only through fewer timeouts and lifts late-game occupancy from 0.74 to 0.98 (E077, E081).
3. A KV cache with fp8 keys and 4-bit values (+0.0016 nats per token; E083), spent on depth: a 160,768-token window and 32,768-token replies at 19 games, plus the timeout fix (E091, E098).
4. Slow platform loads: our one stub-saved rerun scored 0.00 (E146; suspected cause a reported 15-minute idle rule [guess]). The keep-alive version resets one game every 7 minutes until the server answers and waits longer for slow starts (E154). Over four draws it scored 28.89 (SD 1.71; +0.78 over E098, Welch t 0.69, p 0.51): it scores normally; reruns return no log, so whether the keep-alive acted is unknown.

## 3. Evidence

**Horizon reversal.** The ban cut tokens per call to x0.76 (E055). At 60 minutes per game it led at 40k tokens (10 games ahead, 2 behind), then lost 62 to 82 levels (0 ahead, 11 behind; E058); on the hidden games the ban version scored 11.28, against 21.24 and 21.87 for the nearest unbanned versions (E040, E061; one draw each, not identical code).

**Closest work.** On single-answer maths the small-model loss is known (NoWait, arXiv 2506.08343: distilled 4B-14B models lost 13-17 AIME25 points, QwQ-32B little; CGRS, arXiv 2508.05337; s1, arXiv 2501.19393). Model rankings reverse with the token cap on single answers (de Souza and Panisson, arXiv 2608.12150). Liew's ARC-AGI-3 speed-depth trade-off sets exploration against action efficiency (arXiv 2605.25931). Less reasoning helped SWE-bench agents (Cuadron et al., arXiv 2502.08235) and more steps beat deeper per-step reasoning in web agents (TTI, arXiv 2506.07976); neither scores moves, and our ban took more moves and lost. New, as far as searches found, is this reversal over a game's horizon in an agent with scored moves.

**What one draw can show.** Six draws of E098: mean 28.11, SD 1.83. A single draw is uncertain by about ±3 points at 90%; one draw against six detects about 5.8, and a true +1 needs about 50 draws per version (E138).

**Half the tokens.** On 18 public games, truncated 60-minute runs give a score elasticity of 0.59 between 140k and 180k tokens per game (90% interval 0.33-0.89; E072). Stretched to ~154k, over a partly unmeasured range, it predicts about 16 points (21.5-21.9 if returns stopped at 180k) [computed]; E098 scored 28.11 against E081's 23.98. That step, whose registration assumed x0.9 tokens per game, changed five things: reply cap 12k to 32k tokens (4.5-5.1% of replies had hit 12k; E077), window 94,208 to 160,768 tokens, KV format, a timeout fix (6 of 6 timeouts had wiped saved functions; E081), and 19 games instead of 24.

**The stack.** Public-leaderboard scores (55 hidden games): 23.98 (E081), 28.11 (E098) and 28.89 (E154), level with our starting baseline (27.80, 27.89; E001) and below fifth place (56.02, 2026-11-02).

**What did not help.** Prompts (24.83, three draws; E016), generated facts (+0.05, interval including 0; E136), light fine-tuning (E134) and ARC-AGI-2 program search (E141-E152).

**Failure case.** Under strict blind coding a stalled run held the winning run's complete rule in 3 of 15 matched pairs and acted on it in none (lenient coding: 12 of 15; E072), so most stalls lack the rule. Four hand-read games of an earlier, 20-minute agent illustrate it: a rule fixed from one observation, never tested (E048).

**The registered dose test.** We registered a half-clock draw (E149: half the time per game, displayed clock doubled, so only the token dose changes), band 13.5-24.8, below 5 void. Its first draw (v3) scored 0.31, void by that rule; v4 counts them from the first model reply. v4 drew 15.06, inside the band (elasticity 0.90 ± 0.18, the public range's steep end; one draw): the public down-slope transfers to hidden games below E098's dose; it cannot price the halving above it [inference].

## 4. What generalises and what does not

Without ARC words: where actions are scored and thinking is free on fixed compute, judge changes to thinking at the deployment horizon. That includes coding, web and embodied agents with costly actions. On 38 community-made games E098 finished 3.95 levels per game against 4.22 for the E081 version on public games (E097 control), with community levels x1.69 cheaper per token (E110). E155, a registered truncation of this run, is void by its rule (data not obtained by 11-04); the token leg at depth is untested.

Limits: one model and GPU type; the reversal rests on one change, one 60-minute session and single hidden-set draws, while cutting reasoning effort lost even at 20 minutes, with no early lead (E042); E110 crosses game sets and versions; five changes moved together.

## 5. Reproduction

Public notebook: https://www.kaggle.com/code/rusheelsharma/arc-agi-3-short-tests-pick-the-wrong-agent-e154, version 1 (version 189's code; submission 57877052); finals, auto-selected: 57631952 (same code, 31.00), E098 (30.07). Code and ledger: https://github.com/Rusheel86/arc-agi3-short-tests; companion paper: https://github.com/Rusheel86/arc-agi3-short-tests/blob/main/paper/paper_long.pdf. Our code is MIT-0; inherited code keeps Apache-2.0 or MIT; unlicensed material is not redistributed; Flash-Next weights carry the Qwen Community License. Compute: Kaggle's free tier only (RTX PRO 6000), under 150 quota-hours [estimate].
