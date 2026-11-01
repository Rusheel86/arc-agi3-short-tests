"""E110 readout (ledger E110): per-game levels, tokens and actions of the community-game session, engine replay of every
logged game in the local toolkit, and the candidate training records the K1 step of the training plan
(reports/training_plan_2026-10-14.md section 6) starts from.

Plain words: the Kaggle session plays community games and logs every model request. This script (CPU, local)
  1. summarises each game (levels completed, actions per level, generated tokens, requests logged);
  2. replays every logged action in the local engine and checks each logged board; a completed level counts as
     ENGINE-VERIFIED only if the replay matches every board up to and including the completing action and the engine's
     own level counter rises there (the same replay rule as E100/E101: board after each action == logged board, a run
     is used only up to its first divergence);
  3. turns the request logs into candidate training records (format "e110-k1-v1", defined here because the plan does not
     fix one; see K1_FORMAT below): F1 = the last 1-3 model requests before each logged level completion, labelled by
     whether the reply's executed actions completed the level (the plan's primary family), F2 = requests whose reply
     ended in GAME_OVER (the plan's optional family, reported apart).
No RHAE: community games have no human baselines. DreamTeam's metadata baselines are the generator's solution lengths;
their ratio to the agent's actions is printed as "solver ratio" (descriptive, not a score). ARC-Interactive baselines are
not used.

Usage (the downloaded results are data: run with -I from the project root):
  PYTHONUTF8=1 ./.venv/Scripts/python.exe -I analysis/e110_readout.py <e110_results.zip | working dir> \
      [--envs DIR] [--out DIR] [--no-replay] [--workers N] [--no-prompts]
  --envs: the environments dir the session played (default: e110_envs/ inside the results, else rebuilt from
          analysis/dreamteam_games_dataset + analysis/arc_interactive_games_dataset). The DreamTeam import shim is built
          in a temp dir from the DreamTeam dataset's adapter.
Writes <out>/e110_readout.json (summary + per-game + per-level), <out>/trajectories.jsonl (one line per completed level:
the logged action sequence from the level start, engine verdict) and <out>/k1_candidates.jsonl.gz (records).
The notebook embeds this file and calls summarize_run() only (no engine there).
"""
import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import statistics
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(globals().get("__file__") or "e110_readout.py"))  # the notebook execs this file
ROOT = os.path.dirname(HERE)
ARM = "arm_e110"
K1_FORMAT = "e110-k1-v1"
F1_DEPTH = 3                 # the plan's "last 1-3 requests before each logged level completion"
LEVEL_RE = re.compile(r"Current state: step (\d+), level (\d+)\.")
TOK_RE = re.compile(r"tokens=(\d+)")
sha = lambda b: hashlib.sha256(b).hexdigest()


# ================================================================================================ helpers (stdlib only)
def stem_of(game_id):
    return str(game_id).split("-")[0]


def source_of(stem, sources=None):
    if sources and stem in sources:
        return sources[stem]
    return "dreamteam" if len(stem) == 6 else "arc-interactive" if len(stem) == 4 else "unknown"


def _history(g):
    h = g.get("history") or []
    if isinstance(h, str):
        import ast
        h = ast.literal_eval(h)
    return h


def game_tokens(g):
    m = TOK_RE.search(str(g.get("solver_note") or ""))
    if m:
        return int(m.group(1))
    return sum(int(x.get("generated_tokens") or 0) for x in _history(g))


def load_pairs(path):
    """[(request_row, response_row or None)] in log order (analysis/build_probe_set.py's pairing)"""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    rows.append({"event": "corrupt"})
    out, i = [], 0
    while i < len(rows):
        if rows[i].get("event") == "request":
            resp = rows[i + 1] if i + 1 < len(rows) and rows[i + 1].get("event") == "response" else None
            out.append((rows[i], resp))
            i += 2 if resp is not None else 1
        else:
            i += 1
    return out


def reply_of(pairs, k):
    """(assistant message, why): request k's reply = the message request k+1 appends to the history (build_probe_set's
    rule); None when the next request rewrote the history (a trim) or does not exist"""
    req, resp = pairs[k]
    if resp is None:
        return None, "no_response"
    if k + 1 >= len(pairs):
        return None, "last_request"
    M, N = req.get("messages") or [], pairs[k + 1][0].get("messages") or []
    if len(N) <= len(M) or N[:len(M)] != M:
        return None, "history_changed"
    A = N[len(M)]
    if A.get("role") != "assistant":
        return None, "next_not_assistant"
    return A, "ok"


def files_of(arm_dir):
    """game_id -> {'events': path, 'requests': path, 'transcript': path or None}"""
    out = defaultdict(dict)
    art = os.path.join(arm_dir, "artifacts")
    for d, key, suf in ((art, "events", "_p0_events.jsonl"), (arm_dir, "requests", "_p0_requests.jsonl")):
        if os.path.isdir(d):
            for n in os.listdir(d):
                if n.endswith(suf):
                    out[n[:-len(suf)]][key] = os.path.join(d, n)
    tdir = os.path.join(arm_dir, "transcripts")
    if os.path.isdir(tdir):
        for n in os.listdir(tdir):
            gid = n.split("_p0")[0]
            if gid in out:
                out[gid]["transcript"] = os.path.join(tdir, n)
    return dict(out)


# ================================================================================================ 1. summary (no engine)
def summarize_run(arm_dir, sources=None):
    """per-game rows and totals from benchmark.json + the request logs (stdlib only; the notebook calls this)"""
    bench = json.load(open(os.path.join(arm_dir, "benchmark.json"), encoding="utf-8"))
    files = files_of(arm_dir)
    rows = []
    for g in bench.get("game_runs") or []:
        gid = g.get("game_id")
        stem = stem_of(gid)
        apl = [int(x) for x in (g.get("actions_per_level") or [])]
        base = [int(x) for x in (g.get("base_actions_per_level") or [])]
        lc = int(g.get("levels_completed") or 0)
        tok = game_tokens(g)
        n_req = n_resp = n_reply = 0
        reasons = Counter()
        rq = files.get(gid, {}).get("requests")
        if rq and os.path.exists(rq):
            pairs = load_pairs(rq)
            n_req = len(pairs)
            n_resp = sum(1 for _, r in pairs if r is not None)
            for k in range(len(pairs)):
                A, why = reply_of(pairs, k)
                reasons[why] += 1
                n_reply += A is not None
        solver_ratio = [round(apl[i] / base[i], 3) for i in range(min(lc, len(apl), len(base)))
                        if base[i] > 0 and source_of(stem, sources) == "dreamteam"]
        rows.append(dict(game_id=gid, stem=stem, source=source_of(stem, sources), state=g.get("state"),
                         levels_completed=lc, levels_total=len(base) or None, actions=sum(apl) if apl else None,
                         actions_per_completed_level=apl[:lc], solver_ratio=solver_ratio, tokens=tok,
                         requests=n_req, responses=n_resp, replies_recoverable=n_reply, reply_reasons=dict(reasons),
                         events=bool(files.get(gid, {}).get("events")), transcript=bool(files.get(gid, {}).get("transcript"))))
    return dict(games=rows, totals=_totals(rows))


def _totals(rows):
    def agg(rs):
        lv = sum(r["levels_completed"] for r in rs)
        tok = sum(r["tokens"] for r in rs)
        apl = [a for r in rs for a in r["actions_per_completed_level"]]
        sr = [x for r in rs for x in r["solver_ratio"]]
        avail = sum(r["levels_total"] or 0 for r in rs)
        return dict(games=len(rs), levels=lv, levels_per_game=round(lv / len(rs), 3) if rs else None,
                    levels_available=avail, share_of_levels=round(lv / avail, 4) if avail else None,
                    games_with_a_level=sum(1 for r in rs if r["levels_completed"] > 0),
                    games_won=sum(1 for r in rs if str(r["state"]).lower() in ("won", "win")),
                    tokens=tok, tokens_per_game=round(tok / len(rs)) if rs else None,
                    levels_per_1M_tokens=round(lv / tok * 1e6, 3) if tok else None,
                    median_actions_per_completed_level=statistics.median(apl) if apl else None,
                    median_solver_ratio=statistics.median(sr) if sr else None,
                    requests=sum(r["requests"] for r in rs), replies_recoverable=sum(r["replies_recoverable"] for r in rs),
                    games_without_events=[r["stem"] for r in rs if not r["events"]])
    out = {"all": agg(rows)}
    for src in sorted({r["source"] for r in rows}):
        out[src] = agg([r for r in rows if r["source"] == src])
    return out


def format_summary(s):
    lines = ["E110 per game (levels / available, actions per completed level, tokens, requests, recoverable replies):"]
    for r in sorted(s["games"], key=lambda r: (r["source"], r["stem"])):
        lines.append(f"  {r['source'][:10]:10s} {r['stem']:7s} {r['levels_completed']}/{r['levels_total']}  state {str(r['state'])[:9]:9s} "
                     f"apl {r['actions_per_completed_level']}  tokens {r['tokens']:,}  requests {r['requests']}  replies {r['replies_recoverable']}")
    for k, t in s["totals"].items():
        lines.append(f"  TOTAL {k}: {t['games']} games, {t['levels']} levels ({t['levels_per_game']} per game; "
                     f"{t['share_of_levels']} of {t['levels_available']} available), {t['games_with_a_level']} games with >= 1 level, "
                     f"{t['tokens']:,} tokens ({t['tokens_per_game']:,} per game), levels per 1M tokens {t['levels_per_1M_tokens']}, "
                     f"median actions per completed level {t['median_actions_per_completed_level']}, "
                     f"median solver ratio {t['median_solver_ratio']}, requests {t['requests']}, recoverable replies {t['replies_recoverable']}")
    return "\n".join(lines)


# ================================================================================================ 2. engine replay
_W = {}


def _worker_setup(env_dir, shim):
    if _W:
        return
    os.environ["ONLY_RESET_LEVELS"] = "true"
    if shim and shim not in sys.path:
        sys.path.insert(0, shim)
    import logging
    logging.disable(logging.CRITICAL)
    import arc_agi
    from arcengine import GameAction
    _W["arc"] = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=env_dir)
    _W["GA"] = GameAction
    _W["ids"] = {e.game_id for e in _W["arc"].available_environments}


def _action(display, name):
    """(GameAction, data, label) from a logged action, with E101's parser (analysis/e101_patchset2_tests.control_of)"""
    import e101_patchset2_tests as E101
    GA = _W["GA"]
    ctrl, click = E101.control_of(display, name)
    if ctrl == "CLICK":
        return GA.ACTION6, {"x": int(click[1]), "y": int(click[0])}, f"MOUSE(row={click[0]}, col={click[1]})"
    if ctrl == "RESET":
        return GA.RESET, {}, "RESET"
    if ctrl in E101.KEY_IDS:
        return GA.from_id(E101.KEY_IDS[ctrl]), {}, ctrl
    raise ValueError(f"unknown action {display!r} / {name!r}")


def replay_game(args):
    """replay one game's logged actions; returns per-action and per-level records (never raises)"""
    events_path, game_id, env_dir, shim = args
    import numpy as np
    _worker_setup(env_dir, shim)
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    out = dict(game_id=game_id, diverged=None, error=None, actions=[], levels=[])
    try:
        events = [json.loads(l) for l in open(events_path, encoding="utf-8") if l.strip()]
        init = next(e for e in events if e.get("type") == "initial")
        acts = [e for e in events if e.get("type") == "action"]
        if game_id not in _W["ids"]:
            out["error"] = f"{game_id} not in the environments dir"
            return out
        env = _W["arc"].make(game_id)
        prev = np.array(env.observation_space.frame[-1])
        if not np.array_equal(prev, np.array(init["board"])):
            r = env.step(_W["GA"].RESET)
            prev = np.array(env.observation_space.frame[-1])
        if not np.array_equal(prev, np.array(init["board"])):
            out["diverged"] = 0
            out["error"] = "initial board differs"
            return out
        lc = 0
        level = int(init.get("level") or 1)
        for ev in acts:
            act, data, label = _action(ev.get("action_display"), ev.get("action_name"))
            r = env.step(act, data=data)
            frames = [np.array(f) for f in (r.frame or [])] if r is not None else []
            last = frames[-1] if frames else prev
            match = bool(np.array_equal(last, np.array(ev["board"])))
            lc_new = int(getattr(r, "levels_completed", lc) or 0) if r is not None else lc
            state = str(getattr(r, "state", "")).upper()
            logged_done = bool(ev.get("level_completed")) or str(ev.get("state")).upper() == "WIN"
            out["actions"].append(dict(num=ev.get("action_num"), step=ev.get("analysis_step"), level=level, label=label,
                                       match=match, logged_completed=logged_done, engine_completed=lc_new > lc or "WIN" in state,
                                       game_over=bool(ev.get("game_over")) or "GAME_OVER" in state))
            if not match:
                out["diverged"] = ev.get("action_num")
                break
            lc = lc_new
            level = int(ev.get("level") or level)
            prev = last
    except Exception as exc:
        out["error"] = repr(exc)[-400:]
    out["levels"] = levels_from_actions(out["actions"], out["diverged"])
    return out


def levels_from_actions(actions, diverged=None):
    """one record per level reached: the logged actions from its start, completion (logged / engine-verified)"""
    by, order = defaultdict(list), []
    for a in actions:
        if a["level"] not in by:
            order.append(a["level"])
        by[a["level"]].append(a)
    out = []
    for L in order:
        acts = by[L]
        ci = next((i for i, a in enumerate(acts) if a["logged_completed"]), None)
        seq = acts[:ci + 1] if ci is not None else acts
        verified = ci is not None and all(a["match"] is True for a in seq) and seq[-1]["engine_completed"] is True
        out.append(dict(level=L, completed=ci is not None, engine_verified=bool(verified),
                        actions=len(seq), resets=sum(1 for a in seq if a["label"] == "RESET"),
                        game_overs=sum(1 for a in seq if a["game_over"]), first_action=seq[0]["num"] if seq else None,
                        completing_action=seq[-1]["num"] if ci is not None else None,
                        trajectory=[a["label"] for a in seq] if ci is not None else None))
    return out


def logged_actions(events):
    """the logged action list in replay_game's record shape, from the events alone (no engine; match/engine unknown)"""
    init = next((e for e in events if e.get("type") == "initial"), {})
    level = int(init.get("level") or 1)
    out = []
    for ev in events:
        if ev.get("type") != "action":
            continue
        disp = str(ev.get("action_display"))
        out.append(dict(num=int(ev.get("action_num") or 0), step=ev.get("analysis_step"), level=level,
                        label="RESET" if disp.upper().startswith("RESET") else disp, match=None,
                        logged_completed=bool(ev.get("level_completed")) or str(ev.get("state")).upper() == "WIN",
                        engine_completed=None, game_over=bool(ev.get("game_over"))))
        level = int(ev.get("level") or level)
    return out


def merge_levels(logged, engine):
    """logged level records with engine_verified taken from the replay's record of the same level and completing action"""
    eng = {(L["level"], L["completing_action"]): L for L in engine if L["completed"]}
    out = []
    for L in logged:
        L = dict(L)
        e = eng.get((L["level"], L["completing_action"]))
        L["engine_verified"] = bool(e and e["engine_verified"])
        L["engine_replayed"] = e is not None
        out.append(L)
    return out


def level_table(arm_dir, replay):
    """game_id -> the logged levels (from the events) with the engine verdict of the replay merged in"""
    out = {}
    for gid, f in sorted(files_of(arm_dir).items()):
        if not f.get("events"):
            continue
        events = [json.loads(l) for l in open(f["events"], encoding="utf-8") if l.strip()]
        out[gid] = merge_levels(levels_from_actions(logged_actions(events)), (replay.get(gid) or {}).get("levels") or [])
    return out


def replay_all(arm_dir, env_dir, shim, workers=4):
    files = files_of(arm_dir)
    jobs = [(f["events"], gid, env_dir, shim) for gid, f in sorted(files.items()) if f.get("events")]
    if workers <= 1:
        res = [replay_game(j) for j in jobs]
    else:
        from multiprocessing import Pool
        with Pool(min(workers, max(1, len(jobs)))) as pool:
            res = pool.map(replay_game, jobs)
    return {r["game_id"]: r for r in res}


# ================================================================================================ 3. K1 candidate records
def request_action_ranges(pairs, n_actions_final):
    """per request k: the action numbers its reply executed = [a_k, a_{k+1} - 1], a_k = the request's 'action' field
    (1 + actions executed before it); the last request runs to the final action count"""
    starts = [int(req.get("action") or 1) for req, _ in pairs]
    out = []
    for k, a in enumerate(starts):
        nxt = starts[k + 1] if k + 1 < len(starts) else n_actions_final + 1
        out.append((a, max(a, nxt) - 1))
    return out


def extract_k1(arm_dir, replay, sources=None, inline_prompts=True):
    """candidate records (K1_FORMAT) from every game's request log + its replay"""
    files = files_of(arm_dir)
    records, stats = [], Counter()
    for gid, f in sorted(files.items()):
        if not f.get("requests") or not f.get("events"):
            stats["games_missing_logs"] += 1
            continue
        rp = replay.get(gid) or {}
        by_num = {a["num"]: a for a in rp.get("actions") or []}
        events = [json.loads(l) for l in open(f["events"], encoding="utf-8") if l.strip()]
        acts = logged_actions(events)
        ev_acts = [e for e in events if e.get("type") == "action"]
        n_final = max([a["num"] for a in acts] or [0])
        ev_by_num = {int(e.get("action_num") or 0): e for e in ev_acts}
        levels = {L["level"]: L for L in merge_levels(levels_from_actions(acts), rp.get("levels") or [])}
        pairs = load_pairs(f["requests"])
        ranges = request_action_ranges(pairs, n_final)
        stem = stem_of(gid)

        def record(k, family, level, completing_num, offset):
            req, resp = pairs[k]
            A, why = reply_of(pairs, k)
            lo, hi = ranges[k]
            executed = [ev_by_num[n] for n in range(lo, hi + 1) if n in ev_by_num]
            done_here = completing_num is not None and lo <= completing_num <= hi
            lv = levels.get(level) or {}
            prefix = [a["label"] for a in acts if a["num"] < lo]
            rec = dict(format=K1_FORMAT, record_id=f"{gid}-L{level}-{family}-r{k}", family=family, game_id=gid, stem=stem,
                       source=source_of(stem, sources), level=level, offset_before_completion=offset,
                       request=dict(log_index=k, analysis_step=req.get("analysis_step"),
                                    request_index_within_turn=req.get("request_index_within_turn"),
                                    action_field=req.get("action"), file=os.path.relpath(f["requests"], arm_dir).replace(os.sep, "/"),
                                    sha256=sha(json.dumps(req, sort_keys=True).encode("utf-8"))),
                       prompt=(dict(messages=req.get("messages"), tools=req.get("tools"), tool_choice=req.get("tool_choice", "auto"),
                                    chat_template_kwargs=req.get("chat_template_kwargs") or {}) if inline_prompts else None),
                       usage=(resp or {}).get("usage"), finish_reason=(resp or {}).get("finish_reason"),
                       logged_reply=A, reply_source="next_request" if A is not None else None, reply_missing_reason=None if A else why,
                       outcome=dict(actions=[dict(num=int(e.get("action_num") or 0), display=e.get("action_display"),
                                                  level_completed=bool(e.get("level_completed")), game_over=bool(e.get("game_over")))
                                             for e in executed],
                                    n_actions=len(executed), completes_level=bool(done_here),
                                    actions_until_completion=(completing_num - lo + 1) if completing_num is not None else None,
                                    game_over_in_reply=any(bool(e.get("game_over")) for e in executed)),
                       engine=dict(restore="replay the logged actions before this request in the local engine (arc.make(game_id), "
                                           "RESET if the board differs from the initial one, then prefix_actions)",
                                   prefix_actions=prefix, n_prefix_actions=len(prefix),
                                   replay_matched_through_request=all(by_num.get(n, {}).get("match", False) for n in range(1, hi + 1)) if hi >= 1 else True,
                                   level_engine_verified=bool(lv.get("engine_verified"))),
                       label=dict(score=1 if done_here else 0, tie_break_actions=(completing_num - lo + 1) if done_here else None))
            return rec

        # F1: the completing request and up to F1_DEPTH - 1 requests before it that were issued after the level started
        # (a request's first action number >= the level's first action number)
        for L, lv in sorted(levels.items()):
            if not lv["completed"]:
                continue
            c, first = lv["completing_action"], lv["first_action"]
            kc = next((k for k, (lo, hi) in enumerate(ranges) if lo <= c <= hi), None)
            if kc is None:
                stats["f1_completion_without_request"] += 1
                continue
            chain = [k for k in range(kc, -1, -1) if ranges[k][0] >= first][:F1_DEPTH]
            for off, k in enumerate(chain):
                records.append(record(k, "F1", L, c, off))
                stats["f1"] += 1
                stats["f1_engine_verified"] += bool(lv.get("engine_verified"))
                stats["f1_with_reply"] += records[-1]["logged_reply"] is not None
                stats["f1_label_1"] += records[-1]["label"]["score"]
        # F2: requests whose executed actions include a GAME_OVER (optional family of the plan)
        for k, (lo, hi) in enumerate(ranges):
            ex = [ev_by_num[n] for n in range(lo, hi + 1) if n in ev_by_num]
            if any(bool(e.get("game_over")) for e in ex):
                L = next((a["level"] for a in acts if a["num"] == lo), None) or int(ex[0].get("level") or 0)
                records.append(record(k, "F2", L, None, None))
                stats["f2"] += 1
    return records, dict(stats)


# ================================================================================================ inputs
def materialize(src, out_dir):
    """a results zip is extracted under out_dir/extracted (data only, never imported); a folder is used in place"""
    if os.path.isdir(src):
        return os.path.abspath(src)
    dst = os.path.join(out_dir, "extracted")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    with zipfile.ZipFile(src) as z:
        for n in z.namelist():
            p = os.path.normpath(os.path.join(dst, n))
            if not p.startswith(os.path.abspath(dst)):
                raise ValueError(f"unsafe path in zip: {n}")
        z.extractall(dst)
    return dst


def build_envs(root, tmp):
    """(env_dir, shim): the session's own e110_envs/ when present, else rebuilt from the two dataset folders"""
    shim = os.path.join(tmp, "shim")
    os.makedirs(os.path.join(shim, "arc_agi_3", "game_creator"), exist_ok=True)
    for p in (os.path.join(shim, "arc_agi_3", "__init__.py"), os.path.join(shim, "arc_agi_3", "game_creator", "__init__.py")):
        open(p, "w").close()
    shutil.copyfile(os.path.join(HERE, "dreamteam_games_dataset", "arc_agi_3", "game_creator", "arcengine_adapter.py"),
                    os.path.join(shim, "arc_agi_3", "game_creator", "arcengine_adapter.py"))
    own = os.path.join(root, "e110_envs")
    if os.path.isdir(own):
        return own, shim
    env = os.path.join(tmp, "envs")
    for d in (os.path.join(HERE, "dreamteam_games_dataset", "environment_files_generated"),
              os.path.join(HERE, "arc_interactive_games_dataset", "environment_files")):
        if os.path.isdir(d):
            shutil.copytree(d, env, dirs_exist_ok=True)
    return env, shim


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--envs", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--no-replay", action="store_true")
    ap.add_argument("--no-prompts", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    out = os.path.abspath(a.out or os.path.join(os.path.dirname(os.path.abspath(a.src)), "e110_readout"))
    os.makedirs(out, exist_ok=True)
    root = materialize(a.src, out)
    arm = os.path.join(root, ARM)
    man_p = os.path.join(root, "e110", "manifest.json")
    manifest = json.load(open(man_p, encoding="utf-8")) if os.path.exists(man_p) else {}
    sources = (manifest.get("settings") or {}).get("sources")
    summ = summarize_run(arm, sources)
    print(format_summary(summ), flush=True)
    result = dict(summary=summ, manifest_settings=manifest.get("settings"))
    if not a.no_replay:
        tmp = tempfile.mkdtemp(prefix="e110_ro_")
        env_dir, shim = build_envs(root, tmp)
        env_dir = os.path.abspath(a.envs) if a.envs else env_dir
        replay = replay_all(arm, env_dir, shim, a.workers)
        merged = level_table(arm, replay)
        lv = [dict(game_id=g, source=source_of(stem_of(g), sources), **{k: v for k, v in L.items() if k != "trajectory"})
              for g, Ls in merged.items() for L in Ls]
        comp = [x for x in lv if x["completed"]]
        by_src = {src: dict(completed=sum(1 for x in comp if x["source"] == src),
                            engine_verified=sum(1 for x in comp if x["source"] == src and x["engine_verified"]))
                  for src in sorted({x["source"] for x in lv})}
        result["replay"] = dict(games=len(replay), diverged={g: r["diverged"] for g, r in replay.items() if r["diverged"] is not None},
                                errors={g: r["error"] for g, r in replay.items() if r["error"]},
                                actions_replayed=sum(len(r["actions"]) for r in replay.values()),
                                completed_levels=len(comp), engine_verified=sum(x["engine_verified"] for x in comp),
                                by_source=by_src, levels=lv)
        with open(os.path.join(out, "trajectories.jsonl"), "w", encoding="utf-8") as fh:
            for g, Ls in sorted(merged.items()):
                for L in Ls:
                    if L["completed"]:
                        fh.write(json.dumps(dict(game_id=g, source=source_of(stem_of(g), sources), **L)) + "\n")
        recs, kst = extract_k1(arm, replay, sources, inline_prompts=not a.no_prompts)
        with gzip.open(os.path.join(out, "k1_candidates.jsonl.gz"), "wt", encoding="utf-8") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
        result["k1"] = dict(format=K1_FORMAT, records=len(recs), **kst)
        print(f"engine replay: {result['replay']['games']} games, {result['replay']['actions_replayed']} actions, diverged "
              f"{result['replay']['diverged'] or 'none'}, errors {result['replay']['errors'] or 'none'}; completed levels "
              f"{result['replay']['completed_levels']}, engine-verified {result['replay']['engine_verified']}", flush=True)
        print(f"K1 candidates ({K1_FORMAT}): {len(recs)} records {dict(kst)} -> {os.path.join(out, 'k1_candidates.jsonl.gz')}", flush=True)
        shutil.rmtree(tmp, ignore_errors=True)
    json.dump(result, open(os.path.join(out, "e110_readout.json"), "w", encoding="utf-8"), indent=1, default=str)
    print("wrote", os.path.join(out, "e110_readout.json"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
