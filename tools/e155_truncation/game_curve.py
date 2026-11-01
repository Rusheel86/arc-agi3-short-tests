"""The E059/E072 truncation scoring rule, copied unchanged from the project's analysis/k_killtests.py (game_curve, load_runs).

A level k counts at token budget B if the cumulative generated tokens at its completing action are <= B; its score is
min(1, (human actions / agent actions)^2); a game's score = sum over finished levels of k * s_k / sum_k k, x 100.
MIT-0.
"""
import json


def game_curve(g):
    """Return (score_at(B), levels_at(B), total tokens, levels, N) for one logged game; same rule as
    scratch/a014/elasticity_curve.py: level k counts at budget B if the cumulative generated tokens at its last
    action are <= B; level score min(1, (base/actions)^2); game score = sum k s_k / sum k x 100."""
    cum, c = [], 0.0
    for h in g["history"]:
        c += float(h.get("generated_tokens") or 0)
        cum.append(c)
    apl, base, n = g["actions_per_level"], g["base_actions_per_level"], g["number_of_levels"]
    W = n * (n + 1) / 2
    done_at, idx = [], 0
    for k in range(g["levels_completed"]):
        idx += apl[k]
        done_at.append(cum[min(idx, len(cum)) - 1] if idx > 0 else 0)

    def score(B):
        return 100 * sum((k + 1) * min(1.0, (base[k] / max(1, apl[k])) ** 2) for k, t in enumerate(done_at) if t <= B) / W

    def levels(B):
        return sum(1 for t in done_at if t <= B)
    return score, levels, (cum[-1] if cum else 0.0), len(done_at), n


def load_runs(path, ids=None):
    gr = json.load(open(path, encoding="utf-8"))["game_runs"]
    return [g for g in gr if ids is None or g["game_id"][:4] in ids]
