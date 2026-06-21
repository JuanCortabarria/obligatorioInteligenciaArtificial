"""Run the required MATE matchups with explicit per-player metadata.

This script complements the notebook experiments. It does not bake in any
results: it runs real games and writes a CSV that makes player order explicit
(`player1_agent`, `player2_agent`) so comparisons like Random vs Minimax and
Minimax vs Random are visible as separate rows.

Example:
    poetry run python run_required_experiments.py --seeds 20

For a quick smoke run:
    poetry run python run_required_experiments.py --seeds 2 --depths 2 \
        --heuristics solo_mov_diff --minimax-ab-modes on
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from evaluation import weighted_eval
from expectimax_agent import ExpectimaxAgent
from match import play_match
from minimax_agent import MinimaxAgent
from random_agent import RandomAgent
from stratagem import Stratagem


HEURISTIC_PRESETS = {
    "solo_mov_diff": {"h2": 1.0},
    "mov+centro": {"h2": 1.0, "h3": 0.5},
    "mov+acorralar": {"h2": 1.0, "h4": 1.0},
    "balanceada": {"h1": 1.0, "h2": 2.0, "h3": 0.5, "h4": 1.0},
}


REQUIRED_MATCHUPS = [
    ("Minimax", "Minimax"),
    ("Expectimax", "Minimax"),
    ("Minimax", "Expectimax"),
    ("Expectimax", "Expectimax"),
    ("Minimax", "Random"),
    ("Random", "Minimax"),
    ("Expectimax", "Random"),
    ("Random", "Expectimax"),
    ("Minimax", "Stratagem"),
    ("Stratagem", "Minimax"),
    ("Expectimax", "Stratagem"),
    ("Stratagem", "Expectimax"),
]


def moves_of(player: int, plies: int) -> int:
    """Number of moves made by `player` in a game with `plies` total moves."""
    return (plies + 1) // 2 if player == 1 else plies // 2


def build_agent(kind: str, player: int, depth: int, heuristic_name: str, minimax_ab: bool):
    """Create an agent for a concrete player slot."""
    if kind == "Minimax":
        return MinimaxAgent(
            player,
            depth=depth,
            eval_fn=weighted_eval(HEURISTIC_PRESETS[heuristic_name]),
            use_alpha_beta=minimax_ab,
        )
    if kind == "Expectimax":
        return ExpectimaxAgent(
            player,
            depth=depth,
            eval_fn=weighted_eval(HEURISTIC_PRESETS[heuristic_name]),
        )
    if kind == "Random":
        return RandomAgent(player)
    if kind == "Stratagem":
        return Stratagem(player)
    raise ValueError(f"Unknown agent kind: {kind}")


def agent_metadata(kind: str, depth: int, heuristic_name: str, minimax_ab: bool):
    """Metadata values for CSV columns."""
    if kind == "Minimax":
        return depth, heuristic_name, minimax_ab
    if kind == "Expectimax":
        return depth, heuristic_name, ""
    if kind == "Stratagem":
        return 3, "stratagem_internal", "internal"
    return "", "", ""


def run_one(p1_kind: str, p2_kind: str, seed: int, depth: int, heuristic_name: str, minimax_ab: bool):
    """Run one game and return a complete CSV row."""
    p1 = build_agent(p1_kind, 1, depth, heuristic_name, minimax_ab)
    p2 = build_agent(p2_kind, 2, depth, heuristic_name, minimax_ab)
    result = play_match(p1, p2, seed=seed)

    p1_moves = moves_of(1, result["plies"]) or 1
    p2_moves = moves_of(2, result["plies"]) or 1
    p1_nodes = getattr(p1, "total_nodes", "")
    p2_nodes = getattr(p2, "total_nodes", "")

    p1_depth, p1_heuristic, p1_ab = agent_metadata(p1_kind, depth, heuristic_name, minimax_ab)
    p2_depth, p2_heuristic, p2_ab = agent_metadata(p2_kind, depth, heuristic_name, minimax_ab)

    return {
        "seed": seed,
        "player1_agent": p1_kind,
        "player2_agent": p2_kind,
        "matchup": f"{p1_kind} vs {p2_kind}",
        "depth": depth,
        "heuristic": heuristic_name,
        "minimax_alpha_beta": minimax_ab,
        "p1_depth": p1_depth,
        "p2_depth": p2_depth,
        "p1_heuristic": p1_heuristic,
        "p2_heuristic": p2_heuristic,
        "p1_alpha_beta": p1_ab,
        "p2_alpha_beta": p2_ab,
        "winner": result["winner"],
        "p1_won": int(result["winner"] == 1),
        "p2_won": int(result["winner"] == 2),
        "plies": result["plies"],
        "avg_move_time": result["avg_move_time"],
        "p1_avg_move_time": result["avg_move_time_p1"],
        "p2_avg_move_time": result["avg_move_time_p2"],
        "p1_nodes_per_move": (p1_nodes / p1_moves) if p1_nodes != "" else "",
        "p2_nodes_per_move": (p2_nodes / p2_moves) if p2_nodes != "" else "",
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="required_matchups_results.csv")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=5000)
    parser.add_argument("--depths", type=int, nargs="+", default=[2, 3])
    parser.add_argument(
        "--heuristics",
        nargs="+",
        choices=sorted(HEURISTIC_PRESETS),
        default=["solo_mov_diff", "balanceada"],
    )
    parser.add_argument(
        "--minimax-ab-modes",
        nargs="+",
        choices=["on", "off"],
        default=["on", "off"],
        help="Run Minimax with alpha-beta on/off. Ignored by non-Minimax agents.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output = Path(args.output)
    rows = []

    for depth in args.depths:
        for heuristic_name in args.heuristics:
            for ab_mode in args.minimax_ab_modes:
                minimax_ab = ab_mode == "on"
                for p1_kind, p2_kind in REQUIRED_MATCHUPS:
                    for i in range(args.seeds):
                        seed = args.base_seed + i
                        rows.append(run_one(p1_kind, p2_kind, seed, depth, heuristic_name, minimax_ab))

    with output.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
