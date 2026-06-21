"""Minimal executable checks for the MATE Isolation project.

Run with:
    poetry run python smoke_tests.py

The tests are intentionally lightweight and deterministic. They cover the game
rules, terminal utility, both player perspectives, Minimax/Alpha-Beta
equivalence, and Expectimax's uniform chance-node calculation.
"""

from __future__ import annotations

import numpy as np

from board import Board, eliminated_cell
from expectimax_agent import ExpectimaxAgent
from minimax_agent import MinimaxAgent
from search import is_terminal, utility


def controlled_board() -> Board:
    board = Board((4, 4))
    board.grid = np.zeros((4, 4), dtype=int)
    board.grid[1, 1] = 1
    board.grid[3, 3] = 2
    board.grid[0, 0] = eliminated_cell
    return board


def test_play_action_moves_and_eliminates():
    board = controlled_board()
    action = (0, (2, 2))  # player 1 moves up, then eliminates a free cell
    assert board.play(action, 1)
    assert board.grid[0, 1] == 1
    assert board.grid[1, 1] == eliminated_cell
    assert board.grid[2, 2] == eliminated_cell


def test_illegal_elimination_is_rejected():
    board = controlled_board()
    action = (0, (0, 1))  # destination cell cannot also be eliminated
    assert not board.play(action, 1)


def test_terminal_utility_perspective():
    board = Board((4, 4))
    board.grid = np.zeros((4, 4), dtype=int)
    board.grid[0, 0] = 1
    board.grid[3, 3] = 2
    for cell in [(0, 1), (1, 0), (1, 1)]:
        board.grid[cell] = eliminated_cell

    done, winner = is_terminal(board, 1)
    assert done and winner == 2
    assert utility(winner, agent_player=1) == -1
    assert utility(winner, agent_player=2) == 1


def test_agents_return_legal_actions_as_both_players():
    board = controlled_board()

    p1 = MinimaxAgent(1, depth=1, use_alpha_beta=True)
    action1 = p1.next_action(board)
    assert action1 in board.get_possible_actions(1)

    p2 = ExpectimaxAgent(2, depth=1)
    action2 = p2.next_action(board)
    assert action2 in board.get_possible_actions(2)


def test_alpha_beta_matches_minimax_without_ordering():
    board = controlled_board()
    mm = MinimaxAgent(1, depth=2, use_alpha_beta=False)
    ab = MinimaxAgent(1, depth=2, use_alpha_beta=True, move_ordering=False)

    mm.counter.reset()
    mm_action, mm_value = mm._minimax(board, 1, 2)
    ab.counter.reset()
    ab_action, ab_value = ab._alphabeta(board, 1, 2, float("-inf"), float("inf"))

    assert ab_value == mm_value
    assert ab_action == mm_action
    assert ab.counter.nodes <= mm.counter.nodes


class ToyBoard:
    """Tiny deterministic tree for checking Expectimax expectation.

    Root is player 1. Action "safe" leads to two opponent outcomes that both
    win for player 1. Action "risky" leads to one win and one loss, so its
    uniform expectation is 0. Expectimax must choose "safe"; a MIN node would
    value "risky" as -1.
    """

    transitions = {
        ("root", 1, "risky"): "chance_risky",
        ("root", 1, "safe"): "chance_safe",
        ("chance_risky", 2, "good"): "win",
        ("chance_risky", 2, "bad"): "loss",
        ("chance_safe", 2, "good"): "win",
        ("chance_safe", 2, "also_good"): "win",
    }

    actions = {
        ("root", 1): ["risky", "safe"],
        ("chance_risky", 2): ["good", "bad"],
        ("chance_safe", 2): ["good", "also_good"],
    }

    def __init__(self, state="root"):
        self.state = state

    def clone(self):
        return ToyBoard(self.state)

    def play(self, action, player):
        self.state = self.transitions[(self.state, player, action)]
        return True

    def get_possible_actions(self, player):
        return self.actions.get((self.state, player), [])

    def is_end(self, player):
        if self.state == "win":
            return True, 1
        if self.state == "loss":
            return True, 2
        return False, 0


def test_expectimax_uses_uniform_expectation():
    agent = ExpectimaxAgent(1, depth=2)
    action, value = agent._expectimax(ToyBoard(), 1, 2)
    assert action == "safe"
    assert value == 1.0


def main():
    tests = [
        test_play_action_moves_and_eliminates,
        test_illegal_elimination_is_rejected,
        test_terminal_utility_perspective,
        test_agents_return_legal_actions_as_both_players,
        test_alpha_beta_matches_minimax_without_ordering,
        test_expectimax_uses_uniform_expectation,
    ]
    for test in tests:
        test()
        print(f"OK {test.__name__}")


if __name__ == "__main__":
    main()
