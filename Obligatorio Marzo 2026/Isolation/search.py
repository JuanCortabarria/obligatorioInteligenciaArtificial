from board import Board


class NodeCounter:
    """Cuenta nodos expandidos (para el analisis de impacto de Alpha-Beta)."""

    def __init__(self):
        self.nodes = 0

    def tick(self):
        self.nodes += 1

    def reset(self):
        self.nodes = 0


def successors(board: Board, player: int):
    """Lista de (accion, board_hijo) para todas las acciones legales de `player`."""
    result = []
    for action in board.get_possible_actions(player):
        child = board.clone()
        ok = child.play(action, player)
        if not ok:
            raise ValueError(f"Accion ilegal generada: {action}")
        result.append((action, child))
    return result


def is_terminal(board: Board, player_to_move: int):
    """(done, winner) usando la logica del simulador: pierde quien no tiene movidas."""
    return board.is_end(player_to_move)   # -> (bool, 1|2|0)


def utility(winner: int, agent_player: int) -> int:
    """Utilidad terminal desde la perspectiva del agente: +1 gana, -1 pierde.

    En Isolation siempre hay un ganador en estado final (no hay empate).
    """
    return 1 if winner == agent_player else -1


def other(player: int) -> int:
    """Devuelve el numero del jugador rival (1 -> 2, 2 -> 1)."""
    return player % 2 + 1


if __name__ == "__main__":
    # Smoke test del Paso 1.
    import numpy as np
    from board import eliminated_cell

    # 1) successors coincide en cantidad con get_possible_actions y produce Boards.
    b = Board((4, 4))
    p = 1
    succ = successors(b, p)
    assert len(succ) == len(b.get_possible_actions(p))
    assert all(isinstance(child, Board) for _, child in succ)
    # el tablero original no se modifica al generar sucesores
    assert b.find_player_position(p) is not None
    print(f"successors OK: {len(succ)} sucesores para el jugador {p}")

    # 2) utilidad terminal: acorralar al jugador 1 en la esquina (0,0).
    t = Board((4, 4))
    t.grid = np.zeros((4, 4), dtype=int)
    t.grid[0, 0] = 1                        # jugador 1 en la esquina
    t.grid[3, 3] = 2                        # jugador 2 lejos
    for cell in [(0, 1), (1, 0), (1, 1)]:   # destruyo sus 3 vecinos
        t.grid[cell] = eliminated_cell
    done, winner = is_terminal(t, 1)        # puede mover el jugador 1?
    assert done and winner == 2, (done, winner)
    assert utility(winner, agent_player=1) == -1
    assert utility(winner, agent_player=2) == 1
    print("utilidad terminal OK: jugador 1 acorralado -> gana 2")

    # 3) other()
    assert other(1) == 2 and other(2) == 1
    print("other OK")

    print("OK Paso 1")
