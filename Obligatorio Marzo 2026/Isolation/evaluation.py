"""evaluation.py — biblioteca de funciones de evaluacion para MATE (Paso 4 del plan).

Cada heuristica tiene firma `(board, player) -> float` y se evalua **desde la
perspectiva de `player`** (positivo = bueno para `player`, negativo = malo). Esta
convencion unica evita errores de signo en los nodos MIN / de azar y es coherente con
`Stratagem`, que tambien calcula su heuristica relativa a `self.player`.

Las cuatro componentes (lamina 16 del teorico + heuristica de `Stratagem`):

  - h1  Movilidad propia      = nº de casillas adyacentes libres del jugador.
  - h2  Diferencia de movilidad = movilidad propia − movilidad del rival (suma cero).
  - h3  Control de centro      = − distancia Manhattan del jugador al centro (Stratagem.Y).
  - h4  Acorralar al rival     = celdas destruidas alrededor del rival
                                 − celdas destruidas alrededor del jugador (Stratagem.X).

`weighted_eval(weights)` combina las componentes en una unica `eval_fn` inyectable a
`MinimaxAgent` / `ExpectimaxAgent` via su parametro `eval_fn`:  Eval(s) = Σ wᵢ·hᵢ(s).

Justificacion (ver `Documentacion/DocumentacionMATE.md` §3.4): en Isolation perder =
quedarse sin movimientos, asi que la **movilidad** es la señal mas directamente
correlacionada con ganar (criterio 3 de la lam. 16). La movilidad se mide contando
**casillas adyacentes libres** (estilo `Board.has_valid_moves`), NO
`len(get_possible_actions)`, que infla la cuenta al multiplicar por cada celda destruible.

No se modifica ningun archivo dado; este modulo solo consume la API publica de `Board`.
"""

from board import Board, empty_cell, eliminated_cell


# 8-vecindario (ortogonales + diagonales), igual que las 8 direcciones de `board.py`.
_NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1),
              (0, -1),           (0, 1),
              (1, -1),  (1, 0),  (1, 1)]


def _other(player: int) -> int:
    """Numero del jugador rival (1 -> 2, 2 -> 1)."""
    return player % 2 + 1


def _count_neighbors(board: Board, pos, value) -> int:
    """Cantidad de vecinos de `pos` (dentro del tablero) cuya celda vale `value`."""
    if pos is None:
        return 0
    rows, cols = board.board_size
    count = 0
    for dr, dc in _NEIGHBORS:
        r, c = pos[0] + dr, pos[1] + dc
        if 0 <= r < rows and 0 <= c < cols and board.grid[r, c] == value:
            count += 1
    return count


def free_adjacent(board: Board, player: int) -> int:
    """Movilidad: nº de casillas **adyacentes libres** del jugador.

    Es la primitiva de movilidad reutilizada por h1 y h2 (y como default de los agentes).
    """
    return _count_neighbors(board, board.find_player_position(player), empty_cell)


# --- Componentes heuristicos (firma (board, player) -> float) -------------------

def h1_mobility(board: Board, player: int) -> float:
    """Movilidad propia: casillas adyacentes libres del jugador (mayor = mejor)."""
    return float(free_adjacent(board, player))


def h2_mobility_diff(board: Board, player: int) -> float:
    """Diferencia de movilidad: propia − rival (captura la naturaleza de suma cero)."""
    return float(free_adjacent(board, player) - free_adjacent(board, _other(player)))


def h3_center(board: Board, player: int) -> float:
    """Control de centro: − distancia Manhattan del jugador al centro del tablero.

    Mayor (menos negativo) cuando el jugador esta mas cerca del centro, desde donde hay
    mas casillas alcanzables y se tiende a preservar movilidad futura. (= `Stratagem.Y`).
    """
    pos = board.find_player_position(player)
    if pos is None:
        return 0.0
    center = (board.board_size[0] // 2, board.board_size[1] // 2)
    return -float(abs(pos[0] - center[0]) + abs(pos[1] - center[1]))


def h4_surround(board: Board, player: int) -> float:
    """Acorralar: celdas destruidas alrededor del rival − alrededor del jugador.

    Mayor cuando el rival esta mas rodeado de casillas destruidas (mas cerca de quedarse
    sin movimientos) que el propio jugador: ataque directo a su condicion de derrota.
    (= `Stratagem.X`).
    """
    me = board.find_player_position(player)
    opp = board.find_player_position(_other(player))
    around_opp = _count_neighbors(board, opp, eliminated_cell)
    around_me = _count_neighbors(board, me, eliminated_cell)
    return float(around_opp - around_me)


# Registro de componentes por id (lo usa weighted_eval y los experimentos del Paso 6).
HEURISTICS = {
    "h1": h1_mobility,
    "h2": h2_mobility_diff,
    "h3": h3_center,
    "h4": h4_surround,
}


def weighted_eval(weights: dict):
    """Devuelve una `eval_fn(board, player) -> float` = Σ wᵢ·hᵢ(board, player).

    `weights` es un dict {"h1": w1, "h2": w2, "h3": w3, "h4": w4} (las claves ausentes
    cuentan como peso 0). El resultado se inyecta a los agentes como `eval_fn`.
    """
    unknown = set(weights) - set(HEURISTICS)
    if unknown:
        raise KeyError(f"Heuristicas desconocidas en weights: {sorted(unknown)}")
    # Solo se evaluan las componentes con peso != 0 (ahorra calculo en la busqueda).
    active = [(HEURISTICS[k], float(w)) for k, w in weights.items() if w != 0]

    def eval_fn(board: Board, player: int) -> float:
        return sum(w * h(board, player) for h, w in active)

    return eval_fn


if __name__ == "__main__":
    # Smoke test del Paso 4: boards de juguete con valor esperado conocido.
    import numpy as np

    def empty_board():
        b = Board((4, 4))
        b.grid = np.zeros((4, 4), dtype=int)
        return b

    # (1) h3_center: el centro (2,2) debe puntuar mas alto que una esquina (0,0).
    center = empty_board(); center.grid[2, 2] = 1; center.grid[0, 3] = 2
    corner = empty_board(); corner.grid[0, 0] = 1; corner.grid[0, 3] = 2
    assert h3_center(center, 1) > h3_center(corner, 1), "centro deberia superar a la esquina"
    assert h3_center(center, 1) == -0.0 or h3_center(center, 1) == 0.0  # dist 0 al centro (2,2)
    print(f"h3_center OK: centro={h3_center(center, 1)}  esquina={h3_center(corner, 1)}")

    # (2) h1/h2 movilidad: jugador en el centro abierto (8 libres) vs rival en esquina.
    b = empty_board(); b.grid[1, 1] = 1; b.grid[0, 0] = 2
    # jugador 1 en (1,1): 8 vecinos, uno ocupado por el rival (0,0) -> 7 libres.
    assert h1_mobility(b, 1) == 7.0, h1_mobility(b, 1)
    # rival 2 en (0,0): 3 vecinos validos, uno ocupado por el jugador (1,1) -> 2 libres.
    assert h1_mobility(b, 2) == 2.0, h1_mobility(b, 2)
    assert h2_mobility_diff(b, 1) == 5.0, h2_mobility_diff(b, 1)      # 7 - 2
    assert h2_mobility_diff(b, 2) == -5.0, h2_mobility_diff(b, 2)     # perspectiva opuesta
    print(f"h1/h2 OK: mov(1)={h1_mobility(b,1)}  mov(2)={h1_mobility(b,2)}  diff(1)={h2_mobility_diff(b,1)}")

    # (3) h4_surround: rival rodeado de casillas destruidas, jugador no -> positivo.
    s = empty_board(); s.grid[3, 3] = 2; s.grid[0, 0] = 1
    for cell in [(2, 2), (2, 3), (3, 2)]:        # destruyo los 3 vecinos del rival (3,3)
        s.grid[cell] = eliminated_cell
    assert h4_surround(s, 1) == 3.0, h4_surround(s, 1)   # 3 alrededor del rival, 0 del mio
    assert h4_surround(s, 2) == -3.0, h4_surround(s, 2)  # perspectiva opuesta
    print(f"h4_surround OK: desde jugador 1 = {h4_surround(s, 1)}")

    # (4) weighted_eval: linealidad y respeto de pesos.
    w = {"h1": 1.0, "h2": 2.0, "h3": 0.5, "h4": 1.0}
    ef = weighted_eval(w)
    expected = (1.0 * h1_mobility(b, 1) + 2.0 * h2_mobility_diff(b, 1)
                + 0.5 * h3_center(b, 1) + 1.0 * h4_surround(b, 1))
    assert abs(ef(b, 1) - expected) < 1e-9, (ef(b, 1), expected)
    # peso 0 = componente ignorada (mismo resultado que omitir la clave).
    assert weighted_eval({"h2": 1.0})(b, 1) == weighted_eval({"h2": 1.0, "h3": 0.0})(b, 1)
    try:
        weighted_eval({"hX": 1.0})
        raise AssertionError("deberia rechazar heuristicas desconocidas")
    except KeyError:
        pass
    print(f"weighted_eval OK: Eval(b, jugador 1) = {ef(b, 1)}")

    print("OK Paso 4")
