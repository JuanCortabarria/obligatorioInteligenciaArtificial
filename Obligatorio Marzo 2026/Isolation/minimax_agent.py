"""MinimaxAgent — busqueda adversarial con profundidad fija (Pasos 2 y 3 del plan MATE).

Implementa el modelo del teorico (MiniMax.md, lamina 13), V_max,min(s, d):

    V(s, d) = Utilidad(s)                         si EsFinal(s)
            = Eval(s)                             si d = 0
            = max_a V(Suc(s, a), d-1)             si juega el agente (nodo MAX)
            = min_a V(Suc(s, a), d-1)             si juega el rival  (nodo MIN)

Dos modos sobre el MISMO nucleo de busqueda, seleccionables con el flag `use_alpha_beta`:

  - **Minimax puro** (`use_alpha_beta=False`): explora el arbol completo hasta `depth`.
    Es la *referencia de correccion*.
  - **Alpha-Beta** (`use_alpha_beta=True`, Paso 3): poda ramas que no pueden afectar la
    decision. Devuelve un movimiento del **mismo valor** que Minimax expandiendo
    **<= nodos**. Con ordenamiento de movimientos (`move_ordering=True`) la poda es mayor.

Invariantes que se verifican en el smoke test (`__main__`):
  - mismo **valor** en la raiz (Alpha-Beta no cambia la decision, solo el costo);
  - **nodos(AB) <= nodos(Minimax)** siempre;
  - con `move_ordering=False`, Alpha-Beta reproduce **exactamente la misma accion** que
    Minimax (mismo orden de sucesores y mismo desempate); con ordenamiento puede elegir
    otra accion del mismo valor optimo, pero podando mas.

Apoyado en `search.py` (sucesores, utilidad terminal, contador de nodos) y en la
interfaz `Agent` (`next_action`, `heuristic_utility`). No se modifica ningun
archivo dado.
"""

from agent import Agent
from board import Board
from search import successors, is_terminal, utility, other, NodeCounter
from evaluation import h2_mobility_diff


class MinimaxAgent(Agent):
    """Agente Minimax de profundidad fija.

    Parametros:
      - `player`: 1 o 2 (lo fija el simulador al construir el agente).
      - `depth`: profundidad de busqueda d >= 1 (default 3, igual que `Stratagem`).
      - `eval_fn`: funcion de evaluacion inyectable con firma `(board, player) -> float`,
        desde la perspectiva del agente (positivo = bueno para el agente). Si es `None`
        se usa una heuristica por defecto (diferencia de movilidad, = `h2` de
        `evaluation.py`). Las heuristicas combinadas (h1..h4 ponderadas, via
        `evaluation.weighted_eval`) se inyectan por este parametro en los experimentos.
      - `use_alpha_beta`: si `True`, usa poda Alpha-Beta; si `False`, Minimax puro.
      - `move_ordering`: si `True` (default), ordena los sucesores por su evaluacion para
        maximizar la poda (solo aplica con `use_alpha_beta=True`). Si `False`, conserva el
        orden de `get_possible_actions` (util para la prueba de equivalencia con Minimax).

    Instrumentacion (para el analisis de impacto de Alpha-Beta, experimento E1):
      - `self.nodes_last_move`: nodos visitados en la ultima jugada.
      - `self.total_nodes`: nodos visitados acumulados en toda la partida.
    """

    def __init__(self, player, depth=3, eval_fn=None, use_alpha_beta=False,
                 move_ordering=True):
        super().__init__(player)
        assert depth >= 1, "La profundidad debe ser >= 1"
        self.depth = depth
        self.eval_fn = eval_fn
        self.use_alpha_beta = use_alpha_beta
        self.move_ordering = move_ordering
        self.counter = NodeCounter()
        self.nodes_last_move = 0
        self.total_nodes = 0

    # --- Funcion de evaluacion -------------------------------------------------

    def heuristic_utility(self, board: Board) -> float:
        """Evalua un estado no terminal desde la perspectiva del agente.

        Delega en `self.eval_fn` si fue inyectada; si no, usa la heuristica por
        defecto: diferencia de movilidad (mis casillas libres - las del rival).
        """
        if self.eval_fn is not None:
            return self.eval_fn(board, self.player)
        # Default: diferencia de movilidad (= h2 de evaluation.py). Las heuristicas
        # combinadas (h1..h4 ponderadas) se inyectan por `eval_fn` en los experimentos.
        return h2_mobility_diff(board, self.player)

    # --- Busqueda --------------------------------------------------------------

    def next_action(self, obs):
        """Devuelve la mejor accion segun Minimax (con o sin poda) de profundidad `self.depth`."""
        board = obs
        self.counter.reset()
        if self.use_alpha_beta:
            action, _ = self._alphabeta(board, self.player, self.depth,
                                        float("-inf"), float("inf"))
        else:
            action, _ = self._minimax(board, self.player, self.depth)
        self.nodes_last_move = self.counter.nodes
        self.total_nodes += self.counter.nodes
        # Invariante: si el agente esta por mover y el juego no termino, hay >= 1 accion.
        assert action is not None, "next_action no encontro accion legal (estado terminal?)"
        return action

    def _minimax(self, board: Board, player_to_move: int, depth: int):
        """Devuelve `(mejor_accion, valor)` para `board` con `player_to_move` por mover.

        El valor siempre esta en la perspectiva del agente (`self.player`).
        Estructura de `(accion, valor)` analoga a la de `Stratagem`.
        """
        self.counter.tick()  # se cuenta cada nodo visitado por la busqueda

        done, winner = is_terminal(board, player_to_move)
        if done:
            return None, utility(winner, self.player)
        if depth == 0:
            return None, self.heuristic_utility(board)

        best_action = None
        if player_to_move == self.player:          # nodo MAX (juega el agente)
            best_val = float("-inf")
            for action, child in successors(board, player_to_move):
                _, v = self._minimax(child, other(player_to_move), depth - 1)
                if v > best_val:
                    best_val, best_action = v, action
            return best_action, best_val
        else:                                      # nodo MIN (juega el rival)
            best_val = float("inf")
            for action, child in successors(board, player_to_move):
                _, v = self._minimax(child, other(player_to_move), depth - 1)
                if v < best_val:
                    best_val, best_action = v, action
            return best_action, best_val

    # --- Alpha-Beta (Paso 3) ---------------------------------------------------

    def _ordered_successors(self, board: Board, player_to_move: int):
        """Sucesores ordenados para maximizar la poda (si `move_ordering`).

        La poda Alpha-Beta es maxima cuando se exploran primero las jugadas mas
        prometedoras. En un nodo MAX (juega el agente) conviene mirar primero las de
        **mayor** evaluacion; en un nodo MIN (juega el rival) las de **menor**. El
        ordenamiento NO afecta la correccion (el valor final es el mismo), solo cuanto
        se poda. Si `move_ordering=False` se conserva el orden original.
        """
        succ = successors(board, player_to_move)
        if not self.move_ordering:
            return succ
        reverse = (player_to_move == self.player)   # MAX primero los mayores; MIN los menores
        succ.sort(key=lambda ac: self.heuristic_utility(ac[1]), reverse=reverse)
        return succ

    def _alphabeta(self, board: Board, player_to_move: int, depth: int,
                   alpha: float, beta: float):
        """Igual que `_minimax` pero con poda Alpha-Beta.

        `alpha` = mejor valor ya asegurado para MAX en el camino; `beta` = mejor (menor)
        valor ya asegurado para MIN. Se poda en cuanto una rama no puede mejorar la
        decision (ventana cerrada). Devuelve `(mejor_accion, valor)` en la perspectiva
        del agente, identico en valor al de `_minimax`.
        """
        self.counter.tick()  # mismo conteo que `_minimax`: un nodo por llamada

        done, winner = is_terminal(board, player_to_move)
        if done:
            return None, utility(winner, self.player)
        if depth == 0:
            return None, self.heuristic_utility(board)

        best_action = None
        if player_to_move == self.player:          # nodo MAX (juega el agente)
            best_val = float("-inf")
            for action, child in self._ordered_successors(board, player_to_move):
                _, v = self._alphabeta(child, other(player_to_move), depth - 1, alpha, beta)
                if v > best_val:
                    best_val, best_action = v, action
                if best_val > beta:                # poda beta: MIN no permitiria llegar aca
                    break
                alpha = max(alpha, best_val)
            return best_action, best_val
        else:                                      # nodo MIN (juega el rival)
            best_val = float("inf")
            for action, child in self._ordered_successors(board, player_to_move):
                _, v = self._alphabeta(child, other(player_to_move), depth - 1, alpha, beta)
                if v < best_val:
                    best_val, best_action = v, action
                if best_val < alpha:               # poda alpha: MAX no elegiria esta rama
                    break
                beta = min(beta, best_val)
            return best_action, best_val


if __name__ == "__main__":
    import random
    from match import play_match
    from random_agent import RandomAgent
    from isolation_env import IsolationEnv

    # =====================================================================
    # Paso 2 — MinimaxAgent (sin poda) vence consistentemente a RandomAgent.
    # =====================================================================
    DEPTH = 2          # d=2 alcanza para dominar a Random y mantiene el test rapido
    N = 30             # mitad arrancando el Minimax, mitad arrancando el Random
    wins = 0
    for seed in range(N):
        if seed % 2 == 0:
            res = play_match(MinimaxAgent(player=1, depth=DEPTH),
                             RandomAgent(player=2), seed=seed)
            mate_won = (res["winner"] == 1)
        else:
            res = play_match(RandomAgent(player=1),
                             MinimaxAgent(player=2, depth=DEPTH), seed=seed)
            mate_won = (res["winner"] == 2)
        wins += int(mate_won)

    win_rate = wins / N
    print(f"[Paso 2] MinimaxAgent(d={DEPTH}) vs RandomAgent: {wins}/{N} = {win_rate:.0%} win rate")
    assert win_rate >= 0.8, f"win rate demasiado bajo: {win_rate:.0%}"
    print("OK Paso 2: Minimax (sin poda) vence consistentemente a Random")

    # =====================================================================
    # Paso 3 — Equivalencia Minimax vs Alpha-Beta + reduccion de nodos.
    #   Invariantes a igual profundidad y mismo estado:
    #     (1) mismo VALOR en la raiz (la poda no cambia la decision);
    #     (2) nodos(AB) <= nodos(Minimax) siempre;
    #     (3) con move_ordering=False, AB elige EXACTAMENTE la misma accion.
    # =====================================================================
    def make_board(seed, plies):
        """Estado de juego tras `plies` jugadas aleatorias desde un inicio sembrado."""
        random.seed(seed)
        env = IsolationEnv()
        env.reset()
        for _ in range(plies):
            done, _ = env.is_done()
            if done:
                break
            p = env.current_player
            env.step(random.choice(env.grid.get_possible_actions(p)))
        done, _ = env.is_done()
        return env.grid, env.current_player, done

    AB_DEPTH = 3
    K = 8
    total_mm_nodes = total_ab_nodes = 0
    tested = 0
    for seed in range(K):
        board, mover, done = make_board(seed, plies=2 + seed % 3)  # 2-4 aperturas
        if done:
            continue  # estado ya terminal: no hay decision que comparar

        mm = MinimaxAgent(player=mover, depth=AB_DEPTH)
        mm.counter.reset()
        mm_act, mm_val = mm._minimax(board, mover, AB_DEPTH)
        mm_nodes = mm.counter.nodes

        ab = MinimaxAgent(player=mover, depth=AB_DEPTH)  # ordenamiento ON (default)
        ab.counter.reset()
        ab_act, ab_val = ab._alphabeta(board, mover, AB_DEPTH, float("-inf"), float("inf"))
        ab_nodes = ab.counter.nodes

        assert ab_val == mm_val, f"(seed={seed}) valor distinto: mm={mm_val} ab={ab_val}"
        assert ab_nodes <= mm_nodes, f"(seed={seed}) AB expandio mas: {ab_nodes} > {mm_nodes}"

        ab_no = MinimaxAgent(player=mover, depth=AB_DEPTH, move_ordering=False)
        ab_no.counter.reset()
        ab_no_act, _ = ab_no._alphabeta(board, mover, AB_DEPTH, float("-inf"), float("inf"))
        assert ab_no_act == mm_act, \
            f"(seed={seed}) AB sin orden eligio distinto: {ab_no_act} != {mm_act}"

        total_mm_nodes += mm_nodes
        total_ab_nodes += ab_nodes
        tested += 1

    reduction = 1 - total_ab_nodes / total_mm_nodes
    print(f"[Paso 3] Equivalencia OK en {tested} estados (d={AB_DEPTH}). "
          f"Nodos Minimax={total_mm_nodes}, Alpha-Beta={total_ab_nodes} "
          f"-> poda {reduction:.0%}")
    assert total_ab_nodes <= total_mm_nodes
    print("OK Paso 3: Alpha-Beta da el mismo valor que Minimax expandiendo menos nodos")
