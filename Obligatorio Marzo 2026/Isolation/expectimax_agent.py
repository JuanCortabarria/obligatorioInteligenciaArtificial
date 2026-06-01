"""ExpectimaxAgent — busqueda adversarial con nodos de azar (Paso 5 del plan MATE).

Igual estructura que Minimax (MiniMax.md, lam. 13) pero modela al rival como un jugador
**estocastico**: sus nodos dejan de ser `min` y pasan a ser **nodos de azar** (lam. 8):

    V(s, d) = Utilidad(s)                              si EsFinal(s)
            = Eval(s)                                  si d = 0
            = max_a V(Suc(s, a), d-1)                  si juega el agente (nodo MAX)
            = Σ_a σ(s, a) · V(Suc(s, a), d-1)          si juega el rival  (nodo de AZAR)

con `σ` la **politica del rival**. Aca se asume σ **uniforme** sobre las acciones
legales (1/n), que es el modelo correcto para un rival como `RandomAgent`.

Diferencia conceptual con Minimax (ver `DocumentacionMATE.md` §3.3): Minimax supone un
rival **adversarial/optimo** (cota inferior garantizada, lam. 12); Expectimax supone un
rival **estocastico**. Por eso se espera que Expectimax **rinda mejor vs RandomAgent**
(modelo correcto) y **no necesariamente vs Stratagem** (que es Minimax determinista, no
uniforme). Esa comparacion es justamente la que deciden los experimentos E2–E4.

Nota: en Expectimax **no** se aplica Alpha-Beta como en Minimax (los nodos de azar
promedian todas las ramas, no hay corte por cota), por eso esta clase no tiene ese flag.

Apoyado en `search.py` y en la interfaz `Agent`. No se modifica ningun archivo dado.
"""

from agent import Agent
from board import Board
from search import successors, is_terminal, utility, other, NodeCounter
from evaluation import h2_mobility_diff


class ExpectimaxAgent(Agent):
    """Agente Expectimax de profundidad fija.

    Parametros:
      - `player`: 1 o 2 (lo fija el simulador al construir el agente).
      - `depth`: profundidad de busqueda d >= 1 (default 3).
      - `eval_fn`: funcion de evaluacion inyectable `(board, player) -> float` desde la
        perspectiva del agente. Si es `None`, default = diferencia de movilidad
        (`h2` de `evaluation.py`). Misma interfaz que `MinimaxAgent`: las heuristicas
        ponderadas se inyectan por aca en los experimentos.

    Instrumentacion (mismas metricas que Minimax, para comparar costo):
      - `self.nodes_last_move`: nodos visitados en la ultima jugada.
      - `self.total_nodes`: nodos visitados acumulados en la partida.
    """

    def __init__(self, player, depth=3, eval_fn=None):
        super().__init__(player)
        assert depth >= 1, "La profundidad debe ser >= 1"
        self.depth = depth
        self.eval_fn = eval_fn
        self.counter = NodeCounter()
        self.nodes_last_move = 0
        self.total_nodes = 0

    # --- Funcion de evaluacion (misma convencion que MinimaxAgent) -------------

    def heuristic_utility(self, board: Board) -> float:
        """Evalua un estado no terminal desde la perspectiva del agente."""
        if self.eval_fn is not None:
            return self.eval_fn(board, self.player)
        return h2_mobility_diff(board, self.player)

    # --- Busqueda --------------------------------------------------------------

    def next_action(self, obs):
        """Devuelve la accion de mayor **valor esperado** segun Expectimax."""
        board = obs
        self.counter.reset()
        action, _ = self._expectimax(board, self.player, self.depth)
        self.nodes_last_move = self.counter.nodes
        self.total_nodes += self.counter.nodes
        # Invariante: si el agente esta por mover y el juego no termino, hay >= 1 accion.
        assert action is not None, "next_action no encontro accion legal (estado terminal?)"
        return action

    def _expectimax(self, board: Board, player_to_move: int, depth: int):
        """Devuelve `(mejor_accion, valor)` para `board` con `player_to_move` por mover.

        El valor esta en la perspectiva del agente (`self.player`). En los nodos del
        rival se devuelve `(None, valor_esperado)`: no hay una "mejor" accion del rival,
        sino el promedio ponderado de sus jugadas (σ uniforme).
        """
        self.counter.tick()  # mismo conteo por nodo que Minimax, para comparar costo

        done, winner = is_terminal(board, player_to_move)
        if done:
            return None, utility(winner, self.player)
        if depth == 0:
            return None, self.heuristic_utility(board)

        if player_to_move == self.player:          # nodo MAX (juega el agente)
            best_action = None
            best_val = float("-inf")
            for action, child in successors(board, player_to_move):
                _, v = self._expectimax(child, other(player_to_move), depth - 1)
                if v > best_val:
                    best_val, best_action = v, action
            return best_action, best_val
        else:                                      # nodo de AZAR (juega el rival)
            succ = successors(board, player_to_move)
            n = len(succ)                          # n >= 1: si fuese 0, is_terminal habria cortado
            expected = 0.0
            prob = 1.0 / n                         # σ uniforme sobre las n acciones legales
            for action, child in succ:
                _, v = self._expectimax(child, other(player_to_move), depth - 1)
                expected += prob * v
            return None, expected


if __name__ == "__main__":
    # Smoke test del Paso 5.
    from match import play_match
    from random_agent import RandomAgent
    from stratagem import Stratagem

    # (1) Expectimax debe vencer consistentemente a RandomAgent (modelo de rival correcto).
    DEPTH = 2
    N = 30
    wins = 0
    for seed in range(N):
        if seed % 2 == 0:
            res = play_match(ExpectimaxAgent(player=1, depth=DEPTH),
                             RandomAgent(player=2), seed=seed)
            mate_won = (res["winner"] == 1)
        else:
            res = play_match(RandomAgent(player=1),
                             ExpectimaxAgent(player=2, depth=DEPTH), seed=seed)
            mate_won = (res["winner"] == 2)
        wins += int(mate_won)
    win_rate = wins / N
    print(f"[Paso 5] ExpectimaxAgent(d={DEPTH}) vs RandomAgent: {wins}/{N} = {win_rate:.0%} win rate")
    assert win_rate >= 0.8, f"win rate demasiado bajo: {win_rate:.0%}"

    # (2) Debe correr sin errores frente a Stratagem (no se exige ganar; ver E3).
    #     OJO: Stratagem tiene el nombre del parametro ofuscado -> se pasa POSICIONAL,
    #     no con `player=` (kwarg). Default = jugador 2.
    for seed in range(2):
        res = play_match(ExpectimaxAgent(player=1, depth=DEPTH),
                         Stratagem(2), seed=seed)
        assert res["winner"] in (1, 2)
    print(f"[Paso 5] Expectimax vs Stratagem: corrio sin errores ({res})")

    print("OK Paso 5: Expectimax vence a Random y corre frente a Stratagem")
