import time
import random

from isolation_env import IsolationEnv


def play_match(agent_p1, agent_p2, seed=None, render=False):
    """
    Corre UNA partida entre dos agentes ya configurados.

      - `agent_p1` debe tener `player == 1` (mueve primero).
      - `agent_p2` debe tener `player == 2`.

    Para alternar quien arranca, intercambiá los roles al construir los agentes
    (es decir, pasá como jugador 1 a la estrategia que querés que arranque).

    Devuelve un dict con estadisticas de la partida.
    """
    assert agent_p1.player == 1 and agent_p2.player == 2, \
        "agent_p1 debe ser jugador 1 y agent_p2 jugador 2"

    if seed is not None:
        random.seed(seed)            # reproducibilidad: fija la colocacion inicial y el RNG

    env = IsolationEnv()
    obs = env.reset()
    agents = {1: agent_p1, 2: agent_p2}

    done = False
    winner = 0
    plies = 0
    move_times = {1: [], 2: []}          # tiempos por jugada, separados por jugador

    while not done:
        if render:
            env.render()
        current = env.current_player
        t0 = time.perf_counter()
        action = agents[current].next_action(obs)
        move_times[current].append(time.perf_counter() - t0)
        obs, _, done, winner, _ = env.step(action)
        plies += 1

    if render:
        env.render()

    def _avg(ts):
        return sum(ts) / len(ts) if ts else 0.0

    all_times = move_times[1] + move_times[2]
    return {
        "winner": winner,                                   # 1 o 2
        "plies": plies,                                     # jugadas totales
        "avg_move_time": _avg(all_times),                   # promedio sobre AMBOS jugadores
        "avg_move_time_p1": _avg(move_times[1]),            # costo/jugada del jugador 1
        "avg_move_time_p2": _avg(move_times[2]),            # costo/jugada del jugador 2
        "seed": seed,
    }


if __name__ == "__main__":
    # Smoke test del Paso 0: reproducibilidad con misma seed.
    from random_agent import RandomAgent

    r1 = play_match(RandomAgent(player=1), RandomAgent(player=2), seed=42)
    r2 = play_match(RandomAgent(player=1), RandomAgent(player=2), seed=42)
    print("run1:", r1)
    print("run2:", r2)
    assert r1["winner"] == r2["winner"] and r1["plies"] == r2["plies"], \
        "NO reproducible: misma seed deberia dar el mismo resultado"

    # Sanity extra: distinta seed puede dar distinto resultado (no es obligatorio que difiera).
    r3 = play_match(RandomAgent(player=1), RandomAgent(player=2), seed=7)
    print("run3 (seed distinta):", r3)

    print("OK Paso 0: partidas reproducibles con misma seed")
