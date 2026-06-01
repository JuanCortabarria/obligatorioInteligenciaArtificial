
# Isolation

Breve descripción de los archivos contenidos en este directorio.

- **agent.py**: Clase abstracta `Agent` que define la interfaz de los agentes.
- **board.py**: Implementación de la clase `Board` con la representación del tablero y la lógica del juego.
- **input_agent.py**: `InputAgent` sencillo para jugar manualmente desde la consola.
- **random_agent.py**: `RandomAgent` que elige una acción legal al azar usando `board.get_possible_actions()`.
- **stratagem.py**: Agente con una implementación ofuscada.
- **isolation_env.py**: Wrapper Gym `IsolationEnv` que adapta `Board` a la API.
- **play.py**: Utilidad `play_vs_other_agent` para ejecutar partidas entre dos agentes y opcionalmente mostrar el tablero en cada turno.
- **isolation.ipynb**: Notebook Jupyter con demostraciones y los experimentos del proyecto MATE (E1–E6 + gráficos + guardado del `.pkl`).

## Archivos del proyecto MATE (búsqueda adversarial)

Implementados sobre la API pública del simulador, sin modificar los archivos dados:

- **match.py**: `play_match(agent_p1, agent_p2, seed, render)` corre una partida reproducible y devuelve estadísticas (`winner`, `plies`, `avg_move_time`).
- **search.py**: núcleo de búsqueda — `successors`, `is_terminal`, `utility`, `other` y `NodeCounter` (conteo de nodos).
- **evaluation.py**: heurísticas `h1_mobility`, `h2_mobility_diff`, `h3_center`, `h4_surround` y `weighted_eval(weights)` (combina componentes por pesos).
- **minimax_agent.py**: `MinimaxAgent` (profundidad fija) con poda **Alpha-Beta** opcional (`use_alpha_beta`) y ordenamiento de movimientos.
- **expectimax_agent.py**: `ExpectimaxAgent` con nodos de azar (σ uniforme del rival).

Cada `.py` trae un *smoke test* en su `__main__`: `poetry run python <archivo>.py`.

**Artefactos generados por el notebook:** `results.csv` (registro de experimentos), `plots/*.png` (gráficos), `mate_best_config.pkl` (mejor configuración hallada).

**Entorno:** `poetry install --no-root` (instala dependencias sin empaquetar el proyecto).
