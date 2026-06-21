
# Isolation

Breve descripción de los archivos contenidos en este directorio.

- **agent.py**: Clase abstracta `Agent` que define la interfaz de los agentes.
- **board.py**: Implementación de la clase `Board` con la representación del tablero y la lógica del juego.
- **input_agent.py**: `InputAgent` sencillo para jugar manualmente desde la consola.
- **random_agent.py**: `RandomAgent` que elige una acción legal al azar usando `board.get_possible_actions()`.
- **stratagem.py**: Agente con una implementación ofuscada.
- **isolation_env.py**: Wrapper Gym `IsolationEnv` que adapta `Board` a la API.
- **play.py**: Utilidad `play_vs_other_agent` para ejecutar partidas entre dos agentes y opcionalmente mostrar el tablero en cada turno.
- **isolation.ipynb**: Notebook Jupyter con demostraciones y los experimentos del proyecto MATE (E1–E6 + gráficos).

## Archivos del proyecto MATE (búsqueda adversarial)

Implementados sobre la API pública del simulador, sin modificar los archivos dados:

- **match.py**: `play_match(agent_p1, agent_p2, seed, render)` corre una partida reproducible y devuelve estadísticas (`winner`, `plies`, `avg_move_time`).
- **search.py**: núcleo de búsqueda — `successors`, `is_terminal`, `utility`, `other` y `NodeCounter` (conteo de nodos).
- **evaluation.py**: heurísticas `h1_mobility`, `h2_mobility_diff`, `h3_center`, `h4_surround` y `weighted_eval(weights)` (combina componentes por pesos).
- **minimax_agent.py**: `MinimaxAgent` (profundidad fija) con poda **Alpha-Beta** opcional (`use_alpha_beta`) y ordenamiento de movimientos.
- **expectimax_agent.py**: `ExpectimaxAgent` con nodos de azar (σ uniforme del rival).
- **smoke_tests.py**: verificaciones mínimas de reglas, utilidad terminal, Minimax/Alpha-Beta y Expectimax.
- **run_required_experiments.py**: runner reproducible para las comparaciones exigidas por la cátedra, con columnas explícitas para jugador 1 y jugador 2.

Cada `.py` trae un *smoke test* en su `__main__`: `poetry run python <archivo>.py`.

## Ejecución

```bash
poetry install --no-root
poetry run python smoke_tests.py
poetry run python minimax_agent.py
poetry run python expectimax_agent.py
```

Para correr las comparaciones explícitas pedidas por el profesor y generar un CSV nuevo:

```bash
poetry run python run_required_experiments.py --seeds 20
```

Para una corrida rápida de prueba del runner:

```bash
poetry run python run_required_experiments.py --seeds 2 --depths 2 --heuristics solo_mov_diff --minimax-ab-modes on
```

**Artefactos existentes:** `results.csv` (registro de experimentos E1–E6) y `plots/*.png` (gráficos).

**Nota:** el `.pkl` obligatorio corresponde al proyecto LOST. MATE no entrena un modelo; si se decide serializar una configuración elegida, debe tratarse como artefacto opcional/regenerable, no como requisito central de esta entrega.

**Entorno:** `poetry install --no-root` (instala dependencias sin empaquetar el proyecto).
