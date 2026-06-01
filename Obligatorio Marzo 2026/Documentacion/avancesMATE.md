# Avances — Proyecto MATE

> Bitácora de progreso de MATE. Se actualiza **cada vez que se avanza** un paso de la
> resolución de `PlanificacionMATE.md` §8, para que cualquiera del equipo vea el estado
> y sepa **dónde retomar**.
>
> **Leyenda:** ⬜ pendiente · ⏳ en curso · ✅ completado
>
> **Cómo usar este archivo:**
> 1. Al empezar un paso, cambialo a ⏳ en la tabla y agregá una entrada en la bitácora.
> 2. Al terminarlo, marcalo ✅ acá **y** en `PlanificacionMATE.md` §8, y completá la entrada (qué se hizo, archivos, cómo se verificó).
> 3. Mantené actualizado **"Dónde retomar"** al final.

---

## Estado general

| Paso | Descripción | Archivo(s) | Estado | Responsable | Fecha |
|------|-------------|------------|--------|-------------|-------|
| 0 | Setup: `play_match()` con seed/stats (sin tocar archivos dados) | `match.py` | ✅ | Martín | 2026-06-01 |
| 1 | Núcleo de búsqueda (sucesores, nodos, utilidad terminal) | `search.py` | ✅ | Martín | 2026-06-01 |
| 2 | MinimaxAgent profundidad fija (sin poda) | `minimax_agent.py` | ⬜ | — | — |
| 3 | Alpha-Beta + ordenamiento de movimientos | `minimax_agent.py` | ⬜ | — | — |
| 4 | Funciones de evaluación h1–h4 + `weighted_eval` | `evaluation.py` | ⬜ | — | — |
| 5 | ExpectimaxAgent (nodos de azar) | `expectimax_agent.py` | ⬜ | — | — |
| 6 | Experimentos E1–E6 + registro a CSV (en el notebook) | `isolation.ipynb`, `results.csv` | ⬜ | — | — |
| 7 | Gráficos + tablas + guardar `mate_best_config.pkl` | `isolation.ipynb` | ⬜ | — | — |
| 8 | Cierre del notebook + sección MATE del informe | `isolation.ipynb`, informe PDF | ⬜ | — | — |

**Progreso:** 2 / 9 pasos completados.

---

## Bitácora de avances

> Una entrada por sesión de trabajo / por paso. La más reciente arriba.

### Plantilla (copiar para cada avance)

```
### [AAAA-MM-DD] Paso N — <título> — <responsable>
- **Estado:** ⏳ en curso | ✅ completado
- **Qué se hizo:** ...
- **Archivos tocados:** ...
- **Decisiones / desvíos respecto del plan:** ... (si los hubo)
- **Cómo se verificó:** ... (comando / test / resultado observado)
- **Pendiente / próximos pasos:** ...
```

<!-- Agregar las entradas reales debajo de esta línea -->

### [2026-06-01] Paso 1 — Núcleo de búsqueda — Martín
- **Estado:** ✅ completado
- **Qué se hizo:** se creó `Isolation/search.py` con `successors(board, player)`, `is_terminal(board, player_to_move)`, `utility(winner, agent_player)` (±1, sin empates), `other(player)` y la clase `NodeCounter` (para medir poda en Alpha-Beta).
- **Archivos tocados:** `Isolation/search.py` (nuevo).
- **Decisiones / desvíos respecto del plan:** ninguno.
- **Cómo se verificó:** `poetry run python search.py`. (1) `successors` devuelve **91** sucesores en apertura (confirma branching factor ~100/ply) y no muta el tablero original; (2) tablero con jugador 1 acorralado en esquina → `is_terminal` da `(True, 2)` y `utility` da −1/+1 según perspectiva; (3) `other` OK. Salió `OK Paso 1`.
- **Pendiente / próximos pasos:** Paso 2 — `MinimaxAgent` (profundidad fija, sin poda) apoyándose en `search.py`.

### [2026-06-01] Paso 0 — Setup y utilidades base — Martín
- **Estado:** ✅ completado
- **Qué se hizo:** se creó `Isolation/match.py` con `play_match(agent_p1, agent_p2, seed, render)`, que corre una partida y **devuelve** `{winner, plies, avg_move_time, seed}` (a diferencia de `play.py`, que solo imprime). Siembra `random.seed(seed)` para reproducibilidad.
- **Archivos tocados:** `Isolation/match.py` (nuevo). **No** se modificó ningún archivo dado.
- **Decisiones / desvíos respecto del plan:** se descartó el parche a `Board.clone()`: en 4×4 funciona bien y, con `seed`, la ejecución es determinista igual. (Plan §1, T0 y Paso 0 actualizados acorde.)
- **Cómo se verificó:** `poetry install` + `poetry run python match.py`. Dos corridas `RandomAgent(1)` vs `RandomAgent(2)` con `seed=42` dieron resultado idéntico (`winner=2, plies=4`). Salió `OK Paso 0`.
- **Pendiente / próximos pasos:** Paso 1 — `search.py` (`successors`, `is_terminal`, `utility`, `NodeCounter`).

---

## Dónde retomar

- **Próximo paso a ejecutar:** **Paso 2 — `MinimaxAgent`** (profundidad fija, sin poda todavía), usando `search.py`.
- **Listo para usar:** `match.py` → `play_match(agent_p1, agent_p2, seed=...)` devuelve stats. `search.py` → `successors`, `is_terminal`, `utility`, `other`, `NodeCounter`. Entorno Poetry de `Isolation/` ya instalado.
- **Bloqueos actuales:** ninguno.
- **Notas para el equipo:** seguir el orden de `PlanificacionMATE.md` §3 (Minimax sin poda → Alpha-Beta → heurísticas en paralelo → Expectimax → experimentos). Recordar sembrar `random` por partida y alternar lados (ver §4 del plan). Correr scripts con `poetry run python <archivo>` desde `Isolation/`.
