# Planificación — Proyecto MATE (Obligatorio IA, Marzo 2026)

> **Martian Adversarial Tactics Engine** — búsqueda adversarial (Minimax / Alpha-Beta / Expectimax) sobre el juego **Isolation**.
>
> Documento de planificación y **resolución paso a paso** para seguir al pie de la letra durante la implementación. La justificación detallada de cada decisión está en `DocumentacionMATE.md`.

---

## 1. Resumen y alcance

La consigna (`Obligatorio 2026 Marzo.md`, sección *Proyecto MATE*) pide concretamente:

1. **Técnicas:** implementar **Minimax** *y* **Expectimax** para decidir cuál es mejor en este caso. Minimax debe usar **Alpha-Beta Pruning** y **analizar su impacto**.
2. **Funciones de evaluación:** implementarlas y **experimentar** con distintas **combinaciones** y **ponderaciones**.
3. **Experimentación:** definir pruebas para evaluar los agentes y hacer un **registro completo** de resultados.

### Qué entra
- `MinimaxAgent` con poda Alpha-Beta (flag on/off para medir impacto), profundidad fija.
- `ExpectimaxAgent` (nodos de azar con modelo estocástico del rival).
- Biblioteca de funciones de evaluación + combinador ponderado.
- Framework de experimentos reproducible (seeds, swap de lados, logging CSV/JSON) + gráficos.

### Qué NO entra
- Iterative deepening / time limit: **fuera del core** (queda como mejora opcional de anexo). El teórico (`MiniMax.md`, lám. 13) trabaja con **profundidad fija**.

> **Sobre el `.pkl` de MATE — decisión deliberada (no asumir que no hace falta).** La consigna pide *"deben entregar... los modelos computados (.pkl o formatos similares)"* de forma **general**, y solo nombra explícitamente el *primer ejercicio* (LOST) en la cláusula de penalización. La lectura es **ambigua**. Como el costo de cubrirse es mínimo, **MATE entrega su propio `.pkl`** (ver §10 — Entregable de modelo). No se trata como opcional.

### Entregables y formato (sección *Auditoría* de la consigna)
Todo se entrega en un **único `.zip`** con:

| Entregable | Para MATE concretamente |
|------------|-------------------------|
| Código `.py` | `match.py`, `search.py`, `minimax_agent.py`, `expectimax_agent.py`, `evaluation.py`, `experiments.py` (ver §9) |
| Código `.ipynb` | `mate.ipynb` (demos + gráficos, corre de punta a punta) |
| Modelos computados (`.pkl` o formato similar) | **Sí, MATE entrega `.pkl`** (ver §10): `mate_best_config.pkl` (mejor técnica + profundidad + pesos hallados en E6) y, como modelo computado sustantivo, `mate_policy.pkl` (tabla de política/transposición precalculada). Más resultados crudos en `results/*.csv`/`*.json`. |
| Informe `.pdf` ≤ 20 págs. + anexos | Sección MATE del informe compartido con LOST (~8–9 págs.; tablas crudas → anexos) |
| Entorno | **Poetry separado** (ya existe `Isolation/pyproject.toml`) |

**Contenido exigido del informe** (la evaluación se basa en la documentación):
1. **Resumen del abordaje** de cada tarea — interacción con el simulador, **parámetros** usados (profundidad, pesos, N de partidas, seeds), **tiempo de ejecución** y **resultados**.
2. **Apoyo visual** — gráficos claros + comentarios que expliquen el desempeño.
3. **Notas de advertencia** — dificultades encontradas y por qué no se pudieron resolver (ver §6).

El informe debe ser **claro, legible y autocontenido**: suficiente para entender enfoque, resultados y conclusiones sin leer el código.

### Restricciones fijadas (de la consigna + el código dado, no se re-deciden)
- **Simulador dado y completo** en `Isolation/`: tablero **4×4**, 8 direcciones, acción `(direction, cell_to_destroy)`, jugadores 1 (`B`) / 2 (`R`), colocación inicial **aleatoria**, victoria = el rival se queda sin movimientos (`Board.is_end`).
- **Interfaz del agente** (`Isolation/agent.py`): hay que implementar `next_action(obs)` y `heuristic_utility(board)`. El estado es el `Board`; las acciones legales vienen de `board.get_possible_actions(player)`; `board.clone()` permite simular.
- **Oponentes ya provistos:** `RandomAgent` (estocástico) y `Stratagem` (Minimax d=3, deofuscado).
- **Entorno Poetry separado** ya existe (`Isolation/pyproject.toml`, Python ~3.10).
- **`board.clone()` clava 4×4** (reconstruye `Board()` con tamaño por defecto) → se mantiene 4×4; se parchea defensivamente.
- **`place_players()` no siembra `random`** → la reproducibilidad la fuerza el framework de experimentos.
- **Branching factor** ≈ (≤8 direcciones) × (celdas vacías destruibles) ≈ **~100/ply** en apertura → motiva fuertemente Alpha-Beta + ordenamiento de movimientos.

---

## 2. Desglose de tareas con estimación

Equipo: **2 personas** (A y B) dedicadas exclusivamente a MATE (LOST ya cerrado), plazo 1–2 semanas.

| #   | Tarea | Estimación | Resp. |
|-----|-------|-----------|-------|
| T0  | Setup: parchear `clone()` (preservar `board_size`), helper `play_match()` que devuelve resultado + stats, con seed y swap de quién arranca | 0.5 día | A |
| T1  | Núcleo de búsqueda (`search.py`): generación de sucesores, contador de nodos, utilidad terminal ±1/0 | 0.5 día | A |
| T2  | `MinimaxAgent` depth-limited (`V_max,min(s,d)`) + flag Alpha-Beta on/off + contador de nodos | 1 día | A |
| T3  | `ExpectimaxAgent` (nodos de azar, σ uniforme del rival) | 0.5 día | A |
| T4  | Biblioteca de heurísticas (`evaluation.py`): componentes h1–h4 + combinador ponderado | 1 día | B |
| T5  | Framework de experimentos (`experiments.py`): matchups, seeds, logging CSV/JSON | 1 día | B |
| T6  | Experimento Alpha-Beta (nodos/tiempo con vs sin poda, barrido de profundidad) | 0.5 día | A |
| T7  | Experimento Minimax vs Expectimax vs cada oponente | 0.5 día | B |
| T8  | Torneo de heurísticas (round-robin de ponderaciones) | 1 día | B |
| T9  | Gráficos + tablas + `persistence.py` (serializar `mate_best_config.pkl` y `mate_policy.pkl`, ver §10) | 1 día | B |
| T10 | Notebook entregable `mate.ipynb` + redacción sección MATE del informe | 1.5 días | A+B |
|     | **Total** | **~7 días/persona** | |

---

## 3. Orden de resolución recomendado

1. **Minimax SIN poda primero** → es la **referencia de corrección**: Alpha-Beta debe devolver el *mismo* movimiento, solo más rápido.
2. **Alpha-Beta** como flag sobre el mismo código → permite medir impacto contra (1) a igual profundidad.
3. **Heurísticas (T4)** en paralelo (persona B) mientras A hace los algoritmos: son independientes porque la heurística se **inyecta** al agente.
4. **Expectimax** reutilizando el núcleo de búsqueda.
5. **Experimentos** al final, con agentes y heurísticas ya estables: impacto AB → Minimax-vs-Expectimax → torneo de heurísticas.

---

## 4. Diseño del set de experimentos

**Reproducibilidad (clave):** `place_players()` usa `random.shuffle` **sin seed**. Por cada partida se hace `random.seed(k)`. Como el arranque es aleatorio y existe **ventaja de primer jugador**, se **alterna quién arranca** y se corren **N ≥ 100** partidas por matchup.

| Exp. | Matchups | Métricas | Gráfico |
|------|----------|----------|---------|
| **E1 — Impacto Alpha-Beta** | Minimax AB-on vs AB-off, mismo rival, d=1..4 | nodos expandidos, tiempo/jugada | nodos vs profundidad (log) + tiempo vs profundidad |
| **E2 — Sanity vs Random** | Minimax→Random, Expectimax→Random | win rate, largo de partida | barras |
| **E3 — vs Stratagem** | Minimax→Stratagem, Expectimax→Stratagem | win rate, tiempo/jugada | barras agrupadas |
| **E4 — Minimax vs Expectimax** | self-play directo | win rate, quién arranca | barras |
| **E5 — Profundidad** | mejor agente, d=1..4 vs Stratagem | win rate vs d | línea |
| **E6 — Torneo de heurísticas** | round-robin de ~4–5 ponderaciones | win% par a par | heatmap / tabla |

**Hallazgo a *demostrar* (medir, no asumir):** Expectimax modela al rival como estocástico uniforme → se espera que **gane más vs Random** pero **pierda vs Stratagem** (que es Minimax determinista). Esa es la respuesta a *"¿cuál técnica es mejor para este caso?"*: **depende del oponente**, con evidencia.

---

## 5. Heurísticas a experimentar

Todas evaluadas desde la **perspectiva del agente** (positivo = bueno para el agente), vía `heuristic_utility(board)`:

| ID | Heurística | Definición | Origen |
|----|-----------|-----------|--------|
| h1 | Movilidad propia | nº de casillas adyacentes libres del agente | clásica Isolation |
| h2 | Diff de movilidad | `mov_propia − mov_rival` | `W` de Stratagem |
| h3 | Control de centro | `−dist_Manhattan(agente, centro)` | `Y` de Stratagem |
| h4 | Acorralar | celdas destruidas alrededor del rival − alrededor mío | `X` de Stratagem |

**Combinación ponderada:** `Eval(s) = Σ wᵢ · hᵢ(s)`. En E6 se barren pesos `(w1, w2, w3, w4)` para encontrar la mejor combinación.

> **Nota:** para movilidad se usa el conteo de **casillas adyacentes libres** (estilo `has_valid_moves`), no `len(get_possible_actions)`, que infla la cuenta al multiplicar por cada celda destruible.

---

## 6. Riesgos y mitigación

| Riesgo | Mitigación |
|--------|-----------|
| **Explosión combinatoria** (b ≈ 100/ply) | Alpha-Beta + ordenamiento de movimientos + profundidad fija 3–4 |
| **`clone()` clava 4×4** | Parchear `clone()` para preservar `board_size`; mantener 4×4 |
| **No-determinismo del arranque** | `random.seed` por partida + alternar lados + N ≥ 100 |
| **Comparación injusta de heurísticas** | misma profundidad, mismas seeds, swap de lados, mismo N |
| **`get_possible_actions` clona el board por acción → caro en profundidad** | sucesores livianos / limitar d; paralelizar partidas (multiprocessing) entre las 2 personas |
| **Expectimax "peor" vs Stratagem** | NO es bug: es la conclusión teórica esperada; documentarla |
| **Informe ≤ 20 págs. compartido con LOST** | MATE ~8–9 págs.; tablas crudas → anexos |

---

## 7. Checklist "MATE está completa"

Mapeado a cada punto de la consigna:

- [ ] Minimax depth-limited implementado (`V_max,min(s,d)`) → *consigna 1*
- [ ] Alpha-Beta implementado **y** análisis de impacto (nodos/tiempo con vs sin poda) → *consigna 1*
- [ ] Expectimax implementado (nodos de azar) → *consigna 1*
- [ ] Decisión justificada **Minimax vs Expectimax** con evidencia → *consigna 1*
- [ ] ≥ 2 funciones de evaluación + combinaciones ponderadas experimentadas → *consigna 2*
- [ ] Pruebas definidas + **registro completo** (CSV/JSON + seeds) → *consigna 3*
- [ ] **Modelos `.pkl` de MATE entregados** (`mate_best_config.pkl` + `mate_policy.pkl`) → *auditoría: modelos computados (.pkl o formatos similares)*
- [ ] Informe: resumen del abordaje (interacción con simulador, **parámetros**, **tiempos de ejecución**, resultados) → *contenido del informe*
- [ ] Apoyo visual (gráficos claros + comentarios) → *contenido del informe*
- [ ] Notas de advertencia (dificultades y por qué no se resolvieron) → *contenido del informe*
- [ ] Informe **claro, legible y autocontenido**, ≤ 20 págs. + anexos → *criterios de evaluación*
- [ ] Código `.py` + `.ipynb`, entorno Poetry separado, todo en un único `.zip` → *auditoría / ambiente*

---

## 8. Resolución paso a paso

Secuencia accionable. Cada paso indica **archivo**, **qué implementar**, **criterio de hecho** y **cómo verificar**. Marcá el estado de cada paso a medida que avanzás (y registralo en `avancesMATE.md`).

> **Leyenda de estado:** ⬜ pendiente · ⏳ en curso · ✅ completado

### Paso 0 — Setup y utilidades base
- **Estado:** ⬜ pendiente
- **Archivo:** `Isolation/board.py` (parche puntual) + nuevo `Isolation/match.py`.
- **Qué:**
  - Parchear `Board.clone()` para que preserve `board_size`: `Board(self.board_size)` en vez de `Board()`.
  - Crear `play_match(env, agent1, agent2, seed=None, swap=False) -> dict` que corra una partida (basado en `play.py`) y devuelva `{winner, plies, time_per_move, ...}`. Sembrar `random.seed(seed)` antes del `reset`.
- **Hecho cuando:** se pueden correr partidas reproducibles (misma seed → mismo resultado) con stats.
- **Verificar:** correr 2 veces con la misma seed `RandomAgent` vs `RandomAgent` → resultado idéntico.

### Paso 1 — Núcleo de búsqueda
- **Estado:** ⬜ pendiente
- **Archivo:** `Isolation/search.py`.
- **Qué:** funciones puras: `successors(board, player)` (usando `get_possible_actions` + `clone`), `terminal_value(board, player)` (±1 / 0 según `is_end`), y un **contador de nodos** expandidos.
- **Hecho cuando:** dado un board, devuelve sucesores correctos y la utilidad terminal correcta.
- **Verificar:** test manual sobre un board casi terminal (rival con 1 movimiento) → utilidad esperada.

### Paso 2 — MinimaxAgent (profundidad fija)
- **Estado:** ⬜ pendiente
- **Archivo:** `Isolation/minimax_agent.py` (clase `MinimaxAgent(Agent)`).
- **Qué:** implementar `V_max,min(s,d)` del teórico (lám. 13). `next_action` elige el `argmax` en la raíz. Parámetros: `depth`, `eval_fn`, `use_alpha_beta` (flag, en este paso = False). Acumular contador de nodos.
- **Hecho cuando:** vence consistentemente a `RandomAgent`.
- **Verificar:** `MinimaxAgent` vs `RandomAgent`, N=50 → win rate alto; sin excepciones de acción inválida.

### Paso 3 — Alpha-Beta sobre el mismo núcleo
- **Estado:** ⬜ pendiente
- **Archivo:** `Isolation/minimax_agent.py` (mismo, branch por flag).
- **Qué:** agregar poda α-β cuando `use_alpha_beta=True`, idealmente con **ordenamiento de movimientos** (probar primero los de mayor `eval`).
- **Hecho cuando:** a igual profundidad y seed, AB-on devuelve **el mismo movimiento** que AB-off pero expande menos nodos.
- **Verificar:** test de equivalencia: para K boards aleatorios, `argmax(AB-on) == argmax(AB-off)` y `nodos(AB-on) ≤ nodos(AB-off)`.

### Paso 4 — Funciones de evaluación
- **Estado:** ⬜ pendiente
- **Archivo:** `Isolation/evaluation.py`.
- **Qué:** `h1_mobility`, `h2_mobility_diff`, `h3_center`, `h4_surround` (todas `(board, player) -> float`) + `weighted_eval(weights)` que retorna una `eval_fn` combinada. Respetar el signo (perspectiva del agente).
- **Hecho cuando:** cada heurística devuelve valores coherentes y `weighted_eval` se puede inyectar a los agentes.
- **Verificar:** boards de juguete con valor esperado conocido (p. ej. agente en el centro → h3 mayor que en una esquina).

### Paso 5 — ExpectimaxAgent
- **Estado:** ⬜ pendiente
- **Archivo:** `Isolation/expectimax_agent.py` (clase `ExpectimaxAgent(Agent)`).
- **Qué:** igual estructura que Minimax pero los nodos del **rival** son nodos de azar: `Σ σ(s,a)·V(Suc(s,a))` con σ uniforme sobre las acciones legales (lám. 8 del teórico).
- **Hecho cuando:** vence a `RandomAgent` y corre sin errores vs `Stratagem`.
- **Verificar:** `ExpectimaxAgent` vs `RandomAgent`, N=50 → win rate alto.

### Paso 6 — Framework de experimentos
- **Estado:** ⬜ pendiente
- **Archivo:** `Isolation/experiments.py`.
- **Qué:** orquestar matchups E1–E6, alternar lados, sembrar seeds, **registrar resultados a CSV/JSON** en `Isolation/results/`. Columnas: matchup, agente, oponente, profundidad, win, plies, nodos, tiempo.
- **Hecho cuando:** corre todos los matchups y deja archivos de resultados reproducibles.
- **Verificar:** re-correr con las mismas seeds → mismos CSV.

### Paso 7 — Correr experimentos E1–E6
- **Estado:** ⬜ pendiente
- **Qué:** ejecutar el set completo con N ≥ 100. Opcional: paralelizar con `multiprocessing` repartiendo entre las 2 personas.
- **Hecho cuando:** `results/` contiene los datos de las 6 pruebas.
- **Verificar:** revisar tamaños de muestra y ausencia de partidas con error.

### Paso 8 — Gráficos, tablas y modelos `.pkl`
- **Estado:** ⬜ pendiente
- **Archivo:** celdas del notebook + `results/plots/` + `persistence.py` + `models/mate_best_config.pkl` + `models/mate_policy.pkl`.
- **Qué:** los gráficos de la sección 4 (nodos vs profundidad, tiempo vs profundidad, win rate por matchup, win rate vs d, heatmap de torneo). Además, **serializar los dos modelos computados** (ver §10): la mejor configuración (E6) y la tabla de política/transposición (o libro de aperturas si la completa es inviable).
- **Hecho cuando:** cada experimento tiene su visual claro y etiquetado, y ambos `.pkl` existen y se cargan.
- **Verificar:** los gráficos cuentan la historia esperada (AB poda muchos nodos; Expectimax mejor vs Random, peor vs Stratagem); recargar `mate_best_config` reconstruye el agente ganador y `PolicyAgent` cargando `mate_policy.pkl` juega sin recalcular.

### Paso 9 — Notebook entregable + informe
- **Estado:** ⬜ pendiente
- **Archivo:** `Isolation/mate.ipynb` + sección MATE del informe PDF.
- **Qué:** notebook que importa los agentes, corre demos cortas y muestra los gráficos; redacción apoyada en `DocumentacionMATE.md`.
- **Hecho cuando:** el notebook corre de punta a punta y el informe cubre el checklist (sección 7).
- **Verificar:** "Restart & Run All" sin errores; cross-check final contra la consigna.

---

## 9. Estructura de archivos propuesta

```
Isolation/
├── board.py              ← (parche menor en clone())
├── agent.py              ← interfaz dada (sin cambios)
├── random_agent.py       ← dado
├── stratagem.py          ← dado (oponente fuerte)
├── input_agent.py        ← dado
├── isolation_env.py      ← dado
├── play.py               ← dado (base de match.py)
├── match.py              ← NUEVO: play_match() con seed/swap/stats
├── search.py             ← NUEVO: sucesores, nodos, utilidad terminal
├── minimax_agent.py      ← NUEVO: MinimaxAgent + Alpha-Beta (flag)
├── expectimax_agent.py   ← NUEVO: ExpectimaxAgent
├── evaluation.py         ← NUEVO: heurísticas h1–h4 + weighted_eval
├── experiments.py        ← NUEVO: matchups + logging
├── mate.ipynb            ← NUEVO: notebook entregable
├── persistence.py        ← NUEVO: save/load de config y policy (.pkl)
├── models/               ← NUEVO: mate_best_config.pkl, mate_policy.pkl
├── results/              ← NUEVO: CSV/JSON + plots/
└── pyproject.toml        ← dado
```

---

## 10. Entregable de modelo computado (`.pkl`) para MATE

Aunque Minimax/Expectimax **no entrenan** un modelo como Q-Learning, MATE **sí computa** artefactos serializables. Entregamos **dos** `.pkl` (en `Isolation/models/`) para no quedar expuestos ante la cláusula general de *"modelos computados (.pkl o formatos similares)"*:

| Artefacto | Contenido | Cómo se computa |
|-----------|-----------|-----------------|
| `mate_best_config.pkl` | dict con la **mejor configuración**: técnica ganadora (Minimax/Expectimax), profundidad, pesos `(w1..w4)` y métricas asociadas | Resultado de E6 (torneo de heurísticas) + E4 |
| `mate_policy.pkl` | **Tabla de política / transposición**: mapa `estado_canónico → mejor_acción` (y/o valor) para los estados alcanzables, a modo de *libro de aperturas* del 4×4 | Recorrido Minimax (con AB) cacheando la decisión por estado |

**Por qué `mate_policy.pkl` es un "modelo computado" legítimo:** en un tablero 4×4 el espacio de estados alcanzables es acotado; precalcular y serializar la decisión óptima por estado es exactamente *computar un modelo de decisión* offline que luego el agente **consume sin volver a buscar** (lookup O(1)). Es el análogo adversarial de una Q-table.

**Diseño (`persistence.py`):**
- `save_config(path, config: dict)` / `load_config(path) -> dict`.
- `build_policy_table(eval_fn, depth) -> dict` recorre los estados alcanzables y guarda `clave_estado → acción`. La clave es una **canonicalización** del `grid` (p. ej. `bytes(grid)` o tupla) para usarla de índice.
- `save_policy(path, table)` / `load_policy(path)`.
- Un `PolicyAgent(Agent)` opcional que en `next_action` hace lookup en la tabla y, si el estado no está, cae a Minimax en vivo.

**Tope de alcance:** si la tabla completa resultara demasiado grande/lenta de generar, se limita a un **libro de aperturas** (primeros k plies) — sigue siendo un modelo computado válido. Decidir con datos de tamaño/tiempo durante el Paso 8.
