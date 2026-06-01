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
| Código `.ipynb` | `isolation.ipynb` (el notebook dado; se le agregan los experimentos + gráficos de MATE, corre de punta a punta) |
| Modelos computados (`.pkl` o formato similar) | **Sí, MATE entrega `.pkl`** (ver §10): `mate_best_config.pkl` — dict con la mejor técnica + profundidad + pesos hallados en los experimentos. Más el registro en `results.csv`. Se guarda en 2 líneas desde el notebook. |
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
- **No se modifica ningún archivo dado** (`board.py`, `agent.py`, etc.). Para 4×4 `board.clone()` funciona correcto (pisa el `grid` con la copia). La reproducibilidad se garantiza con `random.seed(seed)` en el runner, sin tocar código provisto.
- **`place_players()` no siembra `random`** → la reproducibilidad la fuerza el framework de experimentos.
- **Branching factor** ≈ (≤8 direcciones) × (celdas vacías destruibles) ≈ **~100/ply** en apertura → motiva fuertemente Alpha-Beta + ordenamiento de movimientos.

---

## 2. Desglose de tareas con estimación

Equipo: **2 personas** (A y B) dedicadas exclusivamente a MATE (LOST ya cerrado), plazo 1–2 semanas.

| #   | Tarea | Estimación | Resp. |
|-----|-------|-----------|-------|
| T0  | Setup: helper `play_match()` que devuelve resultado + stats, con seed y swap de quién arranca (**sin modificar archivos dados**) | 0.5 día | A |
| T1  | Núcleo de búsqueda (`search.py`): generación de sucesores, contador de nodos, utilidad terminal ±1/0 | 0.5 día | A |
| T2  | `MinimaxAgent` depth-limited (`V_max,min(s,d)`) + flag Alpha-Beta on/off + contador de nodos | 1 día | A |
| T3  | `ExpectimaxAgent` (nodos de azar, σ uniforme del rival) | 0.5 día | A |
| T4  | Heurísticas (`evaluation.py`): componentes h1–h4 + combinador ponderado | 1 día | B |
| T5  | Experimentos **en `isolation.ipynb`**: matchups E1–E6, seeds, registro a CSV | 1.5 días | A+B |
| T6  | Gráficos + tablas en el notebook | 0.5 día | B |
| T7  | Guardar mejor configuración (`pickle.dump` de un dict, ver §10) | 0.25 día | B |
| T8  | Redacción sección MATE del informe | 1 día | A+B |
|     | **Total** | **~5–6 días/persona** | |

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
- [ ] **`.pkl` de MATE entregado** (`mate_best_config.pkl`: dict técnica + profundidad + pesos) → *auditoría: modelos computados (.pkl o formatos similares)*
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
- **Estado:** ✅ completado
- **Archivo:** nuevo `Isolation/match.py` (**no se modifica `board.py` ni ningún archivo dado**).
- **Qué:** crear `play_match(agent_p1, agent_p2, seed=None, render=False) -> dict` que corra una partida (basado en `play.py`, que solo imprime) y **devuelva** `{winner, plies, avg_move_time, seed}`. Sembrar `random.seed(seed)` antes del `reset`.
- **Hecho cuando:** se pueden correr partidas reproducibles (misma seed → mismo resultado) con stats.
- **Verificar:** correr 2 veces con la misma seed `RandomAgent` vs `RandomAgent` → resultado idéntico.

### Paso 1 — Núcleo de búsqueda
- **Estado:** ✅ completado
- **Archivo:** `Isolation/search.py`.
- **Qué:** funciones puras: `successors(board, player)` (usando `get_possible_actions` + `clone`), `terminal_value(board, player)` (±1 / 0 según `is_end`), y un **contador de nodos** expandidos.
- **Hecho cuando:** dado un board, devuelve sucesores correctos y la utilidad terminal correcta.
- **Verificar:** test manual sobre un board casi terminal (rival con 1 movimiento) → utilidad esperada.

### Paso 2 — MinimaxAgent (profundidad fija)
- **Estado:** ✅ completado
- **Archivo:** `Isolation/minimax_agent.py` (clase `MinimaxAgent(Agent)`).
- **Qué:** implementar `V_max,min(s,d)` del teórico (lám. 13). `next_action` elige el `argmax` en la raíz. Parámetros: `depth`, `eval_fn`, `use_alpha_beta` (flag, en este paso = False). Acumular contador de nodos.
- **Hecho cuando:** vence consistentemente a `RandomAgent`.
- **Verificar:** `MinimaxAgent` vs `RandomAgent`, N=50 → win rate alto; sin excepciones de acción inválida.

### Paso 3 — Alpha-Beta sobre el mismo núcleo
- **Estado:** ✅ completado
- **Archivo:** `Isolation/minimax_agent.py` (mismo, branch por flag).
- **Qué:** agregar poda α-β cuando `use_alpha_beta=True`, idealmente con **ordenamiento de movimientos** (probar primero los de mayor `eval`).
- **Hecho cuando:** a igual profundidad y seed, AB-on devuelve **el mismo movimiento** que AB-off pero expande menos nodos.
- **Verificar:** test de equivalencia: para K boards aleatorios, `argmax(AB-on) == argmax(AB-off)` y `nodos(AB-on) ≤ nodos(AB-off)`.

### Paso 4 — Funciones de evaluación
- **Estado:** ✅ completado
- **Archivo:** `Isolation/evaluation.py`.
- **Qué:** `h1_mobility`, `h2_mobility_diff`, `h3_center`, `h4_surround` (todas `(board, player) -> float`) + `weighted_eval(weights)` que retorna una `eval_fn` combinada. Respetar el signo (perspectiva del agente).
- **Hecho cuando:** cada heurística devuelve valores coherentes y `weighted_eval` se puede inyectar a los agentes.
- **Verificar:** boards de juguete con valor esperado conocido (p. ej. agente en el centro → h3 mayor que en una esquina).

### Paso 5 — ExpectimaxAgent
- **Estado:** ✅ completado
- **Archivo:** `Isolation/expectimax_agent.py` (clase `ExpectimaxAgent(Agent)`).
- **Qué:** igual estructura que Minimax pero los nodos del **rival** son nodos de azar: `Σ σ(s,a)·V(Suc(s,a))` con σ uniforme sobre las acciones legales (lám. 8 del teórico).
- **Hecho cuando:** vence a `RandomAgent` y corre sin errores vs `Stratagem`.
- **Verificar:** `ExpectimaxAgent` vs `RandomAgent`, N=50 → win rate alto.

### Paso 6 — Experimentos en el notebook (`isolation.ipynb`)
- **Estado:** ✅ completado en **pasada rápida** (N chico) — pipeline E1–E6 + `results.csv` validados; **falta subir N** para números finales (ver `avancesMATE.md`).
- **Archivo:** `Isolation/isolation.ipynb` (el notebook dado; se le agregan celdas que importan los agentes y usan `match.py`).
- **Qué:** correr los matchups E1–E6 con `play_match` (N ≥ 100, alternando lados y sembrando seeds) y **registrar resultados a CSV** (p. ej. `results.csv`: matchup, agente, oponente, profundidad, win, plies, nodos, tiempo). El registro se arma con celdas + `pandas`/`csv`, sin módulo aparte.
- **Hecho cuando:** el notebook corre los 6 experimentos y deja el/los CSV reproducibles.
- **Verificar:** re-correr con las mismas seeds → mismos resultados.

### Paso 7 — Gráficos, tablas y `.pkl`
- **Estado:** ✅ completado — `plots/` (4 PNG: E1, E2/E3, E5, E6) + `mate_best_config.pkl` (recargable).
- **Archivo:** celdas del `isolation.ipynb` + un `.pkl`.
- **Qué:** los gráficos de la sección 4 (nodos vs profundidad, tiempo vs profundidad, win rate por matchup, win rate vs d, heatmap de torneo). Además, guardar la **mejor configuración** con `pickle.dump` (ver §10): un dict `{tecnica, profundidad, pesos, metricas}`.
- **Hecho cuando:** cada experimento tiene su visual claro y etiquetado, y el `.pkl` existe y se carga.
- **Verificar:** los gráficos cuentan la historia esperada (AB poda muchos nodos; Expectimax mejor vs Random, peor vs Stratagem); `pickle.load` devuelve el dict y permite reconstruir el agente ganador.

### Paso 8 — Cierre del notebook + informe
- **Estado:** ⏳ notebook ✅ (corre de punta a punta vía `nbconvert --execute`, equivale a "Restart & Run All"); **sección MATE del informe PDF pendiente** de redacción.
- **Archivo:** `Isolation/isolation.ipynb` + sección MATE del informe PDF.
- **Qué:** que el notebook quede ordenado (demos existentes + experimentos + gráficos + guardado del `.pkl`); redacción de la sección MATE apoyada en `DocumentacionMATE.md`.
- **Hecho cuando:** el notebook corre de punta a punta y el informe cubre el checklist (sección 7).
- **Verificar:** "Restart & Run All" sin errores; cross-check final contra la consigna.

---

## 9. Estructura de archivos propuesta

```
Isolation/
├── board.py              ← dado (sin cambios)
├── agent.py              ← interfaz dada (sin cambios)
├── random_agent.py       ← dado
├── stratagem.py          ← dado (oponente fuerte)
├── input_agent.py        ← dado
├── isolation_env.py      ← dado
├── play.py               ← dado (base de match.py)
├── isolation.ipynb       ← dado: se le AGREGAN los experimentos + gráficos + guardado .pkl
├── match.py              ← NUEVO: play_match() con seed + stats
├── search.py             ← NUEVO: sucesores, nodos, utilidad terminal
├── minimax_agent.py      ← NUEVO: MinimaxAgent + Alpha-Beta (flag)
├── expectimax_agent.py   ← NUEVO: ExpectimaxAgent
├── evaluation.py         ← NUEVO: heurísticas h1–h4 + weighted_eval
├── mate_best_config.pkl  ← NUEVO: dict con la mejor configuración (lo genera el notebook)
├── results.csv           ← NUEVO: registro de experimentos (lo genera el notebook)
└── pyproject.toml        ← dado
```

> **No se agregan archivos de más**: no hay `mate.ipynb`, `experiments.py` ni `persistence.py`. Los experimentos, el registro a CSV y el guardado del `.pkl` viven en el `isolation.ipynb` **ya existente**. Solo se suman 3 `.py` de agentes/heurísticas + 2 helpers (`match.py`, `search.py`), siguiendo la convención de la carpeta.

---

## 10. Entregable de modelo computado (`.pkl`) para MATE

Minimax/Expectimax **no entrenan** un modelo como Q-Learning, pero lo que MATE **sí computa** mediante la experimentación es la **mejor configuración de agente**. Eso es lo que serializamos, de la forma más simple posible (sin módulos ni clases extra):

```python
import pickle
best_config = {
    "tecnica": "minimax",       # o "expectimax"
    "profundidad": 3,
    "pesos": {"h1": 1.0, "h2": 2.0, "h3": 0.5, "h4": 1.0},
    "metricas": {"win_rate_vs_stratagem": 0.78, "...": "..."},
}
with open("mate_best_config.pkl", "wb") as f:
    pickle.dump(best_config, f)
```

**Por qué alcanza con esto:** la cláusula general de *Auditoría* pide *"modelos computados (.pkl o formatos similares)"*; la penalización estricta solo nombra el primer ejercicio (LOST). Entregar la mejor configuración hallada (un dict) cubre el requisito para MATE, hace **reproducible** el agente ganador (se reconstruye cargando el `.pkl`) y se hace en **dos líneas dentro del notebook** — sin `persistence.py`, sin tabla de transposición, sin `PolicyAgent`. Mantener esto simple fue una decisión deliberada para **no sobrecomplicar**.
