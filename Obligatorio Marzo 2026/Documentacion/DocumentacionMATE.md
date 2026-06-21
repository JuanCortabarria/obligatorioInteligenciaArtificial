# Documentación — Proyecto MATE (Obligatorio IA, Marzo 2026)

> **Martian Adversarial Tactics Engine** — explicación detallada y **justificación** de las decisiones de diseño de la solución de búsqueda adversarial sobre **Isolation**.
>
> Este documento es la base del informe PDF (sección MATE). La planificación operativa y la resolución paso a paso están en `PlanificacionMATE.md`.

---

## 1. Descripción del problema y del simulador

**Isolation** es un juego de tablero adversarial de **suma cero** y **dos jugadores alternados**, exactamente el escenario que modela el teórico `MiniMax.md` (lám. 2–4). Cada jugador, en su turno, **mueve su ficha** a una casilla adyacente libre **y destruye** una casilla del tablero. Pierde quien se queda **sin movimientos legales**.

El simulador viene **dado y completo** en `Isolation/`. Los elementos clave del modelo (siguiendo la notación del teórico, lám. 4):

| Concepto del teórico | Implementación dada |
|----------------------|---------------------|
| Estado inicial `s_start` | `Board(board_size=(4,4))` con dos fichas colocadas al azar (`place_players`) |
| `Acciones(s)` | `board.get_possible_actions(player)` → lista de `(direction, cell_to_destroy)` |
| `Suc(s,a)` | `board.clone()` + `board.play(action, player)` |
| `EsFinal(s)` | `board.is_end(player) -> (bool, ganador)` |
| `Utilidad(s)` | derivada del ganador: +1 / −1 / 0 |
| `Jugador(s)` | el `current_player` del `IsolationEnv` (alterna 1 ↔ 2) |

**Representación del tablero** (`board.py`): `grid` es una matriz NumPy `4×4` con `0` = vacía, `1` = jugador 1 (`B`), `2` = jugador 2 (`R`), `3` = celda destruida (`X`). Hay **8 direcciones** de movimiento (ortogonales + diagonales).

**Interfaz del agente** (`agent.py`): toda implementación debe definir
- `next_action(obs)` — recibe el `Board` y devuelve una acción `(direction, cell_to_destroy)`;
- `heuristic_utility(board)` — la función de evaluación de un estado no terminal.

**Oponentes provistos:**
- `RandomAgent` — elige una acción legal al azar (**estocástico**).
- `Stratagem` — agente ofuscado que, deofuscado, resulta ser un **Minimax de profundidad 3** cuya heurística suma cuatro componentes: diferencia de celdas destruidas alrededor de cada ficha, distancia (negativa) al centro, distancia (negativa) al rival y **diferencia de movilidad**. Es el baseline "fuerte" a vencer y nos sirve de referencia para diseñar heurísticas.

---

## 2. Marco teórico aplicado

Toda la solución se basa en `MiniMax.md` (Sergio Yovine, ORT) y Russell & Norvig, *AIMA* 3ª ed., cap. 5.

- **Minimax** (lám. 10): el agente maximiza y el oponente minimiza el valor. `V_minimax(s) = max_a V(Suc(s,a))` en nodos del agente y `min_a V(Suc(s,a))` en nodos del oponente. Garantiza la **mejor estrategia frente a un rival que juega óptimo** (lám. 12: el valor minimax es una **cota inferior** del valor real ante cualquier oponente).
- **Expectimax** (lám. 8): cuando el oponente juega una **estrategia estocástica**, sus nodos dejan de ser `min` y pasan a ser **nodos de azar**: `Σ_a σ_oponente(s,a) · V(Suc(s,a))`. El agente sigue maximizando.
- **Funciones de evaluación con profundidad limitada** (lám. 13): como el árbol completo es caro, se limita la profundidad a `d` y en el corte se usa `Eval(s)` en lugar de la utilidad real:

  ```
  V_max,min(s,d) = Utilidad(s)                         si EsFinal(s)
                 = Eval(s)                             si d = 0
                 = max_a V_max,min(Suc(s,a), d−1)      si Jugador(s) = Agente
                 = min_a V_max,min(Suc(s,a), d−1)      si Jugador(s) = Oponente
  ```

- **Alpha-Beta Pruning** (AIMA 5.3): optimización de Minimax que **poda ramas** que no pueden afectar la decisión, devolviendo **exactamente el mismo movimiento** que Minimax pero expandiendo menos nodos.
- **Buena función de evaluación** (lám. 16): debe (1) ordenar los estados terminales igual que la utilidad real (`Eval(win) > Eval(draw) > Eval(loss)`), (2) ser **barata** de calcular, y (3) correlacionar fuertemente con la probabilidad real de ganar.

---

## 3. Decisiones de diseño y su justificación

### 3.1 Profundidad fija (no iterative deepening)
**Decisión:** búsqueda con **profundidad fija** `d`, según `V_max,min(s,d)`.
**Justificación:** es exactamente el modelo del teórico (lám. 13). El tablero es **4×4** (acotado), por lo que una profundidad moderada (3–4) ya da buen juego sin necesidad de un esquema de tiempo. Iterative deepening + time limit agregarían código y riesgo sin estar pedidos por la consigna; quedan como mejora opcional de anexo. Se priorizó **fidelidad al material de estudio**.

### 3.2 Alpha-Beta y cómo se mide su impacto
**Decisión:** implementar Alpha-Beta como un **flag** sobre el mismo Minimax, e idealmente con **ordenamiento de movimientos**.
**Justificación:** en Isolation cada acción combina **dirección × celda a destruir**, así que el *branching factor* en apertura ronda **~100 acciones por ply** (≤8 direcciones × ~13 celdas destruibles). A profundidad 3 eso son ~10⁶ nodos. Alpha-Beta es la palanca natural para domar ese costo. Tener el flag on/off permite el **"análisis de impacto"** que pide la consigna de forma rigurosa: a **igual profundidad y misma seed**, se compara
- **nodos expandidos** con vs sin poda, y
- **tiempo por jugada**,

verificando además que el **valor minimax es idéntico** (la poda no cambia la calidad de la decisión). El ordenamiento de movimientos se justifica porque la poda es máxima cuando se exploran primero las jugadas más prometedoras. *Matiz importante:* con ordenamiento activo, ante **empates de valor** Alpha-Beta puede elegir **otra jugada igualmente óptima**, de modo que partidas sueltas pueden divergir (lo medimos en E7: ~9 %); el **valor** —y por tanto la habilidad— es el mismo, y el aporte medible de la poda es el **costo** (nodos/tiempo).

### 3.3 Minimax vs Expectimax: cuándo conviene cada uno
**Decisión:** implementar ambos y **dejar que los datos decidan**, sin asumir un ganador a priori.
**Justificación (teórica):** Minimax asume un rival **óptimo/adversarial**; su valor es una cota inferior garantizada (lám. 12). Expectimax asume un rival **estocástico** y modela sus acciones con `σ` uniforme (lám. 8). De ahí la **hipótesis** a confirmar experimentalmente:
- **vs `RandomAgent`** (que es genuinamente estocástico): Expectimax debería **rendir mejor**, porque su modelo del rival es correcto y puede explotar jugadas riesgosas con alto valor esperado.
- **vs `Stratagem`** (que es Minimax determinista, *no* uniforme): el modelo uniforme de Expectimax es **incorrecto**, por lo que se espera que **Minimax rinda igual o mejor**.

La respuesta a *"¿cuál técnica es mejor para este caso?"* es por tanto **"depende del oponente"**, y se sustenta con E2–E4. Importante: si Expectimax pierde contra Stratagem, **no es un bug**, es la consecuencia esperada de un modelo de oponente equivocado.

> **Confirmación empírica (ver §5).** Los experimentos a profundidad igualada con Stratagem (**d=3**) confirman la hipótesis: **Minimax (46%) supera a Expectimax (34%)** vs Stratagem, y además a un costo mucho menor (Expectimax a d=3 es ~4.5× más lento por no poder podar). Profundizar la búsqueda **empeora** a Expectimax frente a un rival determinista (48% a d=2 → 34% a d=3) y **mejora** a Minimax (39% → 46%), justo lo que predice la teoría del modelo de oponente. A profundidad baja (d=2) el orden se invierte, lo que muestra que la ventaja de Minimax aquí **requiere profundidad suficiente**. **Matiz (E7, §5.7):** este resultado vale para la ponderación *balanceada*; con la mejor heurística `h2` (E7) Minimax y Expectimax quedan **parejos** vs Stratagem. Por eso la ventaja **robusta** de Minimax no es el win rate sino el **costo** (poda; Expectimax no poda).

### 3.4 Elección de las funciones de evaluación
**Decisión:** una biblioteca de componentes combinables por pesos: movilidad propia (h1), diferencia de movilidad (h2), control de centro (h3) y acorralar al rival (h4), con `Eval(s) = Σ wᵢ·hᵢ(s)`.
**Justificación:**
- En Isolation **perder = quedarse sin movimientos**, por lo que la **movilidad** (casillas adyacentes libres) es la señal más directamente correlacionada con ganar — cumple el criterio (3) de la lám. 16. La **diferencia de movilidad** (h2) captura la naturaleza de suma cero.
- **Control de centro** (h3): desde el centro hay más casillas alcanzables, lo que tiende a preservar movilidad futura.
- **Acorralar** (h4): destruir casillas alrededor del rival reduce su movilidad — ataque directo a su condición de derrota.
- Estos cuatro componentes **replican y generalizan** la heurística de `Stratagem`, lo que nos da un punto de comparación honesto y nos permite experimentar con **ponderaciones** (consigna 2) buscando combinaciones que la superen.

**Medida de movilidad:** se cuenta el número de **casillas adyacentes libres** (estilo `Board.has_valid_moves`) y **no** `len(get_possible_actions)`, porque esta última multiplica por cada celda destruible e infla artificialmente el valor sin reflejar mejor la posición.

**Escalado de los componentes (por qué no hace falta normalizar):** en 4×4 las cuatro componentes ya viven en el **mismo orden de magnitud** — h1 ∈ [0, 8], h2 ∈ [−8, 8], h3 ∈ [−4, 0], h4 ∈ [−8, 8] —, así que ninguna "se come" a las otras por escala y alcanza con los **pesos** `wᵢ` para fijar la importancia relativa. (Si las magnitudes fueran dispares —una en cientos y otra en unidades— habría que escalar antes de combinar, como advierte la lám. 16.)

**Pesos base de los experimentos principales (y por qué no sesgan las conclusiones).** Los matchups E1–E5 usan una ponderación base `{h1:1, h2:2, h3:0.5, h4:1}`: una combinación *neutra* que incluye las cuatro componentes con énfasis en la **diferencia de movilidad** (la más informativa) e inspirada en la heurística de `Stratagem`. La elegimos como punto de partida razonable **antes** de conocer el resultado del torneo de heurísticas. Es clave notar que **la comparación de técnicas (Minimax vs Expectimax) y el análisis de Alpha-Beta son robustos a esta elección**: ambos agentes comparten la *misma* `eval_fn`, de modo que cambiar los pesos no altera *qué* técnica gana ni cuánto poda Alpha-Beta (sí puede mover los valores absolutos de win rate, no el orden entre técnicas). El experimento **E6** (§5.5) explora **por separado** cuál ponderación es la mejor —resultó `solo_mov_diff` (solo h2)—, que queda como configuración recomendada. Es decir: los pesos base sirven para una comparación *justa entre técnicas/poda*, y E6 responde la pregunta *aparte* de cuál es la mejor heurística.

### 3.5 Convención de signo y utilidad terminal
**Decisión:** `heuristic_utility` y la utilidad terminal se expresan **desde la perspectiva del agente**: positivo = bueno para el agente. Utilidad terminal = **+1** (gana el agente), **−1** (pierde), **0** (empate/no decidido).
**Justificación:** mantener un único marco de referencia evita errores de signo en los nodos `min` y de azar, y es coherente con `Stratagem`, que calcula su heurística relativa a `self.player`. Cumple el criterio (1) de la lám. 16 (`Eval(win) > Eval(draw) > Eval(loss)`).

### 3.6 Reproducibilidad de los experimentos (diseño apareado)
**Decisión:** sembrar `random` por partida y usar un **diseño apareado**: cada *seed* se juega **dos veces**, una con nuestro agente de jugador 1 y otra de jugador 2, sobre la **misma colocación inicial**. Se corren decenas de seeds por matchup y se promedia.
**Justificación:** `Board.place_players()` usa `random.shuffle` **sin semilla**, así que sin intervención las partidas no son reproducibles; sembrar por partida las hace deterministas. Además, la colocación inicial aleatoria introduce **varianza** y existe una **ventaja de primer jugador** fuerte (medida en §6). El diseño apareado neutraliza ambos sesgos de forma **controlada**: como cada seed enfrenta la *misma* posición desde los dos lados, la comparación entre técnicas/heurísticas no depende de qué posiciones tocaron ni de quién arrancó. Es más riguroso que solo alternar lados con seeds distintos (que era el enfoque inicial: el agente de jugador 1 veía posiciones distintas que el de jugador 2).

### 3.7 No se modifican los archivos dados; tablero 4×4
**Decisión:** **no modificar ningún archivo provisto** (`board.py`, `agent.py`, `isolation_env.py`, etc.) y trabajar sobre **4×4**.
**Justificación:** el simulador viene dado y funciona; toda la solución se construye **alrededor** de él implementando archivos nuevos que consumen su API pública (`get_possible_actions`, `clone`, `play`, `is_end`). En 4×4 `board.clone()` opera correctamente (reconstruye un tablero 4×4 y le copia el `grid`). Consume el generador aleatorio internamente, pero como el runner siembra `random.seed(seed)` por partida, la ejecución es **igualmente determinista** y la reproducibilidad queda garantizada **sin tocar código provisto**. El alcance se mantiene en 4×4 (tamaño del simulador y del `observation_space` del env).

### 3.8 Configuración recomendada en MATE

**Decisión:** MATE no entrena un modelo como LOST. La experimentación computa una **configuración recomendada** (técnica ganadora Minimax/Expectimax, profundidad, pesos `(w1..w4)` y métricas), pero el `.pkl` obligatorio corresponde al primer proyecto. Si se quisiera serializar esa configuración, alcanza con guardar un dict simple desde el notebook; debe tratarse como artefacto opcional/regenerable, no como requisito central.

**Por qué no más que eso:** se evaluó precalcular una tabla de política/transposición de todos los estados del 4×4 (análogo a una Q-table), pero se **descartó por sobrecomplicación**: no aporta a lo que pide la consigna y agrega código y riesgo innecesarios. El detalle operativo está en `PlanificacionMATE.md` §10.

---

## 4. Metodología experimental

**Qué se mide:**
- **Win rate** por matchup (promediado sobre las partidas del diseño apareado).
- **Nodos por jugada de nuestro agente** (`a_nodes_per_move`) — eje central del análisis de Alpha-Beta y métrica de costo *aislada por agente*.
- **Tiempo por jugada de nuestro agente** (`a_avg_move_time`) — costo computacional real **del agente**, medido por separado de cada jugador (no el promedio del juego). `play_match` devuelve además `avg_move_time` (promedio sobre ambos jugadores), pero el costo del agente se reporta con `a_avg_move_time`.
- **Largo de partida** (plies) — para caracterizar el estilo de juego.

**Cómo se asegura una comparación justa:** misma profundidad y mismas seeds entre las variantes comparadas, con **diseño apareado** (cada seed jugado por ambos lados sobre la misma posición; §3.6). El registro completo de partidas se persiste en `results.csv`; la **mejor configuración hallada** queda documentada en §5.6. Minimax/Expectimax no *entrenan* un modelo, por lo que MATE no depende de un `.pkl` obligatorio.

**Qué gráfico cuenta cada historia:**
- *Nodos vs profundidad* (escala log) y *tiempo vs profundidad* → **impacto de Alpha-Beta** (E1).
- *Win rate por matchup* (barras) → desempeño vs Random y vs Stratagem (E2/E3).
- *Win rate Minimax vs Expectimax* (barras) → la decisión técnica (E4).
- *Win rate vs profundidad* (línea) → cuánto ayuda buscar más hondo (E5).
- *Heatmap del torneo de heurísticas* → mejor combinación de pesos (E6).
- *Matriz de matchups en ambas posiciones* (barras por matchup, faceteado por profundidad) → comparación completa MM/EM vs Random/Stratagem como jugador 1 y 2, incluidos MM-MM y EM-EM (E7).

---

## 5. Resultados experimentales (corrida final)

Parámetros de la corrida final (**diseño apareado**, §3.6): `N_RANDOM=100`, `N_SELF=100`, `N_STRAT=40`, `N_HEUR=30` **seeds**, y cada seed se juega **2 veces** (una por lado) → p. ej. 80 partidas vs Stratagem por matchup (40/lado). Seeds `1000+k`. Pesos base de los agentes "principales" `{h1:1, h2:2, h3:0.5, h4:1}` salvo donde se indica. Registro completo en `results.csv` (1568 filas); gráficos en `plots/`. El costo se reporta **por agente** (`a_avg_move_time`, `a_nodes_per_move`), no el promedio del juego. Tiempo total de la corrida ≈ 15 min (dominado por los ~560 partidos vs Stratagem, que busca a d=3).

### 5.1 E1 — Impacto de Alpha-Beta (gráfico `plots/e1_alpha_beta.png`)
A igual profundidad y mismo estado, la poda devuelve el **mismo valor minimax** (verificado en el test de equivalencia de `minimax_agent.py` y sobre 292 estados; con ordenamiento, el desempate entre jugadas igualmente óptimas puede mover partidas sueltas —E7, ~9 %—, no la habilidad) y **reduce fuertemente los nodos**, cada vez más al profundizar:

| Profundidad | Nodos Minimax | Nodos Alpha-Beta | Reducción | Tiempo/jugada (mm → AB) |
|---|---|---|---|---|
| 1 | 24.5 | 24.5 | 0 % | ~0 s |
| 2 | 436 | 87 | **80 %** | 0.010 → 0.011 s |
| 3 | 3 653 | 934 | **74 %** | 0.083 → 0.064 s |
| 4 | 13 785 | 1 235 | **91 %** | 0.28 → 0.08 s |

La diferencia se vuelve dramática a d=4 (≈11× menos nodos, ≈4× menos tiempo), confirmando que Alpha-Beta es la palanca que vuelve viable buscar más hondo con el alto *branching factor* de Isolation. (E1 mide búsquedas aisladas, así que no depende del diseño apareado.)

### 5.2 E2 — vs RandomAgent (`plots/e2_e3_winrate.png`, izq.)
Ambas técnicas **dominan** al azar: **Minimax 96 %**, **Expectimax 94.5 %** (200 partidas c/u). Sanity check superado.

### 5.3 E3/E4 — Minimax vs Expectimax (la decisión técnica)
**vs Stratagem (80 partidas por celda, 40/lado), por profundidad** (`plots/e2_e3_winrate.png`, der.):

| Técnica | d=2 | d=3 (parejo con Stratagem) |
|---|---|---|
| Minimax | 39 % | **46 %** |
| Expectimax | 48 % | 34 % |

**Enfrentamiento directo (E4, d=2, 200 partidas):** Minimax 44 % / Expectimax 56 % — leve ventaja de Expectimax a profundidad baja. (E4 es un cara a cara al *depth* por defecto; el contraste por profundidad frente a un rival común y fuerte lo da E3, que cubre d=3.)

**Costo por agente a d=3 (E3):** Expectimax cuesta **0.59 s y ~24.100 nodos por jugada**, contra **0.13 s y ~1.300 nodos** de Minimax: Expectimax es **~4.5× más lento y ~18× más nodos**, porque sus nodos de azar **no podan** (promedian todas las ramas). Es decir, a profundidad igualada Expectimax es a la vez **más débil y más caro**.

**Conclusión (respuesta a "¿cuál técnica es mejor?"): depende del oponente y de la profundidad.**
- Frente a un rival **estocástico** (Random), ambas dominan.
- Frente a un rival **determinista** (Stratagem), **a profundidad igualada (d=3) gana Minimax** (46 % vs 34 %) y además es mucho más barato. Profundizar **mejora** a Minimax (39 %→46 %) pero **empeora** a Expectimax (48 %→34 %): propagar más hondo un modelo de oponente *uniforme* —que es **incorrecto** para Stratagem— degrada el juego. Esto confirma la predicción teórica de §3.3.
- La inversión a d=2 (Expectimax por encima) muestra que la ventaja de Minimax **requiere profundidad suficiente**; con poco lookahead ninguna técnica modela bien al rival.

> **Importante (E7, §5.7):** la ventaja de Minimax vs Stratagem **depende de la heurística**. El resultado 46/34 corresponde a la ponderación *balanceada*; con la mejor heurística (`h2`), Minimax y Expectimax quedan **parejos**. Lo robusto a favor de Minimax es el **costo** (poda), no el win rate.

### 5.4 E5 — Efecto de la profundidad (`plots/e5_depth.png`)
Minimax vs Stratagem (80 partidas por profundidad): win rate **monótonamente creciente** — **27 % (d=1) → 39 % (d=2) → 46 % (d=3)**. Buscar más hondo ayuda de forma consistente.

### 5.5 E6 — Torneo de heurísticas (`plots/e6_heatmap.png`)
Round-robin de 4 ponderaciones con Minimax (60 partidas/par). Las cuatro se eligieron para probar la **hipótesis de que la movilidad alcanza**: una con **solo** la diferencia de movilidad (`solo_mov_diff`), dos que le **agregan una componente** (`mov+centro`, `mov+acorralar`) y una que **combina las cuatro** (`balanceada`). Win rate promedio:

| Ponderación | Win rate prom. |
|---|---|
| **`solo_mov_diff`** (solo h2) | **0.700** |
| `mov+centro` (h2+h3) | 0.578 |
| `balanceada` (h1+h2+h3+h4) | 0.483 |
| `mov+acorralar` (h2+h4) | 0.239 |

La **diferencia de movilidad sola (h2)** es la combinación más fuerte: es la señal más directamente ligada a la condición de derrota (quedarse sin movimientos), y agregarle otras componentes (sobre todo "acorralar") tiende a **diluirla**. Esto valida empíricamente la elección de h2 como núcleo de la evaluación (§3.4, criterio 3 de la lám. 16).

### 5.6 Configuración recomendada

La experimentación computa la **mejor configuración**: `{tecnica: "minimax", profundidad: 3, pesos: {h2: 1.0} (solo_mov_diff)}`, con métricas asociadas (`win_vs_stratagem_d3=0.463`, `win_vs_random=0.96`, `e6_winrate=0.70`). Esta configuración puede reconstruirse directamente desde `MinimaxAgent` + `weighted_eval`; se serializa además en `mate_best_config.pkl` (regenerable con `save_best_config.py`), aunque MATE no requiere un `.pkl` obligatorio.

### 5.7 E7 — Matriz completa de matchups en ambas posiciones (`plots/e7_required_matchups.png`)

Respondiendo al **pedido explícito de la cátedra**, se corrió la **matriz completa**: los 12 emparejamientos entre **Minimax, Expectimax, Random y Stratagem** en **ambas posiciones** (jugador 1 y 2), incluyendo los *mirror matches* **MM-MM** y **EM-EM**, variando profundidad (d ∈ {2,3}), heurística y poda on/off — **960 partidas**, registradas en `required_matchups_results.csv` (con columnas `player1_agent`/`player2_agent` explícitas). El grid lo genera `run_required_experiments.py` y lo analiza `analyze_required_matchups.py`. Cada celda = 10 seeds (granularidad 10 %; para comparaciones **finas** de técnica valen E3/E5 con N=40).

**Lecturas (con `h2`, α-β on):**
- **vs Random — dominio total:** Minimax y Expectimax ganan **100 %** arrancando y de segundos (las filas "Random vs …" dan 0 %). Robusto en ambas posiciones.
- **Mirror matches (MM-MM, EM-EM) — la ventaja de primer jugador depende de la profundidad:** a d=2 el jugador 1 gana **80 %**; a d=3 baja a **40 %** (buscar más hondo iguala el juego entre agentes idénticos; con n=10, indicativo).
- **vs Stratagem — competitivo en ambas posiciones:** Minimax ~50–70 % de primero y ~40–60 % de segundo; Expectimax con `h2` incluso algo mejor (90 % de primero a d=2). **El claro Minimax > Expectimax de E3 (46/34) no se reproduce con `h2`** — fue en parte efecto de la ponderación *balanceada*. Con la mejor heurística, las dos técnicas quedan **parejas**.
- **α-β no cambia la habilidad:** valor minimax idéntico; solo difieren ~9 % de partidas por **desempate** entre jugadas igualmente óptimas. El beneficio medible es el **costo**: a d=3 poda **~92 %** de los nodos (24 279 → 1 919 por jugada) — reconfirma E1 en partidas reales.

**Conclusión actualizada:** "¿cuál técnica conviene?" depende del **oponente, la profundidad y la heurística**. Lo robusto: ambas dominan a Random; vs Stratagem quedan parejas a profundidad igualada; y la ventaja sólida de Minimax es el **costo** (poda; Expectimax no poda).

---

## 6. Riesgos conocidos y notas de advertencia

- **Explosión combinatoria:** el alto branching factor (~100/ply) puede hacer lenta la búsqueda profunda. Mitigado con Alpha-Beta, ordenamiento de movimientos y profundidad acotada (3–4).
- **Costo de `get_possible_actions`:** genera un `clone()` por cada dirección y recorre todas las celdas para listar destrucciones; en búsqueda profunda este costo **domina**. Es una limitación del simulador dado; se mitiga limitando profundidad y paralelizando partidas.
- **Memoización con simetrías (mejora opcional, no implementada):** el teórico y la guía sugieren cachear estados equivalentes por **rotaciones y reflexiones** del tablero (transposition table) para no recalcular ramas simétricas. No se implementó porque **Alpha-Beta + profundidad acotada** ya hacen tratable el 4×4 (E1: ~92 % de poda a d=3); queda como mejora futura para profundidades mayores o tableros más grandes.
- **Expectimax vs Stratagem:** que Expectimax rinda peor que Minimax contra Stratagem **a profundidad igualada (d=3)** es **lo esperado** (modelo de oponente uniforme frente a un rival determinista), no un error de implementación — y así se **confirmó** en E3 (§5.3). Nota: a profundidad baja (d=2) el resultado se invierte; la conclusión teórica aplica con profundidad suficiente. **Matiz E7 (§5.7):** con la mejor heurística `h2`, Minimax y Expectimax quedan **parejos** vs Stratagem (el 46/34 es con *balanceada*); la ventaja **robusta** de Minimax es el **costo** (poda), no el win rate.
- **Varianza por arranque aleatorio:** con pocas partidas las conclusiones serían ruidosas; por eso se usa el **diseño apareado** (§3.6) con decenas de seeds por matchup.
- **Ventaja de primer jugador (medida, ahora controlada):** en Isolation 4×4 arrancar **importa mucho**. Medido sobre la corrida apareada: nuestro agente gana **48 % cuando arranca** vs **35 % cuando arranca el rival** (≈13 pts). El **diseño apareado neutraliza este sesgo de forma controlada**: como cada seed se juega por **ambos lados sobre la misma posición**, la comparación entre técnicas/heurísticas no depende de quién arrancó ni de qué posiciones tocaron. (El enfoque inicial —alternar lados con seeds distintos— solo lo promediaba; el apareo lo controla.)
- **Bug del oponente `Stratagem` como jugador 2 (no es nuestro código):** su Minimax interno evalúa **su propia derrota como 0 en vez de −1** cuando juega de jugador 2 (verificado con un test directo: `Stratagem(1)` la evalúa bien en −1, `Stratagem(2)` la evalúa en 0). Consecuencia: `Stratagem` **no evita con fuerza las líneas perdedoras dentro de su horizonte cuando juega segundo**, por lo que es **algo más débil de jugador 2**. No lo corregimos (no se modifican archivos dados); se documenta porque **infla levemente** nuestras victorias cuando Stratagem va segundo. Con el diseño apareado, el efecto queda balanceado entre las variantes comparadas.
- **Métrica de costo por agente (resuelto):** además del promedio del juego (`avg_move_time`, sobre ambos jugadores), `play_match` reporta el tiempo de **cada jugador por separado**, y `run_matchup` registra el costo de **nuestro** agente: `a_avg_move_time` (tiempo/jugada) y `a_nodes_per_move` (nodos/jugada). Así el costo del agente no queda contaminado por el del rival (p. ej. el lento Minimax d=3 de Stratagem). El experimento **E1** complementa con la medición aislada con/sin Alpha-Beta.

---

## 7. Mapeo a la consigna

| Requisito de la consigna | Dónde se cubre |
|--------------------------|----------------|
| Minimax con Alpha-Beta + análisis de impacto | §2, §3.1, §3.2; experimento E1 (§5.1) |
| Expectimax + decidir mejor técnica | §2, §3.3; experimentos E2–E4 (§5.2, §5.3) y **E7** (§5.7) |
| **Comparar MM/EM entre sí y vs Random/Stratagem en ambas posiciones** (MM-MM, EM-EM, …) | **E7** (§5.7): matriz de 12 matchups × 2 posiciones × prof. × heurística × α-β; `run_required_experiments.py` |
| Funciones de evaluación, combinaciones y ponderaciones (+ escalado) | §3.4 (incl. nota de escalado); experimento E6 (§5.5) |
| Definir pruebas + registro completo de resultados | §4, §5; `results.csv` (1568) + `required_matchups_results.csv` (960); tests en `smoke_tests.py` |
| Informe: resumen del abordaje (simulador, **parámetros**, **tiempo de ejecución**, resultados) | §1, §3, §4, §5 |
| Apoyo visual (gráficos claros + comentarios) | §5 + `plots/` (5 PNG, incl. matriz E7) |
| Notas de advertencia (dificultades y por qué no se resolvieron) | §6 |
| Modelos computados (`.pkl` / formato similar) | Requisito obligatorio cubierto por LOST; MATE documenta configuración recomendada (§3.8, §5.6) |
| Entregables (.py + .ipynb, Poetry separado, único `.zip`) | `PlanificacionMATE.md` §1 (Entregables) y §9 |
| Informe claro, legible, autocontenido, ≤ 20 págs. + anexos | `PlanificacionMATE.md` §1 (Entregables) |
