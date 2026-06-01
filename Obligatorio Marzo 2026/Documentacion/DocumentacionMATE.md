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

verificando además que el **movimiento elegido es idéntico** (la poda no cambia la decisión, solo el costo). El ordenamiento de movimientos se justifica porque la poda es máxima cuando se exploran primero las jugadas más prometedoras.

### 3.3 Minimax vs Expectimax: cuándo conviene cada uno
**Decisión:** implementar ambos y **dejar que los datos decidan**, sin asumir un ganador a priori.
**Justificación (teórica):** Minimax asume un rival **óptimo/adversarial**; su valor es una cota inferior garantizada (lám. 12). Expectimax asume un rival **estocástico** y modela sus acciones con `σ` uniforme (lám. 8). De ahí la **hipótesis** a confirmar experimentalmente:
- **vs `RandomAgent`** (que es genuinamente estocástico): Expectimax debería **rendir mejor**, porque su modelo del rival es correcto y puede explotar jugadas riesgosas con alto valor esperado.
- **vs `Stratagem`** (que es Minimax determinista, *no* uniforme): el modelo uniforme de Expectimax es **incorrecto**, por lo que se espera que **Minimax rinda igual o mejor**.

La respuesta a *"¿cuál técnica es mejor para este caso?"* es por tanto **"depende del oponente"**, y se sustenta con E2–E4. Importante: si Expectimax pierde contra Stratagem, **no es un bug**, es la consecuencia esperada de un modelo de oponente equivocado.

### 3.4 Elección de las funciones de evaluación
**Decisión:** una biblioteca de componentes combinables por pesos: movilidad propia (h1), diferencia de movilidad (h2), control de centro (h3) y acorralar al rival (h4), con `Eval(s) = Σ wᵢ·hᵢ(s)`.
**Justificación:**
- En Isolation **perder = quedarse sin movimientos**, por lo que la **movilidad** (casillas adyacentes libres) es la señal más directamente correlacionada con ganar — cumple el criterio (3) de la lám. 16. La **diferencia de movilidad** (h2) captura la naturaleza de suma cero.
- **Control de centro** (h3): desde el centro hay más casillas alcanzables, lo que tiende a preservar movilidad futura.
- **Acorralar** (h4): destruir casillas alrededor del rival reduce su movilidad — ataque directo a su condición de derrota.
- Estos cuatro componentes **replican y generalizan** la heurística de `Stratagem`, lo que nos da un punto de comparación honesto y nos permite experimentar con **ponderaciones** (consigna 2) buscando combinaciones que la superen.

**Medida de movilidad:** se cuenta el número de **casillas adyacentes libres** (estilo `Board.has_valid_moves`) y **no** `len(get_possible_actions)`, porque esta última multiplica por cada celda destruible e infla artificialmente el valor sin reflejar mejor la posición.

### 3.5 Convención de signo y utilidad terminal
**Decisión:** `heuristic_utility` y la utilidad terminal se expresan **desde la perspectiva del agente**: positivo = bueno para el agente. Utilidad terminal = **+1** (gana el agente), **−1** (pierde), **0** (empate/no decidido).
**Justificación:** mantener un único marco de referencia evita errores de signo en los nodos `min` y de azar, y es coherente con `Stratagem`, que calcula su heurística relativa a `self.player`. Cumple el criterio (1) de la lám. 16 (`Eval(win) > Eval(draw) > Eval(loss)`).

### 3.6 Reproducibilidad de los experimentos
**Decisión:** sembrar `random` por partida, **alternar quién arranca** y correr **N ≥ 100** partidas por matchup.
**Justificación:** `Board.place_players()` usa `random.shuffle` **sin semilla**, así que sin intervención las partidas no son reproducibles. Además, la colocación inicial aleatoria introduce **varianza** y existe una **ventaja de primer jugador**; promediar sobre muchas partidas y **alternar lados** neutraliza ambos sesgos y hace que la comparación entre agentes/heurísticas sea **justa**.

### 3.7 Restricción de tablero 4×4 por `clone()`
**Decisión:** trabajar sobre **4×4** y parchear `Board.clone()` defensivamente.
**Justificación:** `clone()` reconstruye con `Board()` (tamaño por defecto 4×4) y solo copia el `grid`; si se usara un tablero de otro tamaño, el clon quedaría mal dimensionado. Como la búsqueda **depende de `clone()`** para generar sucesores, dejamos el parche que preserva `board_size` para robustez, pero el alcance del trabajo se mantiene en 4×4 (tamaño del simulador provisto y del `observation_space` del env).

### 3.8 "Modelo computado" en MATE: qué entregamos y por qué (entregamos `.pkl`)
**Decisión:** **MATE entrega su propio `.pkl`** (no se trata como opcional). Concretamente dos artefactos en `Isolation/models/`:
1. `mate_best_config.pkl` — la **mejor configuración** hallada por experimentación: técnica ganadora (Minimax/Expectimax), profundidad y pesos `(w1..w4)`.
2. `mate_policy.pkl` — una **tabla de política / transposición precalculada**: `estado_canónico → mejor_acción`, a modo de *libro de aperturas* del 4×4.

**Justificación:** la sección *Auditoría* pide, **en general**, *"los modelos computados (.pkl o formatos similares)"*; la cláusula de **penalización** ("al menos un modelo, o el ejercicio se considera no hecho") solo nombra explícitamente el **primer ejercicio (LOST)**. La redacción es **ambigua respecto de MATE**, y el costo de cubrirse es mínimo, así que entregamos `.pkl` igual.

Más allá de la prudencia: Minimax/Expectimax no *aprenden* un modelo como Q-Learning, pero sí **computan** algo serializable. En un tablero **4×4** el espacio de estados alcanzables es acotado, por lo que precalcular la **decisión óptima por estado** y serializarla es literalmente *computar offline un modelo de decisión* que luego el agente **consume por lookup O(1)** sin volver a buscar — el análogo adversarial de una Q-table. Esto convierte el `.pkl` de MATE en un **modelo computado sustantivo**, no en un placeholder. Si la tabla completa resultara demasiado grande o lenta de generar, se reduce a un **libro de aperturas** (primeros k plies), que sigue siendo un modelo válido. El detalle operativo está en `PlanificacionMATE.md` §10.

---

## 4. Metodología experimental

**Qué se mide:**
- **Win rate** por matchup (promediado sobre N partidas, lados alternados).
- **Nodos expandidos** (contador en el agente) — eje central del análisis de Alpha-Beta.
- **Tiempo por jugada** — costo computacional real.
- **Largo de partida** (plies) — para caracterizar el estilo de juego.

**Cómo se asegura una comparación justa:** misma profundidad, mismas seeds, swap de lados y mismo N entre las variantes que se comparan. Los resultados se persisten en CSV/JSON (no se guarda `.pkl`: MATE no entrena un modelo).

**Qué gráfico cuenta cada historia:**
- *Nodos vs profundidad* (escala log) y *tiempo vs profundidad* → **impacto de Alpha-Beta** (E1).
- *Win rate por matchup* (barras) → desempeño vs Random y vs Stratagem (E2/E3).
- *Win rate Minimax vs Expectimax* (barras) → la decisión técnica (E4).
- *Win rate vs profundidad* (línea) → cuánto ayuda buscar más hondo (E5).
- *Heatmap del torneo de heurísticas* → mejor combinación de pesos (E6).

---

## 5. Riesgos conocidos y notas de advertencia

- **Explosión combinatoria:** el alto branching factor (~100/ply) puede hacer lenta la búsqueda profunda. Mitigado con Alpha-Beta, ordenamiento de movimientos y profundidad acotada (3–4).
- **Costo de `get_possible_actions`:** genera un `clone()` por cada dirección y recorre todas las celdas para listar destrucciones; en búsqueda profunda este costo **domina**. Es una limitación del simulador dado; se mitiga limitando profundidad y paralelizando partidas.
- **Expectimax vs Stratagem:** si Expectimax rinde peor que Minimax contra Stratagem, **es lo esperado** (modelo de oponente uniforme frente a un rival determinista), no un error de implementación. Debe quedar documentado como nota de advertencia.
- **Varianza por arranque aleatorio:** con pocas partidas las conclusiones serían ruidosas; por eso N ≥ 100 y lados alternados.

---

## 6. Mapeo a la consigna

| Requisito de la consigna | Dónde se cubre |
|--------------------------|----------------|
| Minimax con Alpha-Beta + análisis de impacto | §2, §3.1, §3.2; experimento E1 |
| Expectimax + decidir mejor técnica | §2, §3.3; experimentos E2–E4 |
| Funciones de evaluación, combinaciones y ponderaciones | §3.4; experimento E6 |
| Definir pruebas + registro completo de resultados | §4; experimentos E1–E6 (CSV/JSON) |
| Informe: resumen del abordaje (simulador, **parámetros**, **tiempo de ejecución**, resultados) | §1, §3, §4 |
| Apoyo visual (gráficos claros + comentarios) | §4 (gráficos) |
| Notas de advertencia (dificultades y por qué no se resolvieron) | §5 |
| Modelos computados (`.pkl` / formato similar) | §3.8 (`mate_best_config` + `results/`) |
| Entregables (.py + .ipynb, Poetry separado, único `.zip`) | `PlanificacionMATE.md` §1 (Entregables) y §9 |
| Informe claro, legible, autocontenido, ≤ 20 págs. + anexos | `PlanificacionMATE.md` §1 (Entregables) |
