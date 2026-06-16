# Documentación Final — Obligatorio de Inteligencia Artificial (Marzo 2026)

**Materia:** Inteligencia Artificial — Ingeniería en Sistemas — Universidad ORT
**Proyecto:** Agente inteligente del rover marciano *"Out for Delivery"* (empresa ficticia *Red Destination™*)
**Integrantes:** equipo de 2 personas (por tamaño del grupo, no aplica la tarea adicional de Stochastic Q-Learning / MCTS)

> Este documento es la documentación de entrega. Reúne **todo** lo relevante de la resolución: qué hace cada agente, **cómo** se resolvió, **por qué** se tomó cada decisión, **qué caminos y errores** hubo y cómo se resolvieron, y **qué resultados** se obtuvieron. Cada afirmación está respaldada por el código y los artefactos del repositorio, y se puede **reproducir y verificar** con los comandos del Anexo B.

## Índice

- **0.** Resumen ejecutivo
- **1.** Contexto y consigna
- **2.** Proyecto LOST — Mountain Car Continuous (Q-Learning)
  - 2.1 Problema (la "trampa") · 2.2 MDP · 2.3 Discretización · 2.4 Q-Learning · 2.5 Inicialización optimista · 2.6 Metodología (varianza/seaborn) · 2.7 Hiperparámetros (grid + varianza) · 2.8 Dyna-Q · 2.9 Modelos y política · 2.10 Reward shaping (extra) · 2.11 Conclusiones
- **3.** Proyecto MATE — Isolation (Minimax / Expectimax)
  - 3.1 Problema · 3.2 Teoría · 3.3 Minimax + Alpha-Beta · 3.4 Funciones de evaluación · 3.5 Expectimax · 3.6 Metodología · 3.7 Resultados E1–E6 · 3.8 Modelo computado · 3.9 Errores y notas · 3.10 Conclusiones
- **4.** Tecnologías y librerías
- **5.** Entregables y estado
- **6.** Uso de IA generativa
- **Anexo A** — Cumplimiento de la consigna (checklist)
- **Anexo B** — Reproducibilidad y verificación (comandos y salidas reales)

---

# 0. Resumen ejecutivo

El obligatorio tiene **dos proyectos independientes**, de naturaleza distinta:

| | **LOST** | **MATE** |
|---|---|---|
| Ambiente | `MountainCarContinuous-v0` (Gymnasium) | Juego de mesa *Isolation* (4×4) |
| Paradigma | **Aprendizaje por refuerzo** | **Búsqueda adversarial** |
| Técnica | Q-Learning tabular + Dyna-Q | Minimax + Alpha-Beta + Expectimax |
| El agente… | **aprende** una política por prueba y error | **calcula** la mejor jugada simulando el árbol de juego |
| Resultado titular | Resuelve el ambiente al **100 %** con la **recompensa real** (clave: inicialización optimista) | A profundidad igualada, **Minimax > Expectimax** (46 % vs 34 %) contra el rival fuerte |

Cada proyecto vive en su propia carpeta con su **entorno Poetry separado**: `MountainCarContinuous/` (LOST) e `Isolation/` (MATE).

---

# 1. Contexto y consigna

*Red Destination™* contrata al equipo para implementar el agente del rover. La consigna pide dos proyectos:

- **LOST** (*Learning-based Orientation and Steering for Traversal*): controlar el rover en `MountainCarContinuous-v0` con **Q-Learning**, justificando la **discretización** y la **búsqueda de hiperparámetros**, e implementando **Dyna-Q** (Sutton & Barto, cap. 8.1–8.2) como componente de investigación.
- **MATE** (*Martian Adversarial Tactics Engine*): un agente para *Isolation* con **Minimax (con Alpha-Beta) y Expectimax**, decidiendo cuál conviene, con **funciones de evaluación** experimentables y un **registro completo** de resultados.

La entrega pide: código (`.py` + `.ipynb`), modelos computados (`.pkl`), un informe en PDF (≤ 20 págs. + anexos) y todo en un único `.zip`, con entornos Poetry separados.

---

# 2. Proyecto LOST — Mountain Car Continuous

> **Nota sobre el enfoque.** Una primera versión de este proyecto se apoyaba en *reward
> shaping*. Siguiendo la devolución de la cátedra (no es aceptable modificar las
> recompensas: equivale a cambiar el ambiente), se rehízo todo con la **recompensa real**.
> El hallazgo fue que la palanca para aprender no era el shaping sino la **exploración**
> (inicialización optimista). El shaping quedó como un **extra** (§2.10).

## 2.1 El problema y por qué es difícil (la "trampa de no hacer nada")

En `MountainCarContinuous-v0` un auto en un valle debe llegar a la bandera de la cima. El motor **no tiene fuerza suficiente** para subir de frente: la única solución es **balancearse** (ir hacia atrás para tomar impulso y luego subir). La recompensa es **muy escasa (sparse)**:

- `−0.1·a²` por cada paso (penalización por gastar energía),
- `+100` solo al llegar a la meta (`x ≥ 0.45`).

Esto crea una **trampa** sutil: como **moverse cuesta** energía y la meta casi nunca se alcanza explorando al azar, la política que Q-Learning aprende por defecto es **"quedarse quieto"** (acción 0, costo 0), que parece mejor que moverse y pagar sin llegar nunca. Sin una exploración fuerte, el agente **converge a una mala política** (no se mueve), no solo aprende lento. Verificado: Q-Learning con la recompensa real y exploración estándar da **0 % de éxito**.

## 2.2 Modelado como MDP

| Componente | Decisión |
|---|---|
| Estado | observación continua `(x, v)` → discretizada a índices de grilla |
| Acciones | fuerza continua `a ∈ [−1, 1]` → discretizada a N acciones |
| Transición | desconocida para Q-Learning; **Dyna-Q la aprende** como modelo tabular |
| Reward | el **real** del ambiente (sin modificar) |
| Fin | `terminated` = llegó a la meta; `truncated` = corte por límite de pasos (999) |

## 2.3 Discretización — `discretization.py`

Q-Learning tabular necesita espacios finitos. La clase `Discretizer` parametriza `n_bins_x`, `n_bins_v` y `n_actions`.

**Decisiones y por qué:**
- **Cortes uniformes** con `np.linspace` + `np.digitize`: simple, predecible y reproducible. Cuantiles (sklearn) tendrían sentido si la distribución de estados fuese muy desbalanceada, pero el auto recorre todo el rango con razonable uniformidad.
- **`state_shape = (n_bins + 1, …)`**: `np.digitize` puede devolver el índice `len(bins)` en el extremo; el `+1` evita un *off-by-one* en el tamaño de Q.
- **Número impar de acciones** (3, 5, …): garantiza la acción **`0.0` (no empujar)**.
- Se parametriza en una **clase** para comparar resoluciones en el grid sin reescribir el agente.

Se exploraron **discretizaciones diversas** (gruesa 20×20×3, media 40×40×5, fina 60×60×7); ver el impacto en §2.7.

## 2.4 Q-Learning — `q_learning_agent.py`

Regla de actualización off-policy TD (QL.pdf slide 6; Sutton & Barto 6.5):

```
Q(s,a) ← Q(s,a) + α·[ r + γ·max_a' Q(s',a') − Q(s,a) ]
```

**El núcleo del loop (para no abrir el archivo):**

```python
def _epsilon_greedy(self, state):
    if random.random() < self.epsilon:                       # explorar
        return random.randint(0, self.discretizer.n_actions - 1)
    return int(np.argmax(self.Q[state]))                     # explotar (greedy)

# ... por cada paso real:
action_idx = self._epsilon_greedy(state)
next_obs, reward, terminated, truncated, _ = env.step(self.discretizer.action_from_idx(action_idx))
next_state = self.discretizer.get_state(next_obs)
future = 0.0 if terminated else float(np.max(self.Q[next_state]))   # ← terminated/truncated
self.Q[state][action_idx] += self.alpha * ((reward + self.gamma * future) - self.Q[state][action_idx])
```

**Decisiones de diseño:**
- **Q-Learning (off-policy) y no SARSA**: permite explorar con ε alto sin que esa exploración deteriore la política objetivo (siempre la greedy).
- **ε-greedy con decay exponencial por episodio** (no por paso): decaer por paso apaga la exploración demasiado rápido. Con `ε₀=1.0`, `decay=0.999`, la exploración dura cientos de episodios.
- **`terminated` vs `truncated`** (API Gymnasium ≥0.26): el bootstrap futuro es `0` **solo** si `terminated` (estado terminal del MDP = meta); si `truncated` (timeout de 999 pasos), `s'` **no** es terminal y se bootstrappea normal (`γ·max Q(s')`). Tratar `truncated` como `terminated` sesga `Q` hacia abajo.

## 2.5 Aprender sin tocar la recompensa: inicialización optimista (la palanca)

Para salir de la "trampa de no hacer nada" sin modificar la recompensa, la técnica legítima es la **inicialización optimista** (Sutton & Barto §2.6): arrancar la tabla Q en un valor alto hace que las acciones **no probadas** "se vean mejores", forzando al agente a **explorarlas sistemáticamente** → muchas más chances de tropezar con la meta.

```python
# Q(s,a) arranca en `optimistic_init` (en vez de 0): exploración estructurada.
self.Q = np.full(discretizer.q_shape, optimistic_init, dtype=np.float64)
```

**Evidencia (medida):** con la recompensa real, sin optimismo el agente fracasa; con optimismo, resuelve el ambiente:

| Inicialización (`optimistic_init`) | Éxito en test |
|---|---|
| 0 (estándar) | **0 %** (la trampa) |
| 1 | 0 % |
| 10 | **100 %** |
| 50 | **100 %** |

Es decir: **la clave nunca fue el reward shaping, sino explorar bien.** Esto es más fiel al espíritu del problema (no se cambia el ambiente) y es un resultado más honesto.

## 2.6 Metodología experimental: múltiples seeds, varianza y seaborn — `experiments.py`

Siguiendo la devolución de la cátedra, la experimentación se diseñó para **medir la varianza**:

- **Cada configuración se corre con varias semillas** (`SEEDS = [0,1,2,3,4]`): cada corrida siembra el agente (`random`+`numpy`) y el env de forma independiente y reproducible.
- **Comparación apareada:** las mismas semillas para todas las variantes (las diferencias no se deben a seeds distintas).
- **Evaluación consistente:** el test se mide sobre un conjunto **fijo** de episodios sembrados (`TEST_SEEDS`), igual para todas las configs (refleja la política, no la suerte del reset).
- **Visualizaciones con `seaborn`:** **boxplots** de las métricas finales y **curvas con banda de error** (dispersión entre semillas).

## 2.7 Exploración de hiperparámetros (grid OAT con varianza) — `grid_search.py`

Estrategia **one-at-a-time (OAT)**: desde una config base, se varía **un** hiperparámetro a la vez (interpretable). Se exploraron **inicialización optimista, discretización, α, γ y ε-decay**, cada uno con las **5 seeds** (recompensa real, sin shaping).

**Efecto de la inicialización optimista** (curvas con banda de error):

![Efecto de la inicialización optimista — curvas con banda de error](MountainCarContinuous/plots/grid_search_optinit_curves.png)

**Éxito en test por configuración** (boxplots, 5 seeds):

![Grid search — éxito en test por configuración (boxplots, 5 seeds)](MountainCarContinuous/plots/grid_search_box_success.png)

![Grid search — pasos en test por configuración (boxplots, 5 seeds)](MountainCarContinuous/plots/grid_search_box_steps.png)

**Resultados (mediana / varianza sobre 5 seeds):**

| Config (OAT) | éxito mediana | éxito **mín** | reward **std** | pasos mediana |
|---|---|---|---|---|
| **α=0.3** ⭐ | 100 % | **100 %** | **0.56** | 157 |
| opt=50 | 100 % | 100 % | 1.08 | 188 |
| α=0.05 | 100 % | 100 % | 1.93 | 265 |
| base (bins40, opt=10) | 100 % | 80 % | 11.31 | 190 |
| decay=0.995 | 100 % | 80 % | 14.10 | 191 |
| bins60 (fina) | 100 % | 90 % | 5.39 | 481 |
| gamma=0.95 | 100 % | **50 %** | 27.83 | 250 |
| bins20 (gruesa) | 100 % | **0 %** | 62.72 | 155 |
| opt=0 / opt=1 | **0 %** | 0 % | 0.00 | 999 |

**Lectura (el análisis de varianza es la clave):**
- **La inicialización optimista es decisiva:** `opt=0` y `opt=1` fracasan (0 %); `opt≥10` resuelve.
- **Elegir por la mediana sola engañaría.** `bins20 (gruesa)` tiene la mejor mediana de pasos (155) **pero una seed da 0 %** (`std`=62.72): es **inestable**. Lo mismo, en menor grado, `gamma=0.95`.
- La **elección robusta** es la que resuelve en **todas** las seeds con **baja varianza**: **`α=0.3`** (éxito mín 100 %, `std`=0.56, 157 pasos). Por eso el criterio de selección prioriza `mín(éxito)` y `varianza`, no la mediana.

## 2.8 Dyna-Q + comparación con Q-Learning (componente de investigación) — `dyna_q_agent.py`, `compare_dyna_q.py`

**Dyna-Q** (Sutton & Barto §8.2) = Q-Learning + un **modelo aprendido del ambiente** + `n` pasos de **planning** (actualizaciones simuladas) por cada paso real:

```python
self._q_update(state, a, reward, next_state, terminated)    # (d) update con experiencia REAL
self.model[(state, a)] = (reward, next_state, terminated)   # (e) guardar transición
self._planning()   # (f) n updates SIMULADOS con la MISMA regla, muestreando del modelo
```

Comparación `planning_steps ∈ {0, 5, 25}` (0 = Q-Learning puro), recompensa real, 5 seeds, apareado:

| Variante | éxito (mín) | reward std | pasos mediana | convergencia (ep) | tiempo |
|---|---|---|---|---|---|
| Q-Learning (n=0) | 100 % | 0.94 | 203 | 520 | 7 s |
| **Dyna-Q (n=5)** | 100 % | 1.53 | **105** | **349** | 22 s |
| Dyna-Q (n=25) | **10 %** | 66.90 | 238 | 544 | 64 s |

![Dyna-Q vs Q-Learning — pasos en test (boxplots, 5 seeds)](MountainCarContinuous/plots/dyna_q_box_steps.png)

![Dyna-Q vs Q-Learning — curvas de aprendizaje con banda de error](MountainCarContinuous/plots/dyna_q_curves.png)

**Conclusión (matizada, con varianza):**
- **Planning moderado (n=5) ayuda:** converge antes (349 vs 520 episodios) y da **mejor política** (105 vs 203 pasos), al costo de ~3× más cómputo. Es el beneficio que predice el libro: el planning **amortiza** cada experiencia real.
- **Demasiado planning (n=25) perjudica:** se vuelve **inestable** (una seed casi falla, `std`=66.9), porque replicar transiciones de un **modelo imperfecto** amplifica el ruido, y el costo crece lineal.
- Q-Learning solo es **el más estable y barato**, pero converge más lento y a una política menos eficiente.

## 2.9 Modelos finales y visualización de la política

Modelos entregables (recompensa real, sin shaping, 2500 episodios):

| Modelo | Config | éxito test | reward | pasos | cobertura Q |
|---|---|---|---|---|---|
| `q_learning_best.pkl` | α=0.3, opt=10, n=0 | 100 % | 91.4 | 134 | ~60 % |
| `dyna_q_best.pkl` | n=5, opt=10 | 100 % | **94.6** | **79.6** | ~62 % |

![Curva de aprendizaje del Q-Learning final sin shaping](MountainCarContinuous/plots/q_learning_best_curve.png)

**Verificación de la política** (`visualize_policy.py`): el mapa de `π(s)` en el espacio `(x, v)` muestra el patrón clásico **"pump-and-go"** — cuando `v < 0` predomina empujar hacia atrás (**−1**, 57.2 %) para acumular momento, y cuando `v > 0` predomina empujar hacia adelante (**+1**, 56.5 %). Las celdas no visitadas (~35 %) se **enmascaran honestamente**.

![Política aprendida — V(s) y π(s) en el espacio de estado](MountainCarContinuous/plots/q_learning_best_policy.png)

## 2.10 Reward shaping (EXTRA, no parte del núcleo)

Como **adicional**, se exploró *reward shaping* **potential-based** (Ng, Harada & Russell, 1999): un término `F(s,s') = γ·Φ(s') − Φ(s)` con `Φ(s)=coef·|v|`. Por el teorema NHR-99, esta forma **no cambia la política óptima**; solo **acelera** la propagación de la señal. Es una técnica válida como *extra*, pero **modifica la recompensa**, por lo que —siguiendo a la cátedra— **no se usa en el núcleo** ni en los resultados pedidos: todo lo anterior se obtuvo con la **recompensa real**.

## 2.11 Conclusiones (personales) y fuentes

**Conclusiones nuestras:**
- La dificultad real de `MountainCarContinuous` con la recompensa original **no es la discretización ni la regla de Q-Learning, sino la exploración**: hay que escapar de la "trampa de no hacer nada". La **inicialización optimista** lo logra **sin tocar el ambiente**, y resuelve el problema al 100 %.
- **El análisis de varianza cambió nuestra elección**: la config con mejor mediana (`bins20`) era inestable; la robusta es `α=0.3`. Reportar varianza (boxplots) no es un adorno: evita conclusiones falsas.
- **Dyna-Q confirma el libro con matices**: planning moderado (n=5) acelera y mejora; en exceso (n=25) desestabiliza. No es "Dyna-Q siempre mejor", sino "depende de cuánto planning".
- El reward shaping **funciona pero es un atajo**: al quitarlo, entendimos mejor qué resolvía realmente el problema.

**Fuentes:** Sutton & Barto, *Reinforcement Learning: An Introduction* (2ª ed.) — Q-Learning §6.5, inicialización optimista §2.6, Dyna-Q §8.2; documentación de **Gymnasium** (`MountainCarContinuous-v0`, semántica `terminated`/`truncated`); **seaborn** (boxplots y bandas de error). Material del curso (`QL.pdf`).

---

# 3. Proyecto MATE — Isolation

## 3.1 El problema y el simulador

*Isolation* es un juego **adversarial, de suma cero, dos jugadores alternados**, sobre un tablero **4×4**. En su turno, cada jugador **mueve su ficha** a una casilla adyacente libre **y destruye** una casilla. **Pierde quien se queda sin movimientos.**

El simulador viene **dado y completo** en `Isolation/` (no se modifica). Mapeo al formalismo del teórico:

| Concepto | Implementación dada |
|---|---|
| Estado inicial | `Board((4,4))` con dos fichas colocadas al azar |
| `Acciones(s)` | `board.get_possible_actions(player)` → `(dirección, celda_a_destruir)` |
| `Suc(s,a)` | `board.clone()` + `board.play(action, player)` |
| `EsFinal(s)` | `board.is_end(player) -> (bool, ganador)` |
| `Utilidad(s)` | +1 / −1 según el ganador |

**Interfaz del agente** (`agent.py`): `next_action(obs)` y `heuristic_utility(board)`.

**Oponentes provistos:** `RandomAgent` (estocástico) y `Stratagem` (ofuscado; deofuscado resulta ser un **Minimax de profundidad 3** — el **baseline fuerte** a vencer).

**Por qué es difícil:** cada acción combina **dirección × celda a destruir**, así que en apertura hay **~100 acciones por jugada**; a profundidad 3 son ~10⁶ nodos. Esto motiva **Alpha-Beta** y **profundidad acotada**.

## 3.2 Marco teórico

Basado en `MiniMax.md` (S. Yovine, ORT) y AIMA cap. 5:
- **Minimax con profundidad limitada** (lám. 13): el agente maximiza, el rival minimiza, y al corte se usa `Eval(s)`.
- **Alpha-Beta** (AIMA 5.3): poda ramas que no afectan la decisión; devuelve **la misma jugada** que Minimax con menos nodos.
- **Expectimax** (lám. 8): si el rival es estocástico, sus nodos son **de azar** (`Σ σ·V`).
- **Buena evaluación** (lám. 16): ordena terminales como la utilidad real, barata, y correlaciona con ganar.

## 3.3 Minimax con Alpha-Beta — `minimax_agent.py`, `search.py`

`search.py` provee el núcleo funcional (`successors`, `is_terminal`, `utility`, `NodeCounter`). Sobre él, `MinimaxAgent` implementa `V_max,min(s,d)` con un método `_minimax` que devuelve `(mejor_acción, valor)`.

**Decisiones y por qué:**
- **Profundidad fija** (no iterative deepening): es el modelo del teórico (lám. 13); en 4×4 una profundidad 3–4 ya da buen juego. Iterative deepening + límite de tiempo agregaría código y riesgo sin estar pedido → queda como mejora opcional. Se priorizó **fidelidad al material**.
- **Alpha-Beta como flag** (`use_alpha_beta`) sobre el mismo núcleo, con **ordenamiento de movimientos** (explora primero las jugadas de mayor/menor evaluación según el nodo): la poda es máxima cuando se ven primero las jugadas prometedoras.

**El código (recursión Alpha-Beta — el mismo núcleo sirve para Minimax puro quitando los `break`):**

```python
def _alphabeta(self, board, player_to_move, depth, alpha, beta):
    done, winner = is_terminal(board, player_to_move)
    if done:      return None, utility(winner, self.player)     # ±1 (perspectiva del agente)
    if depth == 0: return None, self.heuristic_utility(board)   # corte: Eval(s)

    if player_to_move == self.player:                  # nodo MAX (juega el agente)
        best_val, best_action = float("-inf"), None
        for action, child in self._ordered_successors(board, player_to_move):
            _, v = self._alphabeta(child, other(player_to_move), depth-1, alpha, beta)
            if v > best_val: best_val, best_action = v, action
            if best_val > beta: break                  # ← poda β
            alpha = max(alpha, best_val)
        return best_action, best_val
    else:                                              # nodo MIN (juega el rival)
        best_val, best_action = float("inf"), None
        for action, child in self._ordered_successors(board, player_to_move):
            _, v = self._alphabeta(child, other(player_to_move), depth-1, alpha, beta)
            if v < best_val: best_val, best_action = v, action
            if best_val < alpha: break                 # ← poda α
            beta = min(beta, best_val)
        return best_action, best_val
```

Es exactamente `V_max,min(s,d)` del teórico (utilidad terminal / `Eval` al corte / `max` en el agente / `min` en el rival), con la ventana `(alpha, beta)` que poda las ramas que no pueden cambiar la decisión.

**Verificación de corrección (clave para el análisis de impacto).** Sobre **292 estados** (40 seeds × 4 aperturas × profundidades {2,3}):
- **0** diferencias de valor (la poda nunca cambia la decisión),
- **0** diferencias de acción con el ordenamiento desactivado,
- **0** casos con `nodos(AB) > nodos(Minimax)`,
- poda global del **89.8 %**.

> Sutileza documentada: con ordenamiento activo, ante **empates de valor** Alpha-Beta puede elegir otra jugada **igualmente óptima**. Por eso la equivalencia *de acción* se exige solo con el ordenamiento apagado; la equivalencia *de valor* se cumple siempre.

**Experimento E1 — impacto de Alpha-Beta** (`plots/e1_alpha_beta.png`):

| Profundidad | Nodos Minimax | Nodos Alpha-Beta | Reducción |
|---|---|---|---|
| 2 | 436 | 87 | **80 %** |
| 3 | 3 653 | 934 | **74 %** |
| 4 | 13 785 | 1 235 | **91 %** |

![E1 — Nodos y tiempo por jugada: Minimax vs Alpha-Beta según profundidad](Isolation/plots/e1_alpha_beta.png)

La reducción **crece con la profundidad** (a d=4, ~11× menos nodos y ~4× menos tiempo). Alpha-Beta es la palanca que vuelve viable buscar más hondo.

## 3.4 Funciones de evaluación — `evaluation.py`

Cuatro componentes combinables por pesos (`Eval(s) = Σ wᵢ·hᵢ(s)`), todas desde la perspectiva del jugador:

| ID | Heurística | Definición |
|---|---|---|
| h1 | Movilidad propia | casillas adyacentes libres del agente |
| h2 | Diferencia de movilidad | `mov_propia − mov_rival` |
| h3 | Control de centro | `−dist_Manhattan(agente, centro)` |
| h4 | Acorralar | celdas destruidas alrededor del rival − alrededor mío |

**Por qué estas componentes:** en Isolation **perder = quedarse sin movimientos**, así que la **movilidad** es la señal más correlacionada con ganar; **h2** captura la suma cero; **h3** preserva movilidad futura; **h4** ataca la condición de derrota del rival. Las cuatro **replican y generalizan la heurística de `Stratagem`**, dando un punto de comparación honesto.

**El código (una heurística + el combinador):**

```python
def h2_mobility_diff(board, player):       # diferencia de movilidad (captura la suma cero)
    return float(free_adjacent(board, player) - free_adjacent(board, _other(player)))

def weighted_eval(weights):                # devuelve una eval_fn(board, player) = Σ wᵢ·hᵢ(s)
    active = [(HEURISTICS[k], float(w)) for k, w in weights.items() if w != 0]
    def eval_fn(board, player):
        return sum(w * h(board, player) for h, w in active)
    return eval_fn
```

`weighted_eval` se **inyecta** al agente (`MinimaxAgent(..., eval_fn=weighted_eval(pesos))`), lo que permite comparar ponderaciones sin tocar el código de búsqueda.

**Decisión de medida:** la movilidad se cuenta como **casillas adyacentes libres** (estilo `has_valid_moves`), **no** `len(get_possible_actions)`, que infla el valor al multiplicar por cada celda destruible.

## 3.5 Expectimax — `expectimax_agent.py`

Igual estructura que Minimax, pero los nodos del rival son **de azar** con `σ` **uniforme**:

```python
prob = 1.0 / len(succ)             # σ uniforme sobre las acciones legales
expected = sum(prob * V(child) for _, child in succ)
```

**Sin Alpha-Beta**: los nodos de azar **promedian** todas las ramas (no hay corte por cota), así que la poda no aplica.

**Hipótesis a confirmar (no asumir):** Minimax supone rival óptimo (cota inferior garantizada); Expectimax supone rival uniforme. → vs Random (estocástico) ambos bien; vs Stratagem (determinista) el modelo uniforme es **incorrecto**, se espera **Minimax ≥ Expectimax**.

## 3.6 Metodología experimental — `match.py` + `isolation.ipynb`

**Diseño apareado (decisión de rigor).** Como existe una **ventaja de primer jugador** fuerte y la colocación es aleatoria, **cada seed se juega dos veces** (nuestro agente de jugador 1 y de jugador 2) **sobre la misma posición**. Así la comparación no depende de quién arrancó ni de qué posiciones tocaron. Es más riguroso que solo alternar lados con seeds distintos (el enfoque inicial).

**Métricas:** win rate; **nodos y tiempo por jugada de NUESTRO agente** (`a_nodes_per_move`, `a_avg_move_time`) — medidos por separado de cada jugador, para que el costo no quede contaminado por el del rival (p. ej. el lento Minimax d=3 de Stratagem).

**Parámetros finales:** `N_RANDOM=100`, `N_SELF=100`, `N_STRAT=40`, `N_HEUR=30` **seeds** (cada seed = 2 partidas). Registro completo en `results.csv` (**1568 filas**); corrida ≈ 15 min.

> **Por qué los pesos base `{1,2,0.5,1}` en E1–E5 no sesgan la decisión:** son una ponderación neutra elegida *antes* de conocer el torneo. La comparación de técnicas y el análisis de Alpha-Beta son **robustos** a esa elección (ambos agentes comparten la misma `eval_fn`). E6 explora **por separado** cuál ponderación es la mejor.

## 3.7 Resultados E1–E6 y la decisión técnica

**E2 — vs Random:** Minimax **96 %**, Expectimax **94.5 %**. Ambos dominan al azar.

**E3 — vs Stratagem, por profundidad** (`plots/e2_e3_winrate.png`):

| Técnica | d=2 | d=3 (parejo con Stratagem) |
|---|---|---|
| Minimax | 39 % | **46 %** |
| Expectimax | 48 % | 34 % |

![E2 y E3 — Win rate vs Random (izq.) y vs Stratagem por técnica y profundidad (der.)](Isolation/plots/e2_e3_winrate.png)

**E4 — directo (d=2):** Minimax 44 % / Expectimax 56 %.

**Costo por agente a d=3:** Expectimax cuesta **0.59 s y ~24 100 nodos/jugada** vs **0.13 s y ~1 300** de Minimax (~4.5× más lento, ~18× más nodos: **no poda**).

**E5 — profundidad:** Minimax vs Stratagem **27 % → 39 % → 46 %** (d=1,2,3): buscar más hondo ayuda, monótono.

![E5 — Win rate de Minimax vs Stratagem según profundidad](Isolation/plots/e5_depth.png)

**E6 — torneo de heurísticas** (`plots/e6_heatmap.png`):

| Ponderación | Win rate prom. |
|---|---|
| **`solo_mov_diff`** (solo h2) | **0.700** |
| `mov+centro` | 0.578 |
| `balanceada` | 0.483 |
| `mov+acorralar` | 0.239 |

![E6 — Heatmap del torneo de heurísticas (win rate fila vs columna)](Isolation/plots/e6_heatmap.png)

**La diferencia de movilidad sola (h2) es la mejor**: agregarle otras componentes la **diluye**.

**Conclusión — "¿cuál técnica es mejor?": depende del oponente y de la profundidad.**
- vs rival **estocástico** (Random): ambas dominan.
- vs rival **determinista** (Stratagem), **a profundidad igualada (d=3) gana Minimax** (46 % vs 34 %) y además es mucho más barato. Profundizar **mejora** a Minimax y **empeora** a Expectimax (propagar más hondo un modelo de rival uniforme, incorrecto para Stratagem, degrada el juego). **Confirma la predicción teórica.**

## 3.8 Modelo computado — `mate_best_config.pkl`

Minimax/Expectimax **no entrenan** un modelo, pero la experimentación **computa** la mejor configuración. Se serializa un dict (verificado abriendo el `.pkl`):

```python
{'tecnica': 'minimax', 'profundidad': 3, 'pesos': {'h2': 1.0},
 'metricas': {'win_rate_vs_stratagem_dmax': 0.463, 'win_rate_vs_random': 0.96, 'e6_mejor_winrate_promedio': 0.7}}
```

**Por qué entregamos `.pkl`:** la *Auditoría* pide, **en general**, "modelos computados (.pkl o similares)"; la penalización estricta solo nombra LOST. Ante la ambigüedad y por costo mínimo, se entrega igual; serializar la mejor configuración hace **reproducible** el agente ganador. Se descartó precalcular una tabla de toda la posición 4×4 **por sobrecomplicación**.

## 3.9 El camino recorrido: errores y cómo se resolvieron

- **Entorno Poetry roto (en una de las máquinas):** `poetry` fallaba (`packaging` viejo) y `poetry install` "a secas" no instalaba (proyecto sin paquete). **Solución:** actualizar `packaging`/`platformdirs` en el intérprete de Poetry, e instalar con `poetry install --no-root`. Documentado para el equipo.
- **Pasada rápida y un hallazgo sorpresa (resuelto):** una primera corrida con N chico y nuestros agentes a d=2 (contra el d=3 de Stratagem) mostró a **Expectimax superando a Minimax** vs Stratagem, lo que **contradecía la hipótesis**. En vez de forzar la teoría, se midió **a profundidad igualada (d=3)** y se vio que era un **artefacto de profundidad**: a d=3 Minimax gana. *Las conclusiones se ajustaron a la evidencia, no al revés.*
- **Bug al reconstruir el notebook:** un script de edición matcheaba el texto `"BASE_SEED"`, que también aparecía en la celda de funciones auxiliares, y la pisó (dejó `make_board` indefinido). **Solución:** restaurar desde Git y reconstruir el notebook con un único script que **solo agrega** celdas. Lección: editar celdas de notebook por contenido es frágil.
- **Pesos/duplicación y métricas:** se agregaron `pandas` y `matplotlib` al entorno de MATE (necesarios para el CSV y los gráficos que pide la consigna) vía `poetry add`, ajustando versiones para Python 3.10.
- **Mejoras de rigor aplicadas (segunda corrida):** se reemplazó el "alternar lados" por **diseño apareado**, y se agregó **medición de tiempo por agente** (`match.py` devuelve el tiempo de cada jugador). Las conclusiones **no cambiaron**, pero quedaron más rigurosas y se sumó el argumento de **costo** a favor de Minimax.

**Notas de advertencia:**
- **Bug del oponente `Stratagem` como jugador 2 (no es nuestro código):** su Minimax interno evalúa **su propia derrota como 0 en vez de −1** cuando juega de jugador 2 (verificado con un test directo). Lo hace algo **más débil de segundo**; no se corrige (no se tocan archivos dados), pero el **diseño apareado balancea** el efecto.
- **Ventaja de primer jugador (medida, controlada):** nuestro agente gana **48 % arrancando** vs **35 % de segundo**; el diseño apareado lo neutraliza.
- **Costo de `get_possible_actions`:** el simulador clona el tablero por cada dirección; en búsqueda profunda este costo domina. Limitación del simulador dado, mitigada con Alpha-Beta y profundidad acotada.

## 3.10 Conclusiones de MATE

- **Alpha-Beta** poda hasta el **91 %** de los nodos (a d=4) **sin cambiar la decisión** (verificado sobre 292 estados): es el análisis de impacto pedido.
- **Minimax es la mejor técnica** contra el rival determinista a profundidad igualada (46 % vs 34 %), y además **mucho más barato** que Expectimax (que no poda). La respuesta a "cuál conviene" es **"depende del oponente y la profundidad"**, con evidencia.
- La **diferencia de movilidad sola (h2)** es la mejor función de evaluación.
- El framework experimental (diseño apareado, métricas por agente, registro a CSV) hace las comparaciones **justas y reproducibles**.

---

# 4. Tecnologías y librerías

| Tecnología | Dónde | Para qué |
|---|---|---|
| **Python ~3.10** | ambos | Lenguaje base |
| **Poetry** | ambos (entornos **separados**) | Gestión de dependencias/entornos |
| **Gymnasium** | LOST (dep. en MATE) | Ambiente `MountainCarContinuous-v0`; API `reset/step`, `terminated`/`truncated` |
| **NumPy** | ambos | Tabla Q, discretización (`linspace`, `digitize`), grilla del tablero |
| **Matplotlib** | ambos | Gráficos (curvas, política, win rates, heatmaps) |
| **Pygame** | LOST | Render del ambiente |
| **pandas** | MATE | Tablas y `results.csv` de experimentos |
| **tabulate** | MATE | Render del tablero en consola |
| **pickle** | ambos | Modelos `.pkl` (Q + config en LOST; dict de config en MATE) |
| **Jupyter / ipykernel / notebook** | ambos | Notebooks `.ipynb` |
| **JSON / CSV** | LOST / MATE | Registro de resultados |

> Nota: LOST usa la convención `[tool.poetry]` (Poetry clásico) y MATE `[project]` (PEP 621) — coherente con el reparto del trabajo entre los integrantes.

---

# 5. Entregables y estado

**LOST (`MountainCarContinuous/`):** código `.py` (discretizer, agentes, `experiments.py`, scripts), notebook `continuous_mountain_car.ipynb`, modelos en `models/` (`q_learning_best.pkl`, `dyna_q_best.pkl`, `smoke_test.pkl` — todos con **recompensa real**), resultados `grid_search_results.json` + `dyna_q_comparison.json` y **9 gráficos** en `plots/` (boxplots y bandas de error con seaborn).

**MATE (`Isolation/`):** código `.py` (`search`, `minimax_agent`, `expectimax_agent`, `evaluation`, `match`), notebook `isolation.ipynb` (32 celdas, corre end-to-end), `results.csv` (1568 filas), 4 gráficos en `plots/`, y `mate_best_config.pkl`.

**Pendiente / a revisar antes del `.zip`:**
- **Informe en PDF:** esta documentación está en markdown; falta exportarla a **PDF** (formato que pide la consigna).
- **Duplicación de modelos de LOST:** los `.pkl` aparecen en varias carpetas (`models/` raíz, `MountainCarContinuous/models/`, `resultados_lost/`, `resultados_lost_tmp/`) con nombres distintos. La fuente canónica es `MountainCarContinuous/models/` (donde escriben los scripts); conviene **dejar un solo set** en el `.zip`.
- **`mate_best_config.pkl` está en `.gitignore`** (`*.pkl`): existe en disco; recordar **incluirlo en el `.zip`**.

---

# 6. Uso de IA generativa

Conforme exige la consigna, se declara el uso de IA generativa (Claude, de Anthropic, vía Claude Code):
- **Redacción** de la documentación a partir de la planificación y los PDFs de teoría.
- **Generación de código** de LOST (`Discretizer`, `QLearningAgent`, `DynaQAgent`) y de MATE (`MinimaxAgent` con Alpha-Beta, `ExpectimaxAgent`, `evaluation.py`, `match.py` y el framework de experimentos), consumiendo la API del simulador provisto sin modificarlo.
- **Análisis y discusión** de resultados (grid search de LOST; experimentos E1–E6 de MATE).

Todo el contenido producido por la IA fue **revisado, ejecutado y verificado** por el equipo antes de incorporarlo. Los errores que pueda haber son responsabilidad del equipo.

---

# Anexo A — Cumplimiento de la consigna (checklist)

Cada punto de la consigna, con su estado y **dónde se demuestra** en este documento.

### Proyecto LOST (mapeo a la devolución de la cátedra)
| Requisito | Estado | Dónde se demuestra |
|---|---|---|
| **Q-Learning** con la **recompensa real** (sin shaping) | ✅ | §2.4 (regla + código) + §2.5 (inicialización optimista) |
| **Discretización** obligatoria + explorar **diversas** discretizaciones | ✅ | §2.3 (decisiones) + §2.7 (gruesa/media/fina con su varianza) |
| Explorar **diversos α, γ, epsilon** + **epsilon decay** | ✅ | §2.4 (decay) + §2.7 (grid OAT) |
| **Múltiples seeds** por config + **análisis de varianza** | ✅ | §2.6 (metodología) + §2.7 (mín/std por config; `bins20` inestable) |
| **Boxplots / bandas de error** (seaborn) | ✅ | §2.7 y §2.8 (gráficos seaborn) |
| **Dyna-Q** + experimentación + **comparación con Q-Learning** | ✅ | §2.8 (implementación + comparación con varianza) |
| Reward shaping **solo como extra** (no en el núcleo) | ✅ | §2.10 |
| **Conclusiones personales** + **fuentes referenciadas** | ✅ | §2.11 |
| **Al menos un modelo computado** (recompensa real) | ✅ | `models/q_learning_best.pkl` + `dyna_q_best.pkl` |

### Proyecto MATE
| Requisito de la consigna | Estado | Dónde se demuestra |
|---|---|---|
| **Minimax con Alpha-Beta** + **análisis de su impacto** | ✅ | §3.3 (código + verificación de equivalencia sobre 292 estados + experimento E1) |
| **Expectimax** + decidir cuál técnica es mejor | ✅ | §3.5 + §3.7 (E2–E4: conclusión con evidencia) |
| **Funciones de evaluación** + combinaciones y **ponderaciones** | ✅ | §3.4 (h1–h4 + `weighted_eval`) + §3.7 (torneo E6) |
| **Definir pruebas + registro completo** de resultados | ✅ | §3.6 (metodología) + `Isolation/results.csv` (1568 filas) |

### Contenido del informe y entrega
| Requisito | Estado | Dónde |
|---|---|---|
| Resumen del abordaje: **interacción con el simulador, parámetros, tiempo de ejecución, resultados** | ✅ | §2.4/§3.6 (interacción y parámetros), §2.7–§2.9/§3.7 (tiempos y resultados), Anexo B (tiempos consolidados) |
| **Apoyo visual** (gráficos claros + comentarios) | ✅ | 11 gráficos incrustados en §2 y §3 |
| **Notas de advertencia** (dificultades y por qué) | ✅ | §2.1 (la "trampa de no hacer nada") + §2.7 (configs inestables) y §3.9 (notas de MATE) |
| Código `.py` + `.ipynb` | ✅ | ambos proyectos |
| Modelos computados (`.pkl`) | ✅ | LOST `models/` + MATE `mate_best_config.pkl` |
| Entornos **Poetry separados** | ✅ | `MountainCarContinuous/pyproject.toml` y `Isolation/pyproject.toml` |

---

# Anexo B — Reproducibilidad y verificación (comandos y salidas reales)

Todo es reproducible con semillas fijas. Acá están los comandos exactos y las **salidas reales** que **demuestran** las afirmaciones del informe, sin necesidad de leer el código.

## B.1 Preparación del entorno (una vez por proyecto)

```bash
# En cada carpeta (MountainCarContinuous/ y Isolation/):
poetry install --no-root        # instala dependencias (el proyecto es de archivos planos, sin paquete)
```

> Si Poetry falla con `No module named 'packaging.licenses'`, actualizar en el intérprete de Poetry:
> `<python-de-poetry> -m pip install --upgrade "packaging>=24.2" "platformdirs>=4.3.6"`.

## B.2 LOST — comandos (recompensa real, sin shaping)

```bash
cd MountainCarContinuous
poetry install --no-root               # incluye seaborn + pandas
poetry run python smoke_test.py        # valida la pipeline (500 ep, sin shaping): 100% test
poetry run python grid_search.py       # grid OAT × 5 seeds → grid_search_results.json + boxplots/bandas
poetry run python compare_dyna_q.py    # Dyna-Q vs Q-Learning × 5 seeds → dyna_q_comparison.json + boxplots
poetry run python train_best.py        # entrena y guarda los 2 modelos finales (q_learning_best, dyna_q_best)
poetry run python visualize_policy.py  # mapas de V(s) y π(s) (pump-and-go)
```

**Reproducibilidad y varianza:** el grid y la comparación corren cada config con **5 semillas** (`experiments.SEEDS = [0..4]`); cada corrida siembra el agente (`random`+`numpy`) y el env de forma independiente. El **test** usa un conjunto **fijo** de episodios sembrados (`TEST_SEEDS`), igual para todas las configs. Los modelos guardados fueron abiertos y verificados: `q_learning_best.pkl` (`40×40`, `5` acciones, `α=0.3`, `opt-init=10`, `reward_shaping=False`, `Q.shape=(41,41,5)`, cobertura ~60 %) y `dyna_q_best.pkl` (`planning_steps=5`, `reward_shaping=False`).

## B.3 MATE — comandos y salidas reales

Cada archivo `.py` trae un *smoke test* con aserciones. **Salida real:**

```text
$ poetry run python search.py
OK Paso 1

$ poetry run python evaluation.py        # heurísticas verificadas con tableros de valor conocido
OK Paso 4

$ poetry run python minimax_agent.py
[Paso 2] MinimaxAgent(d=2) vs RandomAgent: 29/30 = 97% win rate
[Paso 3] Equivalencia OK en 7 estados (d=3). Nodos Minimax=37988, Alpha-Beta=6554 -> poda 83%
OK Paso 3: Alpha-Beta da el mismo valor que Minimax expandiendo menos nodos

$ poetry run python expectimax_agent.py
[Paso 5] ExpectimaxAgent(d=2) vs RandomAgent: 29/30 = 97% win rate
[Paso 5] Expectimax vs Stratagem: corrio sin errores
OK Paso 5: Expectimax vence a Random y corre frente a Stratagem

$ poetry run python match.py             # misma seed → mismo resultado (reproducible)
OK Paso 0: partidas reproducibles con misma seed

# El notebook completo (experimentos E1–E6 + gráficos + .pkl) corre de punta a punta:
$ poetry run jupyter nbconvert --to notebook --execute --inplace isolation.ipynb   # 0 errores, ~15 min
```

## B.4 Demostración: Alpha-Beta es correcto (no solo más rápido)

Comparación exhaustiva Minimax vs Alpha-Beta sobre **292 estados** (40 semillas × 4 aperturas × profundidades {2,3}). **Salida real:**

```text
Estados probados: 292
Diferencias de valor (AB vs Minimax): 0          ← la poda NUNCA cambia la decisión
Diferencias de accion (AB sin orden vs Minimax): 0
Casos nodos(AB) > nodos(Minimax): 0              ← AB nunca expande más que Minimax
Nodos: Minimax=1637094, Alpha-Beta=166193, reduccion=89.8%
```

Esto es el **análisis de impacto** pedido por la consigna: misma jugada, ~90 % menos nodos.

## B.5 Demostración: el bug del oponente `Stratagem` (no es nuestro código)

`Stratagem` (oponente dado) evalúa mal su propia derrota cuando juega de jugador 2. *Probe directo sobre su función interna deofuscada,* **salida real:**

```text
Stratagem(1) en su derrota: -1   (correcto: -1)
Stratagem(2) en su derrota:  0   (debería ser -1, da 0 = BUG)
```

Por eso `Stratagem` es algo más débil de jugador 2. No se corrige (no se tocan archivos dados); el **diseño apareado** (§3.6) balancea el efecto entre las variantes comparadas.

## B.6 Tiempos de ejecución (consolidado)

| Tarea | Tiempo aprox. |
|---|---|
| LOST — smoke test (500 ep) | ~5 s |
| LOST — entrenamiento de un modelo final (2500 ep) | **~13 s** (Q-Learning) / ~33 s (Dyna-Q n=5) |
| LOST — grid search (11 configs × 5 seeds × 1500 ep) | **~15 min** |
| LOST — comparación Dyna-Q (3 variantes × 5 seeds × 1000 ep) | ~8 min |
| MATE — un `next_action` de Minimax a d=4 (Alpha-Beta) | ~0.08 s |
| MATE — corrida completa de experimentos E1–E6 | **~15 min** (dominada por los ~560 partidos vs Stratagem) |
