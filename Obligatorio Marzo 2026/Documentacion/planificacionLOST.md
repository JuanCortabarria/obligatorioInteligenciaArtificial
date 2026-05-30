# Plan — Proyecto LOST (Obligatorio IA, Marzo 2026)

> **Nota de estado:** este archivo es una planificacion historica inicial. No
> describe completamente el estado actual del codigo ni de los artefactos.
> Para el detalle final usar `LOST_decisiones_tecnicas_justificacion.md`,
> `LOST_experimentos_resultados.md`, `LOST_estado_y_plan.md` y `mejoras.md`.

## Contexto

El obligatorio pide implementar un agente que aprenda a conducir el rover en el ambiente **MountainCarContinuous-v0** (Gymnasium) usando aprendizaje por refuerzo. La consigna exige:

1. **Discretizar** observaciones (posición, velocidad) y acciones (fuerza continua en `[-1, 1]`), justificando la elección.
2. Aplicar **Q-Learning** (off-policy TD control — visto en `QL.pdf`, slides 5–6).
3. **Exploración de hiperparámetros** (α, γ, ε, decay, episodios, bins) con una métrica clara de evaluación.
4. **Componente de investigación**: leer Sutton & Barto, cap. 8.1–8.2, e implementar **Dyna-Q** (planning con modelo aprendido), analizando y comparando contra Q-Learning tabular.

**Grupo de 2** → no hay tarea adicional (Stochastic Q-Learning / MCTS quedan fuera).

**Desafío central del ambiente**: el reward es **sparso** — `−0.1·a²` por step y `+100` solo al alcanzar `x ≥ 0.45`. Con política aleatoria, el carro **casi nunca** llega a la meta, por lo que Q permanece en cero y no se aprende nada. Esto motiva tanto la **discretización agresiva**, como la **exploración prolongada**, la **inicialización optimista** y, sobre todo, **reward shaping** comparativo.

Entrega: `.zip` con código (`.py` + `.ipynb`) + al menos un **modelo `.pkl`** entrenado (obligatorio: sin esto el ejercicio se considera no hecho) + informe ≤ 20 págs.

---

## Estructura de archivos

Se mantiene el scaffold provisto y se agregan dos archivos:

```
MountainCarContinuous/
├── q_learning_agent.py          ← clase QLearningAgent (a completar)
├── dyna_q_agent.py              ← NUEVO: clase DynaQAgent
├── discretization.py            ← NUEVO: utilidades de bins + reward shaping
├── continuous_mountain_car.ipynb ← entrenamiento, evaluación, gráficos
├── models/                      ← NUEVO: .pkl entrenados
│   ├── q_learning_best.pkl
│   └── dyna_q_best.pkl
└── pyproject.toml
```

---

## Diseño teórico (anclado en los PDFs)

### MDP del ambiente

Mapeando a la definición de `Introducción a MDP.pdf` (slide 7):

- **Estados**: `(x, v)` con `x ∈ [−1.2, 0.6]`, `v ∈ [−0.07, 0.07]` → tras discretizar, `S = bins_x × bins_v`.
- **Acción**: fuerza `a ∈ [−1, 1]` → tras discretizar, `A = {a₁, …, aₖ}`.
- **P(s′ | s, a)**: dinámica del carro (determinista en este env, pero el agente la "desconoce" → la aprende en Dyna-Q).
- **R(s, a, s′)**: `−0.1·a²` o `+100` si `x′ ≥ 0.45`.
- **done(s)**: `x ≥ 0.45` o se llega al tope de steps (999).

### Q-Learning (de `QL.pdf`, slide 6)

Actualización off-policy TD:

```
Q(s, a) ← Q(s, a) + α · [ r + γ·max_a′ Q(s′, a′) − Q(s, a) ]
```

con política ε-greedy para selección. Justificación de **off-policy**: permite explorar agresivamente (necesario por el reward sparso) mientras se aprende la política greedy óptima.

### Dyna-Q (Sutton & Barto, cap. 8.1–8.2)

Q-Learning + modelo del ambiente aprendido + **n pasos de planning** por step real:

```
Loop por episodio:
  s ← estado actual
  a ← ε-greedy(Q, s)
  s′, r ← env.step(a)              # experiencia real
  Q[s,a] ← Q[s,a] + α·(r + γ·max Q[s′,·] − Q[s,a])
  Modelo[s,a] ← (r, s′)             # update determinista del modelo
  Repetir n veces:                  # planning
    s_p, a_p ← muestrear de pares ya observados
    r_p, s′_p ← Modelo[s_p, a_p]
    Q[s_p,a_p] ← Q[s_p,a_p] + α·(r_p + γ·max Q[s′_p,·] − Q[s_p,a_p])
```

**Hipótesis para el informe**: Dyna-Q debería converger en menos *episodios reales* que Q-Learning puro porque amortiza cada transición real con `n` updates simulados. El costo: tiempo por episodio sube ~lineal en `n`.

---

## Pasos de implementación

### Paso 1 — Discretización (`discretization.py`)

Exponer una clase `Discretizer` parametrizada por `n_bins_x`, `n_bins_v`, `n_actions`:

- `x_space = np.linspace(-1.2, 0.6, n_bins_x)`
- `v_space = np.linspace(-0.07, 0.07, n_bins_v)`
- `actions = np.linspace(-1, 1, n_actions)`
- `get_state(obs) → (xi, vi)` usando `np.digitize` (ya esbozado en la notebook, celda 10).
- `get_action_idx(a) / action_from_idx(i)`.

**Configuraciones a comparar en el informe**:

| Config | bins_x | bins_v | n_actions | tamaño Q |
|--------|--------|--------|-----------|----------|
| Gruesa | 20     | 20     | 3         | 1.2 K    |
| Media  | 40     | 40     | 5         | 8 K      |
| Fina   | 100    | 100    | 10        | 100 K    |

**Justificación** que se pide en la consigna: trade-off granularidad vs. tabla manejable vs. velocidad de aprendizaje (más bins → tabla más sparsa → más exploración necesaria).

### Paso 2 — `QLearningAgent` (`q_learning_agent.py`)

Completar la clase ya scaffoldeada. Firma:

```python
class QLearningAgent:
    def __init__(self, discretizer, alpha=0.1, gamma=0.99,
                 epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 optimistic_init=0.0, reward_shaping=False)
    def next_action(self, obs)                # greedy puro (para test)
    def _epsilon_greedy(self, state)
    def train_agent(self, env, episodes)      # devuelve historia de rewards/steps
    def test_agent(self, env, episodes=10)    # sin exploración
    def save(self, path)  /  load(path)       # .pkl
```

Notas de implementación:

- **Inicialización optimista**: `Q = np.full((bins_x+1, bins_v+1, n_actions), optimistic_init)`. Con `optimistic_init > 0` se favorece explorar acciones no probadas (truco clásico de Sutton & Barto cap. 2.6).
- **ε-decay exponencial** por episodio: `epsilon = max(epsilon_min, epsilon_start · decay^ep)`.
- **Reward shaping** (flag opcional, activable desde `__init__`): sumar al reward base una bonificación `c · |v|` o `c · (x − x_inicial)` por step. Esto da señal densa para que el agente descubra que ganar velocidad es bueno → cumple con "*aprenda que avanzar suele ser mejor que no hacerlo*".

### Paso 3 — `DynaQAgent` (`dyna_q_agent.py`)

Hereda de `QLearningAgent` o duplica con un atributo extra `planning_steps n` y un `model: dict[(s,a) → (r, s′)]`. Override de `train_agent` con el loop interno de planning descrito arriba.

### Paso 4 — Notebook: entrenamiento, búsqueda y evaluación

Estructura sugerida del notebook:

1. **Setup**: imports, `env = gym.make('MountainCarContinuous-v0', render_mode='rgb_array')` (importante: `rgb_array` para entrenar rápido, `human` solo para demo final).
2. **Smoke test** de `Discretizer` y `QLearningAgent` con pocos episodios.
3. **Grid search** sobre Q-Learning. Combinaciones acotadas (no producto cartesiano completo — sería intratable). Sugerencia: ~12–20 corridas eligiendo:
   - `(bins_x, bins_v, n_actions)` ∈ {gruesa, media, fina}
   - `α` ∈ {0.05, 0.1, 0.3}
   - `γ` ∈ {0.95, 0.99}
   - `epsilon_decay` ∈ {0.99, 0.995, 0.999}
   - `reward_shaping` ∈ {False, True}
4. **Métricas de evaluación** (definirlas antes de mirar resultados, como pide la consigna):
   - **Tasa de éxito**: % de los últimos 100 episodios de train donde llegó a meta.
   - **Reward promedio** últimos 100 episodios.
   - **Steps-to-goal promedio** en episodios exitosos.
   - **Curva de aprendizaje** (reward por episodio, media móvil 50).
5. **Mejor config Q-Learning** → entrenar más largo (5k–20k episodios), guardar `.pkl`.
6. **Dyna-Q**: barrer `n ∈ {0, 5, 25, 50}` con la mejor config de hiperparámetros del paso anterior. `n=0` ≡ Q-Learning (sanity check).
7. **Comparación final**: gráfico con eje X = episodios reales, eje Y = reward, una curva por `n` planning steps. Comentar la hipótesis de Sutton & Barto cap. 8.2 (Dyna-Q acelera la convergencia en términos de experiencia real).
8. **Demo visual**: cargar mejor modelo, correr 1–2 episodios con `render_mode='human'`.

### Paso 5 — Persistencia (.pkl)

`pickle.dump({'Q': self.Q, 'discretizer_config': {...}, 'hyperparams': {...}}, f)`. **Obligatorio** entregar al menos uno (la consigna lo marca explícitamente).

### Paso 6 — Informe

Secciones mínimas (≤ 20 págs):

1. Introducción y modelado MDP.
2. Discretización: tabla comparativa y justificación.
3. Q-Learning: ecuación, pseudocódigo, decisiones (optimistic init, ε-decay, shaping).
4. Resultados Q-Learning: gráficos de las corridas del grid search, elección de hiperparámetros.
5. Dyna-Q: explicación del cap. 8.1–8.2, pseudocódigo, resultados comparativos.
6. Análisis comparativo Q-Learning vs Dyna-Q.
7. Conclusiones y dificultades encontradas.
8. **Citación obligatoria** de uso de IA generativa (la consigna lo exige explícitamente, p. 1).

---

## Archivos críticos a tocar

- [MountainCarContinuous/q_learning_agent.py](MountainCarContinuous/q_learning_agent.py) — completar clase (hoy solo tiene `pass`).
- [MountainCarContinuous/continuous_mountain_car.ipynb](MountainCarContinuous/continuous_mountain_car.ipynb) — completar el update de Q en la celda 22 (`# Q[state][action_idx] = ... # Completar`) y agregar celdas de grid search + plots.
- **NUEVOS**: `discretization.py`, `dyna_q_agent.py`, carpeta `models/`.

---

## Verificación end-to-end

1. `cd MountainCarContinuous && poetry install`
2. `poetry run jupyter notebook continuous_mountain_car.ipynb` — correr todas las celdas; smoke test debe completar sin error y devolver `total_reward` finito.
3. Correr un entrenamiento corto (500 episodios) con config "media" + `reward_shaping=True`: la tasa de éxito en los últimos 100 episodios debería ser **> 0** (señal de que la pipeline aprende algo). Si es 0, revisar shaping o subir epsilon.
4. Correr entrenamiento largo de la mejor config; guardar `.pkl`.
5. Cargar `.pkl` en una notebook fresca y correr `test_agent(env, episodes=10)` con `render_mode='human'` — confirmar visualmente que el carro llega a la bandera.
6. Repetir 4–5 para Dyna-Q.
7. Verificar tamaño total del `.zip` < 40 MB (límite de Gestión).
