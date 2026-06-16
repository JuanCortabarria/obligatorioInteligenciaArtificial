# Planificación LOST v2 — Rework según devolución del profesor

> Plan **aprobado para ejecutar** del rework del proyecto LOST (Mountain Car Continuous),
> a partir de la devolución del profesor. Reemplaza el enfoque anterior (centrado en reward
> shaping) por uno con **recompensa real** y **análisis de varianza**. MATE no se toca.

---

## 0. Punto de partida: la letra inicial (scaffold provisto)

> **Por qué esta sección.** En la planificación inicial saltamos a codear **sin dejar
> registrado el punto de partida** (la interfaz y el entorno que dio la cátedra). Eso fue
> una de las fuentes de problemas: terminamos **cambiando la interfaz provista** y el
> entorno sin documentarlo, y construyendo alrededor del reward shaping. v2 arranca
> registrando exactamente qué se nos dio.

### 0.1 El código que entregó la cátedra (`q_learning_agent.py` original)

```python
class QLearningAgent:
    def __init__(self):
        pass
    def next_action(self, obs):
        pass
    def train_agent(self, env, episodes=1000, epsilon=.9, gamma=.9, alpha=.99):
        pass
    def test_agent(self, env, episodes=10):
        pass
```

**Contrato implícito (la interfaz que esperaría la cátedra):** un agente con
- `__init__(self)` — **sin parámetros**;
- `next_action(self, obs)` — elegir acción dada una observación;
- `train_agent(self, env, episodes, epsilon, gamma, alpha)` — entrenar, con
  **`epsilon`, `gamma`, `alpha` como argumentos del método** (defaults sugeridos:
  `epsilon=0.9`, `gamma=0.9`, `alpha=0.99`);
- `test_agent(self, env, episodes)` — evaluar.

La discretización **no** estaba dada: el scaffold deja la observación/acción continuas como
vienen del env; discretizar es responsabilidad nuestra (lo pide la consigna).

### 0.2 El entorno original (`pyproject.toml` provisto)

`[tool.poetry]` (formato clásico), Python `~3.10`, dependencias: **numpy, gymnasium,
matplotlib, pygame**; dev: **notebook, black**. **No** incluía `pandas` ni `seaborn`.

### 0.3 Desviaciones que hicimos respecto del scaffold (y su justificación)

| Elemento del scaffold | Lo que hicimos | Por qué |
|---|---|---|
| `__init__(self)` | `__init__(self, discretizer, alpha, gamma, epsilon_start, epsilon_min, epsilon_decay, optimistic_init, reward_shaping, seed)` | La **configuración vive en el objeto** → se puede `save`/`load` y reusar; **necesario para el grid search** (instanciar muchas configs) y para fijar seeds. |
| `train_agent(env, episodes, epsilon, gamma, alpha)` | `train_agent(env, episodes, max_steps, verbose_every, env_seed, reset_epsilon)` | Los hiperparámetros pasaron a `__init__`; se agregó **siembra/reproducibilidad** (`env_seed`, `reset_epsilon`) y `max_steps` como red de seguridad. |
| `next_action(self, obs)` | **Mantenida** (mismo nombre y semántica: acción greedy) | Es la interfaz mínima de uso del agente. |
| `test_agent(self, env, episodes)` | **Extendida**: `test_agent(env, episodes, max_steps, render, test_seeds)` | `test_seeds` da una **evaluación fija y comparable** entre configs. |
| (no estaba) | `Discretizer`, persistencia `save/load`, subclase `DynaQAgent`, `experiments.py` | Necesarios para la consigna (discretización, modelo computado, Dyna-Q, varianza). |
| deps originales | **+pandas, +seaborn** | Para tablas y los **boxplots/bandas de error** que pide el profesor. |

### 0.4 Decisión — conformidad de interfaz

Nuestro agente **no conserva la firma exacta** de `__init__`/`train_agent` del scaffold: si
la cátedra evaluara haciendo `QLearningAgent()` + `train_agent(env, episodes, epsilon, gamma,
alpha)`, **no sería un reemplazo directo** (nuestro `__init__` exige `discretizer`).

**Decisión (equipo): mantener nuestra interfaz**, documentada como **desviación justificada**
(la configuración vive en el objeto para soportar `save`/`load` y el grid search). Razones:
(1) la devolución del profesor fue **toda de metodología** y **nunca** mencionó la interfaz;
(2) la evaluación se hace leyendo el informe + corriendo **nuestro** notebook/scripts, no
enchufando el agente a un test oculto; (3) `next_action(self, obs)` —la interfaz mínima de
uso— **sí se mantiene** compatible. Si en el futuro hiciera falta, alcanzaría con un *shim*
delgado que acepte la firma original; no se implementa ahora para no agregar complejidad.

---

## 1. Qué pidió el profesor (textual → requisitos)

1. **Q-Learning obligatorio** y **discretización obligatoria**.
2. La experimentación debe contener:
   - exploración de **diversos α, γ, epsilon** y **discretizaciones**;
   - **epsilon con política de decay**;
   - **varios experimentos con la misma configuración y distintas seeds**, para **analizar la varianza**;
   - **visualizaciones con bandas de error o boxplots** (sugiere `seaborn`).
3. **No es aceptable cambiar las recompensas** (reward shaping = cambiar el ambiente). El shaping es un **extra**, pero **no se puede dejar de hacer lo pedido** con la recompensa real.
4. **Dyna-Q obligatorio**: implementarlo, **experimentarlo y compararlo con Q-Learning**.
5. Todo **bien documentado**, **fuentes bien referenciadas**, **conclusiones personales**.

## 2. Decisiones tomadas (acordadas con el equipo)

| Decisión | Elección |
|---|---|
| Reward shaping | **Demotado a "extra"**, mención mínima y aparte (sin experimento comparativo dedicado). El núcleo usa la **recompensa real**. |
| Presupuesto de cómputo | **Ligero**: **5 seeds**, ~**1500 episodios** en el grid (y ~2000–2500 para la corrida final de las mejores), grid acotado. |
| Si Q-Learning puro no resuelve sin shaping | **Conclusión honesta**: reportarlo (con boxplots de alta varianza) y mostrar que **Dyna-Q** es el que resuelve. |
| Documentación a actualizar | **Ambos**: `DocumentacionFinal.md` y `Documentacion.md`. |

## 3. Análisis técnico (por qué este enfoque es el correcto)

**El problema real sin shaping.** La recompensa es `−0.1·a²` por paso (casi cero) y `+100` solo en la meta. Esto crea una **trampa de "no hacer nada"**: como moverse cuesta energía y la meta casi nunca se alcanza explorando, la política que Q-Learning aprende por defecto es **quedarse quieto** (acción 0, costo 0), que parece mejor que moverse y pagar sin llegar nunca. Por eso el Q-Learning puro daba **0 %** sin shaping. *No es solo lento: converge a una mala política.*

**Las palancas legítimas (no tocan la recompensa):**
- **Inicialización optimista** (Sutton & Barto §2.6): valores iniciales altos hacen que las acciones no probadas "se vean buenas" → exploración sistemática → muchas más chances de tropezar con la meta. Palanca #1.
- **Epsilon decay largo + más episodios**: más oportunidades de descubrir la meta.
- **γ alto** (0.99–0.999): propaga el `+100` lejano hacia atrás (con γ=0.99 sobre ~100 pasos, `+100` llega como ~37 al inicio: señal fuerte).
- **Discretización**: una grilla **más gruesa** puede facilitar alcanzar y generalizar la meta sin shaping (hipótesis a medir; es el opuesto del caso con shaping, donde ganó la media).
- **Dyna-Q** (obligatorio): el *planning* **repite las raras transiciones que llegan a la meta**, propagando la señal escasa. Ya observamos **75 % sin shaping (n=50)** vs **0 % de Q-Learning puro**.

**El hallazgo que ordena el relato:** el régimen "sin shaping = recompensa escasa" es **exactamente donde el libro (S&B cap. 8) motiva Dyna-Q**. La restricción del profesor **mejora** la comparación Q-Learning vs Dyna-Q en vez de complicarla.

**Expectativa honesta:** Dyna-Q es el camino confiable. Q-Learning puro es el signo de pregunta (puede aprender con optimismo + presupuesto, o quedar inestable/insuficiente). Medir esa varianza con boxplots **es justo lo que se pide**.

## 4. Mejoras de rigor incorporadas (correcciones al plan anterior)

1. **Seeding correcto por corrida:** sembrar `random`, `numpy` **y** `env.reset(seed)` de forma independiente por cada seed. (Hoy se siembra solo el primer reset → arreglarlo; sin esto la varianza medida no sería real.)
2. **Comparación apareada:** usar **el mismo set de 5 seeds** para todas las variantes/comparaciones, así las diferencias no se deben a seeds distintas (mismo criterio que en MATE).
3. **Protocolo de evaluación consistente:** `test_agent` debe evaluar sobre un **conjunto fijo de episodios sembrados** (mismos para todas las configs), no con el env sin seed.
4. **Selección por la distribución, no por una corrida:** elegir config por **mediana de éxito + baja varianza**, no por un único número afortunado.
5. **Presupuesto en dos niveles:** *screening* del grid (1500 ep, 5 seeds) → corrida final de las mejores (2000–2500 ep, 5 seeds).
6. **Al menos un modelo entregable** entrenado con **recompensa real** (probablemente el Dyna-Q), guardado en `models/`.

## 5. Fases de ejecución

### Fase 0 — Entorno e infraestructura
- Agregar **`seaborn`** y **`pandas`** al `MountainCarContinuous/pyproject.toml`.
- **Arreglar el seeding** en `train_agent`/`test_agent` (semilla por corrida, evaluación con seeds fijas).
- Definir helper de experimentos **multi-seed** (corre una config con las 5 seeds y devuelve todas las corridas) y un helper de **gráficos seaborn** (curvas con banda + boxplots).
- Seeds: **[0, 1, 2, 3, 4]**.

### Fase 1 — Factibilidad sin shaping (despeja el riesgo) ⭐ primero
- Sobre la **recompensa real**, experimentar para que Q-Learning y Dyna-Q aprendan:
  - **inicialización optimista** ∈ {0, 1, 10} (clave),
  - epsilon_decay largo (0.999), γ ∈ {0.99, 0.999}, episodios ~1500–2000,
  - discretizaciones {20×20×3 (gruesa), 40×40×5 (media)}.
- **Salida:** config(s) base que aprenden con recompensa real (idealmente Q-Learning; si no, Dyna-Q).

### Fase 2 — Exploración de hiperparámetros con varianza (sin shaping)
- Grid **OAT** desde la config base, variando:
  - **discretización** (gruesa / media [/ fina si el presupuesto lo permite]),
  - **α** ∈ {0.05, 0.1, 0.3},
  - **γ** ∈ {0.95, 0.99, 0.999},
  - **epsilon_decay** ∈ {0.995, 0.999},
  - **optimistic_init** ∈ {0, 1, 10}.
- Cada variante × **5 seeds** × 1500 ep. Registrar **todas las corridas** a JSON/CSV (no solo el promedio).

### Fase 3 — Dyna-Q vs Q-Learning (recompensa real, apareado, con varianza)
- `planning_steps` ∈ {0, 5, 25, 50}, cada uno × 5 seeds, misma config base.
- Comparación **justa** (mismas seeds) Q-Learning (n=0) vs Dyna-Q.

### Fase 4 — Visualizaciones (seaborn)
- **Curvas de aprendizaje con banda de error** (media ± dispersión entre seeds) — `sns.lineplot`.
- **Boxplots** de métricas finales (éxito, reward real, pasos) por configuración.
- **Boxplot Q-Learning vs Dyna-Q**.
- Mantener el **mapa de política aprendida** (V y π).

### Fase 5 — Reward shaping como EXTRA (mínimo)
- Una **sección breve** en anexo: el shaping es **potential-based (Ng-Harada-Russell)**, **no cambia la política óptima**, y **acelera** la convergencia — pero es **adicional** y **no la base** de los resultados. Sin experimento comparativo dedicado.

### Fase 6 — Documentación (ambos docs) + entregables
- Reescribir la sección **LOST** en `DocumentacionFinal.md` **y** `Documentacion.md`: nuevo enfoque (recompensa real, varianza, boxplots), **conclusiones personales**, **fuentes referenciadas** (S&B §2.6/6.5/8.2; Gymnasium MountainCarContinuous y `terminated`/`truncated`; seaborn).
- Actualizar los `.md` de apoyo de LOST.
- Actualizar el **notebook** (lee artefactos, muestra boxplots).
- **Re-generar los modelos** entregables (recompensa real) en `models/`.

## 6. Archivos que se tocan
`MountainCarContinuous/`: `pyproject.toml` (+seaborn, +pandas), `q_learning_agent.py` (seeding + protocolo de test), `grid_search.py` (multi-seed, sin shaping), `compare_dyna_q.py` (multi-seed), nuevo `plots_seaborn.py` (helper de gráficos), `train_best.py`, `continuous_mountain_car.ipynb`, `models/*.pkl` (regenerar), `*.json`. Docs: `DocumentacionFinal.md`, `Documentacion.md`, docs LOST de apoyo.

## 7. Diseño experimental concreto (resumen)
- **Seeds:** [0,1,2,3,4] (5). **Episodios:** 1500 (grid) / 2000–2500 (final).
- **Métricas:** tasa de éxito (ventana últimos 100 de train), **éxito en test sobre seeds fijas**, reward real promedio, pasos promedio, episodio de convergencia.
- **Reporte:** mediana + dispersión; gráficos con banda de error y boxplots.

## 8. Riesgos y mitigación
| Riesgo | Mitigación |
|---|---|
| Q-Learning puro no resuelve sin shaping | Conclusión honesta + boxplots de varianza; Dyna-Q como solución (decisión tomada). |
| Multi-seed multiplica el cómputo | Presupuesto **ligero** (5 seeds, 1500 ep) + grid acotado + screening/final en dos niveles. |
| Dyna-Q con n alto también lento | Acotar `planning_steps` y episodios; tabular es barato. |
| Documentar en dos docs → inconsistencia | Escribir una vez el contenido y replicarlo; checklist de consistencia al final. |

## 9. Checklist de cumplimiento de la devolución — ✅ COMPLETO
- [x] Q-Learning con recompensa real (sin shaping en el núcleo) — `q_learning_agent.py`, §2.4/2.5
- [x] Discretización + exploración de **diversas** discretizaciones — gruesa/media/fina, §2.7
- [x] Exploración de **α, γ, epsilon** + **epsilon decay** — grid OAT, §2.7
- [x] **Múltiples seeds** por config + **análisis de varianza** — 5 seeds, mín/std, §2.6/2.7
- [x] **Boxplots / bandas de error** (seaborn) — `experiments.py` + 6 gráficos seaborn
- [x] **Dyna-Q** + experimentación + **comparación con Q-Learning** (recompensa real) — §2.8
- [x] Reward shaping **solo como extra**, mínimo y aparte — §2.10
- [x] Documentación en ambos docs (`DocumentacionFinal.md` + `Documentacion.md`) + **fuentes** + **conclusiones personales** — §2.11
- [x] Al menos un **modelo computado** entregado (recompensa real) — `q_learning_best.pkl` + `dyna_q_best.pkl`

**Hallazgo central:** la palanca para aprender sin shaping era la **inicialización optimista** (no la recompensa). Q-Learning resuelve el ambiente al 100 % con la recompensa real. El análisis de varianza cambió la elección de config (la de mejor mediana era inestable). Dyna-Q n=5 ayuda; n=25 desestabiliza.
