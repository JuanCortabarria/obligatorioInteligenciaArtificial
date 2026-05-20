# Documentación — Obligatorio IA, Marzo 2026

**Materia:** Inteligencia Artificial — Ingeniería en Sistemas — Universidad ORT
**Entrega:** 06/07/2026
**Alumno:** Juan Cortabarría

> Este documento acompaña la entrega y deja registro del razonamiento, decisiones de diseño, justificación de hiperparámetros y resultados de cada paso. Se construye de forma incremental junto al código.

---

## 1. Contexto y consigna

La empresa ficticia *Red Destination™* nos contrata para implementar el agente inteligente del rover marciano "Out for Delivery". La consigna se divide en dos proyectos independientes:

- **Proyecto LOST** (Learning-based Orientation and Steering for Traversal): aprender a controlar el rover en `MountainCarContinuous-v0` (Gymnasium) usando **Q-Learning** tabular, con un componente de investigación que pide implementar **Dyna-Q** (Sutton & Barto, cap. 8.1–8.2).
- **Proyecto MATE** (Martian Adversarial Tactics Engine): implementar un agente para el juego *Isolation* usando **Minimax + Alpha-Beta** y **Expectimax**, con funciones de evaluación experimentables.

Como el grupo es de **2 personas**, no hay tarea adicional (Stochastic Q-Learning / MCTS quedan fuera de alcance).

### Por qué el problema es difícil

`MountainCarContinuous-v0` tiene un reward muy *sparso*:

- `−0.1·a²` por cada step (penalización por gastar energía).
- `+100` solo cuando se llega a la meta `x ≥ 0.45`.

Con política aleatoria el carro casi nunca llega a la cima — la tabla Q queda en cero y el agente no aprende nada. Eso justifica varias decisiones que iremos tomando: **discretización agresiva**, **exploración prolongada (ε alto, decay lento)**, **inicialización optimista** y, sobre todo, **reward shaping** opcional.

---

## 2. Proyecto LOST — Mountain Car Continuous

### 2.1 Modelado como MDP

Mapeando al formalismo clásico (Introducción a MDP, slide 7):

| Componente | En el ambiente |
|------------|----------------|
| Estados `S` | Pares `(x, v)` con `x ∈ [−1.2, 0.6]`, `v ∈ [−0.07, 0.07]`. Tras discretizar: `S = bins_x × bins_v`. |
| Acciones `A` | Fuerza `a ∈ [−1, 1]`. Tras discretizar: `A = {a₁, …, aₖ}`. |
| Transición `P(s' \| s, a)` | Determinista en el simulador (física de Newton), pero el agente la trata como desconocida. Dyna-Q luego la **aprende** explícitamente como modelo. |
| Reward `R(s, a, s')` | `−0.1·a²` por step, `+100` al alcanzar la meta. |
| Done | `x ≥ 0.45` o se alcanza el tope (999 steps). |

### 2.2 Paso 1 — Discretización

**Archivo:** [`MountainCarContinuous/discretization.py`](MountainCarContinuous/discretization.py)

Se implementó una clase `Discretizer` parametrizada por `n_bins_x`, `n_bins_v` y `n_actions`. Por qué los parámetros van en una clase y no como constantes globales (como estaba en el notebook scaffold):

1. **Permite hacer grid search**: instanciar varios `Discretizer(...)` con distintas resoluciones sin reescribir el código del agente.
2. **Centraliza la conversión** observación ↔ índice de estado e índice ↔ acción real, evitando duplicarla en el notebook y en el agente.
3. **Encapsula el `+1`** que aparece por la forma en que funciona `np.digitize` (puede devolver índice = `len(bins)` cuando el valor cae sobre o por encima del último bin), evitando un off-by-one al definir el shape de Q.

**Método de discretización:** `np.linspace` para definir los bordes de los bins (espaciado uniforme) y `np.digitize` para mapear observación → índice. Es el método más simple y predecible; alternativas como `KBinsDiscretizer` de sklearn con estrategia *quantile* tendrían más sentido si la distribución de estados visitados fuese muy desbalanceada, pero acá la dinámica del carro recorre todo el rango de `x` y `v` con razonable uniformidad, así que el `linspace` uniforme es suficiente.

**Configuraciones que vamos a comparar más adelante en el grid search:**

| Config | bins_x | bins_v | n_actions | tamaño Q | Comentario |
|--------|--------|--------|-----------|----------|-----|
| Gruesa | 20     | 20     | 3         | ~1.2 K   | Aprende rápido, pero pierde detalle cerca del valle. |
| Media  | 40     | 40     | 5         | ~8 K     | Compromiso razonable de partida. |
| Fina   | 100    | 100    | 10        | ~100 K   | Más resolución pero la tabla queda más *sparsa* — necesita más episodios. |

El trade-off central: **más bins ⇒ tabla más expresiva pero más sparsa ⇒ más exploración necesaria para llenarla**. La consigna pide justamente justificar esta elección y medir su impacto, así que las tres configs van a ser corridas en el grid search.

**Discretización de acciones:** se eligió un número *impar* (3, 5, ...) de acciones a propósito. Esto garantiza que `a = 0` (no aplicar fuerza) sea una acción discreta accesible — sin ella, el agente solo puede empujar en una dirección o la otra, lo cual es coherente con la dinámica del MountainCar pero quita una acción "neutral" que puede ser útil en el aprendizaje temprano.

### 2.3 Paso 2 — Q-Learning *(pendiente)*

### 2.4 Paso 3 — Búsqueda de hiperparámetros *(pendiente)*

### 2.5 Paso 4 — Dyna-Q *(pendiente)*

---

## 3. Proyecto MATE — Isolation *(pendiente)*

---

## 4. Uso de IA Generativa

Conforme exige la consigna (p. 1 del PDF), declaro el uso de IA generativa:

- **Herramienta utilizada:** Claude (Anthropic), modelo Claude Opus 4.7, accedido a través de Claude Code.
- **Contexto de uso:**
  - **Redacción inicial** de esta documentación a partir de la planificación previa y los PDFs de teoría del curso.
  - **Generación de código** de la clase `Discretizer`, el agente `QLearningAgent` y `DynaQAgent`, partiendo del scaffold provisto por la cátedra y del pseudocódigo de Sutton & Barto.
  - **Análisis y discusión** de resultados del grid search.
- Todo el contenido producido por la IA fue **revisado, ejecutado y verificado** por el alumno antes de ser incorporado. Los errores que pueda haber son responsabilidad del alumno.
