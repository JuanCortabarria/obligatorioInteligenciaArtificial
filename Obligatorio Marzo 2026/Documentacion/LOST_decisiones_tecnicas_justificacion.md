# LOST - Decisiones tecnicas y justificacion

Este documento explica las decisiones tomadas para resolver el proyecto LOST
del obligatorio, basado en `MountainCarContinuous-v0`. Complementa la
planificacion inicial y deja asentado por que la implementacion actual usa
Q-Learning tabular, discretizacion uniforme, reward shaping potential-based y
Dyna-Q como componente de investigacion.

## 1. Modelado del ambiente como MDP

El ambiente se modelo como un MDP episodico:

| Componente | Decision tomada |
|------------|-----------------|
| Estado | Observacion continua `(x, v)`, con posicion `x` y velocidad `v`. |
| Acciones | Fuerza continua `a` en `[-1, 1]`, discretizada para poder usar una tabla Q. |
| Transicion | La dinamica del simulador se trata como desconocida para Q-Learning; Dyna-Q aprende un modelo tabular de las transiciones observadas. |
| Reward | Reward real del ambiente, sin shaping, usado para reportar metricas comparables. |
| Fin de episodio | `terminated=True` significa llegada a la meta; `truncated=True` significa corte artificial por limite de pasos. |

La distincion entre `terminated` y `truncated` es importante. En Gymnasium, un
timeout no es un estado terminal del MDP. Por eso el bootstrap futuro se anula
solo con `terminated=True`; si el episodio termina por `truncated=True`, el
estado siguiente sigue teniendo valor estimado.

## 2. Discretizacion

Archivo principal: `../MountainCarContinuous/discretization.py`.

Como Q-Learning tabular requiere espacios discretos, se implemento una clase
`Discretizer` con tres parametros: `n_bins_x`, `n_bins_v` y `n_actions`.

Decisiones:

- Se uso discretizacion uniforme con `np.linspace` para posicion, velocidad y acciones.
- Se uso `np.digitize` para mapear cada observacion continua a indices enteros.
- La forma de la tabla usa `n_bins + 1` por dimension de estado porque `np.digitize` puede devolver el indice `len(bins)` en el extremo superior.
- Se usaron cantidades impares de acciones para incluir la accion neutral `0.0`.

La discretizacion uniforme es simple, reproducible y suficiente para este
ambiente. El costo principal es el trade-off entre resolucion y sparsity: mas
bins permiten distinguir mejor estados cercanos, pero agrandan la tabla Q y
exigen mas exploracion. Por eso se compararon configuraciones gruesas, medias y
finas en la busqueda de hiperparametros.

## 3. Q-Learning

Archivo principal: `../MountainCarContinuous/q_learning_agent.py`.

Se implemento Q-Learning off-policy con la regla:

```text
Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
```

La eleccion de Q-Learning es directa porque la consigna lo pide, pero ademas es
adecuada para este caso: permite explorar con una politica epsilon-greedy y, al
mismo tiempo, aprender la politica greedy objetivo.

Decisiones de entrenamiento:

- Politica de comportamiento epsilon-greedy.
- Decay exponencial de epsilon por episodio, no por step, para no apagar la exploracion demasiado rapido.
- `epsilon_min` conserva exploracion residual durante todo el entrenamiento.
- `optimistic_init` queda parametrizado para comparar inicializacion optimista, aunque el mejor resultado actual no depende de ella.
- Las metricas guardan reward real sin shaping, para que los experimentos con y sin shaping sean comparables.

Persistencia:

- `save()` guarda la tabla Q, configuracion de discretizacion e hiperparametros.
- `load()` reconstruye el agente y deja `epsilon` en `epsilon_min`, apropiado para evaluar sin exploracion fuerte.
- La ruta esperada por los scripts actuales es `../MountainCarContinuous/models/*.pkl`. Esa carpeta ya existe y contiene los modelos entregables principales; ademas se agregaron excepciones en `.gitignore` para que esos `.pkl` puedan incluirse en Git/ZIP sin habilitar todos los modelos genericos del repo.

## 4. Reward shaping

El reward original de `MountainCarContinuous-v0` es muy escaso: el agente recibe
una senal fuerte solo al llegar a la meta. Para que el aprendizaje tabular sea
viable, se agrego reward shaping opcional.

La decision final fue usar shaping potential-based:

```text
F(s,s') = gamma * Phi(s') - Phi(s)
Phi(s) = coef * abs(v)
reward_shaped = reward + F(s,s')
```

Justificacion:

- `abs(v)` premia construir momento, que es la estrategia fisica necesaria para salir del valle.
- La forma potential-based preserva la politica optima bajo las condiciones teoricas de Ng, Harada y Russell.
- Un shaping aditivo simple puede cambiar el problema, porque el agente podria preferir oscilar para acumular reward en vez de llegar a la meta.

Caso terminal:

- Si `terminated=True`, se toma `Phi(s') = 0`.
- Por lo tanto, en el step terminal se aplica `reward + gamma * 0 - Phi(s)`, no solo `reward`.
- Esta decision evita sobre-premiar estados pre-meta con velocidad alta.

## 5. Dyna-Q

Archivos principales: `../MountainCarContinuous/dyna_q_agent.py` y
`../MountainCarContinuous/compare_dyna_q.py`.

Dyna-Q se implemento como extension de Q-Learning:

- Se toma una transicion real del ambiente.
- Se actualiza Q con esa experiencia.
- Se guarda un modelo tabular `Model(s,a) -> (reward, s', terminated)`.
- Se hacen `planning_steps` actualizaciones simuladas muestreando pares ya observados.

El ambiente es determinista, por lo que un modelo tabular que sobrescribe la
ultima transicion observada para cada `(s,a)` es razonable. Para ambientes
estocasticos haria falta modelar distribuciones o promedios, pero eso queda
fuera del alcance pedido.

El modelo guarda el reward shaped, no el reward crudo. La razon es que el
planning debe repetir la misma senal que vio el agente en la experiencia real.
Si se guardara reward crudo y se intentara recalcular shaping durante planning,
se mezclaria una transicion discretizada con una funcion de potencial definida
sobre observaciones continuas.

## 6. Busqueda de hiperparametros

Archivo principal: `../MountainCarContinuous/grid_search.py`.

La busqueda se hizo con estrategia one-at-a-time (OAT): partir de una
configuracion base validada y variar un hiperparametro por corrida. Esta
estrategia no explora todas las interacciones posibles, pero es mucho mas
interpretable que un producto cartesiano grande y alcanza para justificar el
impacto de discretizacion, `alpha`, `gamma`, epsilon decay, shaping e
inicializacion optimista.

Metricas definidas:

- `train_success_rate_last100`: porcentaje de exitos en los ultimos 100 episodios de entrenamiento.
- `train_avg_reward_last100`: reward real promedio de los ultimos 100 episodios.
- `convergence_ep_50w_0.9`: primer episodio donde una ventana movil de 50 episodios alcanza al menos 90% de exito.
- `test_success_rate`: porcentaje de exitos con politica greedy.
- `test_avg_reward`: reward real promedio en test greedy.
- `test_avg_steps`: pasos promedio en test greedy.

Criterio de seleccion:

1. Maximizar `test_success_rate`.
2. Entre empates, minimizar `test_avg_steps`.
3. Como desempate final, maximizar `test_avg_reward`.

El criterio prioriza resolver el ambiente de forma consistente y eficiente. En
este problema, llegar en menos pasos suele indicar una politica mas estable y
tambien reduce el costo acumulado de las acciones.
