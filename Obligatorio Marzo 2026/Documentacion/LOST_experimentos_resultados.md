# LOST - Experimentos y resultados

Este documento resume los resultados actuales del proyecto LOST. Los numeros
fueron tomados de los artefactos existentes en
`../MountainCarContinuous/grid_search_results.json`,
`../MountainCarContinuous/dyna_q_comparison.json` y
`../MountainCarContinuous/dyna_q_no_shaping.json`.

Nota de artefactos: la carpeta esperada `../MountainCarContinuous/models/` ya
existe y contiene `q_learning_best.pkl`, `dyna_q_best.pkl`,
`q_learning_grid_best.pkl` y `smoke_test.pkl`. Los modelos principales fueron
verificados con `python3.10`: cargan con las clases actuales y coinciden con la
configuracion final documentada (`40x40`, `5` acciones, `alpha=0.05`,
`gamma=0.99`, shaping activo; Dyna-Q usa `planning_steps=5`).

## 1. Smoke test

El smoke test valida que la pipeline basica aprende: discretizacion,
epsilon-greedy, update de Q, reward shaping, evaluacion greedy, guardado y
grafica.

Configuracion base del smoke test:

- Ambiente: `MountainCarContinuous-v0`.
- Discretizacion: `40x40` bins de estado y `5` acciones.
- `alpha=0.1`, `gamma=0.99`.
- `epsilon_start=1.0`, `epsilon_min=0.05`, `epsilon_decay=0.995`.
- Reward shaping potential-based con `shaping_coef=300.0`.
- Semilla: `42`.
- Episodios: `500`.

Grafico asociado:

- `../MountainCarContinuous/plots/smoke_test_learning_curve.png`

Lectura del resultado: el smoke test no busca ser el modelo final, sino probar
que la tabla Q deja de estar en cero y que el agente logra encontrar politicas
exitosas con una configuracion razonable.

## 2. Grid search de Q-Learning

La busqueda de hiperparametros se hizo con estrategia OAT. Cada corrida entreno
800 episodios y luego evaluo la politica greedy en 20 episodios.

| Run | Conv. ep | Train success ult. 100 | Test success | Test reward | Test steps |
|-----|----------|------------------------|--------------|-------------|------------|
| `alpha_0.05` | 73 | 100% | 100% | 92.48 | 115.75 |
| `bins_fina_100` | 128 | 100% | 100% | 93.27 | 119.20 |
| `shaping_coef_600` | 72 | 100% | 100% | 92.06 | 120.10 |
| `optimistic_init_1.0` | 73 | 100% | 100% | 92.43 | 121.40 |
| `base` | 72 | 100% | 100% | 92.25 | 127.90 |
| `eps_decay_0.99` | 62 | 100% | 100% | 92.03 | 131.50 |
| `eps_decay_0.999` | 175 | 100% | 100% | 91.99 | 151.20 |
| `alpha_0.3` | 77 | 100% | 100% | 91.48 | 153.85 |
| `bins_gruesa_20` | 64 | 100% | 95% | 83.94 | 131.50 |
| `gamma_0.95` | 75 | 100% | 95% | 86.22 | 189.20 |
| `shaping_coef_100` | 86 | 100% | 95% | 88.11 | 226.85 |
| `shaping_off` | - | 0% | 0% | 0.00 | 999.00 |

Graficos asociados:

- `../MountainCarContinuous/plots/grid_search_curves.png`
- `../MountainCarContinuous/plots/grid_search_summary.png`

Interpretacion:

- El mejor run segun el criterio definido es `alpha_0.05`: mantiene 100% de exito en test y logra el menor promedio de pasos entre las politicas con exito completo.
- `bins_fina_100` obtiene reward levemente mayor, pero converge mas tarde y no mejora los pasos promedio.
- `shaping_off` no aprende una politica util en este setup: termina con 0% de exito en test y 999 pasos promedio.
- `shaping_coef_100` queda corto; `shaping_coef_600` funciona, pero no mejora a `300`.
- `gamma_0.95` y la discretizacion gruesa llegan a buenos resultados en entrenamiento, pero pierden robustez en test.

## 3. Dyna-Q con reward shaping

El experimento compara `planning_steps` en `{0, 5, 25, 50}` usando la misma
familia de configuracion que el mejor Q-Learning: bins `40x40`, `5` acciones,
`alpha=0.05`, `gamma=0.99`, epsilon decay `0.995` y shaping potential-based con
coeficiente `300`.

| Planning steps | Conv. ep | Train success ult. 100 | Test success | Test reward | Test steps | Tiempo | Tamano modelo |
|----------------|----------|------------------------|--------------|-------------|------------|--------|---------------|
| 0 | 73 | 100% | 100% | 94.00 | 89.44 | 0.90s | 5328 |
| 5 | 73 | 100% | 100% | 93.18 | 94.28 | 3.14s | 5299 |
| 25 | 104 | 100% | 100% | 93.34 | 112.90 | 9.68s | 4985 |
| 50 | 133 | 100% | 100% | 90.11 | 153.64 | 19.34s | 4846 |

Graficos asociados:

- `../MountainCarContinuous/plots/dyna_q_comparison.png`
- `../MountainCarContinuous/plots/dyna_q_convergence_vs_time.png`

Interpretacion:

- Con shaping efectivo, Q-Learning puro (`n=0`) ya aprende muy rapido.
- `n=5` empata la convergencia en episodios reales, pero usa mas tiempo de computo.
- `n=25` y `n=50` amplifican transiciones tempranas de un modelo incompleto y empeoran pasos promedio.
- El costo crece casi linealmente con `planning_steps`, tal como se espera.

Decision reportable: si se exige un modelo Dyna-Q con planning real, `n=5` es
el mejor compromiso entre exito, pasos y costo. Si solo importara desempeno
final en esta corrida, `n=0` es Q-Learning puro y no cuenta como Dyna-Q.

## 4. Dyna-Q sin reward shaping

Este experimento sirve para aislar el efecto del planning cuando la senal de
reward vuelve a ser escasa. En este caso se uso una configuracion mas simple
sin shaping.

| Planning steps | Conv. ep | Train success ult. 100 | Test success | Test reward | Test steps | Tiempo | Tamano modelo |
|----------------|----------|------------------------|--------------|-------------|------------|--------|---------------|
| 0 | - | 0% | 0% | 0.00 | 999.00 | 6.90s | 636 |
| 5 | - | 0% | 0% | 0.00 | 999.00 | 16.30s | 672 |
| 25 | 306 | 93% | 0% | -72.17 | 999.00 | 23.20s | 766 |
| 50 | 459 | 88% | 75% | 42.18 | 388.65 | 47.30s | 771 |

Grafico asociado:

- `../MountainCarContinuous/plots/dyna_q_no_shaping.png`

Interpretacion:

- Sin shaping, Q-Learning puro no descubre una politica util.
- Dyna-Q empieza a ser valioso cuando `planning_steps` es alto.
- `n=50` logra 75% de exito en test, mostrando que el planning ayuda a propagar experiencia escasa.
- El resultado respalda la lectura de Sutton y Barto: Dyna-Q aporta mas cuando cada experiencia real es costosa o rara.

## 5. Conclusiones experimentales

Las conclusiones actuales son:

- La decision mas determinante fue usar reward shaping potential-based correctamente.
- La configuracion `40x40` con `5` acciones es un buen equilibrio entre resolucion y cantidad de estados.
- El mejor Q-Learning registrado en el grid es `alpha_0.05`.
- Dyna-Q no supera claramente a Q-Learning cuando el shaping ya hace densa la senal.
- Dyna-Q si muestra valor en el escenario sin shaping, donde el reward escaso dificulta mucho el aprendizaje.
- Para el ZIP final, los modelos ya estan ubicados en `../MountainCarContinuous/models/`; resta revisar que queden incluidos en el paquete final junto con codigo, notebooks, graficos y documentacion.
