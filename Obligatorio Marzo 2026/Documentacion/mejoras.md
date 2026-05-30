# LOST - Errores, inconsistencias y mejoras detectadas

Alcance: este archivo cubre solo el proyecto LOST
(`MountainCarContinuous-v0`). No analiza MATE/Isolation.

## 1. `models/` ya existe y contiene los modelos esperados

Los scripts y el notebook esperan modelos en:

- `../MountainCarContinuous/models/q_learning_best.pkl`
- `../MountainCarContinuous/models/dyna_q_best.pkl`
- `../MountainCarContinuous/models/q_learning_grid_best.pkl`
- `../MountainCarContinuous/models/smoke_test.pkl`

La carpeta ya fue agregada y contiene esos cuatro archivos. Se verifico con
`python3.10` que los dos modelos principales cargan con las clases actuales:

- `q_learning_best.pkl`: Q shape `(41, 41, 5)`, discretizacion `40x40`, `5` acciones, `alpha=0.05`, `gamma=0.99`, shaping activo.
- `dyna_q_best.pkl`: misma discretizacion e hiperparametros, `planning_steps=5`, modelo interno con 5299 entradas.

Mejora aplicada:

- Se agregaron excepciones en `.gitignore` para permitir versionar los `.pkl` de `../MountainCarContinuous/models/`.
- Ya se puede afirmar que los modelos entregables principales estan en la ruta esperada.

## 2. Los `.pkl` de `resultados_lost/` parecen artefactos legacy

Existen modelos en:

- `../MountainCarContinuous/resultados_lost/`
- `../MountainCarContinuous/resultados_lost_tmp/`

Los metadatos de `resultados_lost/` indican una configuracion distinta a la
implementacion/documentacion actual:

| Archivo | Clase | Configuracion | Episodios | Resultado test |
|---------|-------|---------------|-----------|----------------|
| `best_model.pkl` | `DynaQAgent` | `31x31`, `7` acciones, `planning_steps=2` | 120 | 100%, reward 90.35, 291.5 steps |
| `dyna_q_shaping_model.pkl` | `DynaQAgent` | `31x31`, `7` acciones, `planning_steps=2` | 120 | 100%, reward 90.35, 291.5 steps |
| `q_learning_base_model.pkl` | `QLearningAgent` | `31x31`, `7` acciones, sin shaping | 120 | 0%, reward 0.00, 999 steps |
| `q_learning_shaping_model.pkl` | `QLearningAgent` | `31x31`, `7` acciones, con shaping | 120 | 90%, reward 72.49, 467.7 steps |

Problema:

- El grid search actual documenta como mejor configuracion `40x40`, `5` acciones, `alpha=0.05`.
- Los `.pkl` disponibles no siguen el formato actual de `save()`/`load()` (`Q`, `discretizer_config`, `hyperparams`), sino otro formato con `q_table`, `config` y `metadata`.
- Por eso no deben tratarse automaticamente como los modelos finales esperados por los scripts actuales.

Mejora recomendada:

- Mantener estos modelos como evidencia historica si se desea.
- Generar modelos nuevos con la interfaz actual antes de entregar.

## 3. `.gitignore` ahora permite los modelos LOST

El `.gitignore` ignora:

- `models/`
- `*.pkl`
- `*.pickle`
- otros formatos de modelos.

Esto sigue siendo razonable para no versionar artefactos pesados genericos.
Como LOST exige modelos computados, se agregaron excepciones especificas para:

- `../MountainCarContinuous/models/`
- `../MountainCarContinuous/models/*.pkl`

Mejora recomendada:

- Verificar con `git status` que esos `.pkl` aparezcan como agregables.
- Revisar el contenido del ZIP final antes de subirlo a Gestion.
- Recordar que el PDF exige al menos un modelo computado para el primer ejercicio.

## 4. El notebook ya tiene rutas de modelos disponibles

`../MountainCarContinuous/continuous_mountain_car.ipynb` contiene celdas que
guardan/cargan:

- `models/smoke_test.pkl`
- `models/q_learning_best.pkl`
- `models/dyna_q_best.pkl`

Como `models/` ya existe, las celdas no deberian fallar por ausencia de archivos
finales. Falta, de todos modos, confirmar una ejecucion completa del notebook en
el entorno de entrega.

Mejora recomendada:

- Ejecutar el notebook completo antes del ZIP final.
- Si aparece un error de entorno, documentar dependencia/comando exacto de ejecucion.

## 5. `planificacionLOST.md` conserva lenguaje historico

El archivo `planificacionLOST.md` fue escrito como plan inicial. Todavia
menciona frases como:

- "Completar la clase ya scaffoldeada"
- "hoy solo tiene `pass`"
- "completar el update de Q"
- "NUEVO: carpeta `models/`"

Eso ya no describe fielmente el estado actual del codigo.

Mejora aplicada:

- Se agrego una nota al inicio aclarando que el archivo es historico.

Mejora recomendada:

- Usar `LOST_estado_y_plan.md` como fuente de estado actual.
- Usar `LOST_decisiones_tecnicas_justificacion.md` y `LOST_experimentos_resultados.md` para el informe final.

## 6. `Documentacion.md` de la raiz requiere revision final

El archivo `../Documentacion.md` contiene mucho material util, pero tambien
fue escrito antes de esta normalizacion. Las referencias a `models/` ya no son
contradictorias por existencia de carpeta, pero conviene revisar que no conserve
notas viejas sobre modelos faltantes o resultados legacy.

Mejora recomendada:

- Antes de convertir la documentacion a informe PDF, revisar y alinear esas referencias.
- Si se decide mantener `Documentacion.md` como borrador historico, indicarlo explicitamente.

## 7. Mejora futura: ciclos de re-exploracion

La documentacion historica registra un hallazgo interesante: cargar un modelo y
re-entrenarlo con `reset_epsilon=True` puede mejorar la politica porque fuerza
una nueva fase de exploracion. No es necesario para cumplir la consigna, pero
puede mejorar pasos promedio.

Mejora recomendada:

- No cambiarlo para la entrega salvo que sobre tiempo.
- Si se experimenta, documentar claramente que es una extension y comparar contra el modelo base.
