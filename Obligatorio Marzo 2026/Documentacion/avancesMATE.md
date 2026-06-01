# Avances — Proyecto MATE

> Bitácora de progreso de MATE. Se actualiza **cada vez que se avanza** un paso de la
> resolución de `PlanificacionMATE.md` §8, para que cualquiera del equipo vea el estado
> y sepa **dónde retomar**.
>
> **Leyenda:** ⬜ pendiente · ⏳ en curso · ✅ completado
>
> **Cómo usar este archivo:**
> 1. Al empezar un paso, cambialo a ⏳ en la tabla y agregá una entrada en la bitácora.
> 2. Al terminarlo, marcalo ✅ acá **y** en `PlanificacionMATE.md` §8, y completá la entrada (qué se hizo, archivos, cómo se verificó).
> 3. Mantené actualizado **"Dónde retomar"** al final.

---

## Estado general

| Paso | Descripción | Archivo(s) | Estado | Responsable | Fecha |
|------|-------------|------------|--------|-------------|-------|
| 0 | Setup: `play_match()` con seed/stats (sin tocar archivos dados) | `match.py` | ✅ | Martín | 2026-06-01 |
| 1 | Núcleo de búsqueda (sucesores, nodos, utilidad terminal) | `search.py` | ✅ | Martín | 2026-06-01 |
| 2 | MinimaxAgent profundidad fija (sin poda) | `minimax_agent.py` | ✅ | Juan | 2026-06-01 |
| 3 | Alpha-Beta + ordenamiento de movimientos | `minimax_agent.py` | ✅ | Juan | 2026-06-01 |
| 4 | Funciones de evaluación h1–h4 + `weighted_eval` | `evaluation.py` | ✅ | Juan | 2026-06-01 |
| 5 | ExpectimaxAgent (nodos de azar) | `expectimax_agent.py` | ✅ | Juan | 2026-06-01 |
| 6 | Experimentos E1–E6 + registro a CSV (en el notebook) | `isolation.ipynb`, `results.csv` | ✅ (corrida final N alto) | Juan | 2026-06-01 |
| 7 | Gráficos + tablas + guardar `mate_best_config.pkl` | `isolation.ipynb`, `plots/`, `mate_best_config.pkl` | ✅ | Juan | 2026-06-01 |
| 8 | Cierre del notebook + sección MATE del informe | `isolation.ipynb`, informe PDF | ⏳ notebook ✅ / informe PDF pendiente | Juan | 2026-06-01 |

**Progreso:** 8 / 9 pasos completados. Código, notebook (corre de punta a punta), datos, gráficos y `.pkl` **listos**. Resta solo **redactar la sección MATE del informe PDF** (Paso 8) apoyándose en `DocumentacionMATE.md` y `plots/`.

---

## Notas de entorno (setup de Poetry)

> Leer antes de correr nada en una máquina nueva. Estas notas surgieron al levantar el
> proyecto en la máquina de Juan (la de Martín ya lo tenía andando).

1. **`poetry install` "a secas" falla** con `No file/folder found for package isolation`.
   La causa es que `Isolation/` tiene los `.py` planos (no es un paquete instalable) y el
   `pyproject.toml` está en *package mode*. **Solución:** instalar con
   `poetry install --no-root` (instala solo las dependencias, no el "proyecto"). No se
   modifica `pyproject.toml`.

2. **Poetry puede estar roto** con el error `No module named 'packaging.licenses'`. Es un
   `packaging` viejo en el intérprete donde vive Poetry. **Solución** (una sola vez, en la
   instalación de Poetry, no en el venv del proyecto):
   `<python-de-poetry> -m pip install --upgrade "packaging>=24.2" "platformdirs>=4.3.6"`.
   En la máquina de Juan ese intérprete era `~/.pyenv/versions/3.10.20/bin/python`.

3. **Comando estándar** para correr y verificar todo en MATE (desde `Isolation/`):
   `poetry run python <archivo>.py`.

---

## Bitácora de avances

> Una entrada por sesión de trabajo / por paso. La más reciente arriba.

### Plantilla (copiar para cada avance)

```
### [AAAA-MM-DD] Paso N — <título> — <responsable>
- **Estado:** ⏳ en curso | ✅ completado
- **Qué se hizo:** ...
- **Archivos tocados:** ...
- **Decisiones / desvíos respecto del plan:** ... (si los hubo)
- **Cómo se verificó:** ... (comando / test / resultado observado)
- **Pendiente / próximos pasos:** ...
```

<!-- Agregar las entradas reales debajo de esta línea -->

### [2026-06-01] Mejoras aplicadas (seeds apareados + tiempo por agente + N) y re-corrida — Juan
- **Estado:** ✅ completado
- **Qué se hizo:** se implementaron las **3 mejoras** que habían quedado propuestas y se recalculó todo:
  1. **Diseño de seeds apareado** en `run_matchup`: cada seed se juega **dos veces** (nuestro agente de jugador 1 y de jugador 2) sobre la **misma colocación inicial** → controla posición + ventaja de primer jugador (antes solo se alternaban lados con seeds distintos).
  2. **Tiempo por agente:** `match.py` ahora devuelve `avg_move_time_p1`/`avg_move_time_p2` además del promedio del juego; `run_matchup` registra `a_avg_move_time` (costo de **nuestro** agente, no contaminado por el rival).
  3. **N por celda:** con el apareo, N por lado = nº de seeds; quedó 40/lado vs Stratagem (antes ~25).
- **Archivos tocados:** `match.py` (tiempos por jugador), `isolation.ipynb` (celdas intro/config/helpers reescritas, re-ejecutado), `results.csv` (**1568 filas**, nueva col. `a_avg_move_time`), `plots/` (4 PNG regenerados), `mate_best_config.pkl`. No se tocó ningún archivo del simulador.
- **Cómo se verificó:** `jupyter nbconvert --execute --inplace` → **0 errores**, ~15 min; los **5 smoke tests** en verde (incl. `match.py` con el cambio); 1568 filas con la columna nueva.
- **Resultados nuevos (diseño apareado — confirman y AFINAN los anteriores):**
  - **E1 (Alpha-Beta):** idéntico (mide búsquedas aisladas): 80 %/74 %/91 % de poda a d=2/3/4.
  - **E2 (vs Random):** Minimax **96 %**, Expectimax **94.5 %** (200 partidas c/u).
  - **E3 (vs Stratagem, 80 part./celda):** **d=2:** Expectimax 48 % > Minimax 39 %. **d=3:** **Minimax 46 % > Expectimax 34 %** (hipótesis confirmada, brecha más limpia). **Dato nuevo del tiempo por agente:** a d=3 Expectimax cuesta **0.59 s y ~24.100 nodos/jugada** vs **0.13 s y ~1.300** de Minimax (~4.5× más lento, ~18× más nodos: no poda). Expectimax es a la vez **más débil y más caro**.
  - **E4 (directo, d=2):** Minimax 44 % / Expectimax 56 %.
  - **E5 (profundidad):** 27 % → 39 % → 46 % (d=1,2,3), monótono.
  - **E6 (heurísticas):** `solo_mov_diff` **0.700** (mejor) > `mov+centro` 0.578 > `balanceada` 0.483 > `mov+acorralar` 0.239.
  - **`mate_best_config.pkl`:** `{minimax, d=3, {h2:1.0}}`, `win_vs_stratagem_d3=0.463`, `win_vs_random=0.96`, `e6=0.70`.
  - **Efecto de lado (control):** 48 % ganando de arranque vs 35 % de segundo → el apareo lo neutraliza.
- **Documentación actualizada:** DocumentacionMATE §3.3 (confirmación con nuevos números), §3.6 (diseño apareado), §4 (métricas por agente), §5 (todos los resultados), §6 (caveats: ventaja de primer jugador ahora *controlada*, costo por agente *resuelto*).
- **Decisiones / desvíos:** las conclusiones del informe **no cambian** (Minimax mejor a profundidad igualada; movilidad sola como mejor heurística); el apareo y el tiempo por agente las hacen **más rigurosas** y agregan el argumento de costo a favor de Minimax.
- **Pendiente / próximos pasos:** re-commit de los artefactos regenerados (`match.py`, `isolation.ipynb`, `results.csv`, `plots/`, docs); Paso 8 (informe PDF).

### [2026-06-01] Verificación general (errores + mejoras) — Juan
- **Estado:** ✅ completado
- **Qué se hizo:** auditoría de correctitud de todo el código MATE + búsqueda de errores y mejoras.
- **Chequeos pasados:**
  - **Smoke tests:** los 5 (`search`, `match`, `evaluation`, `minimax_agent`, `expectimax_agent`) en verde.
  - **Equivalencia Alpha-Beta ↔ Minimax (exhaustiva):** sobre **292 estados** (40 seeds × 4 aperturas × d∈{2,3}): **0** diferencias de valor, **0** diferencias de acción con `move_ordering=False`, **0** casos con `nodos(AB) > nodos(Minimax)`. Poda global **89.8 %**. La poda no cambia la decisión, solo el costo. ✓
  - **Notebook:** 0 celdas con error en sus outputs; corre de punta a punta.
- **Errores / hallazgos:**
  - **Bug en archivo dado (`stratagem.py`), NO en nuestro código:** `Stratagem(2)` evalúa su propia derrota como **0 en vez de −1** (test directo: `Stratagem(1)` → −1 correcto; `Stratagem(2)` → 0). Lo hace más débil de jugador 2. No se corrige (no se tocan archivos dados); se documentó como nota de advertencia (DocumentacionMATE §6) porque infla un poco nuestras victorias cuando Stratagem va segundo.
  - **Ventaja de primer jugador (medida):** nuestro agente gana 43 % arrancando vs 25 % cuando arranca el rival → valida alternar lados. Al desagregar por celda (técnica×prof×lado) el N es chico (~25) y ruidoso.
  - **Métrica `avg_move_time` mal interpretable:** es el promedio por ply sobre **ambos** jugadores (no por agente); vs Stratagem la domina Stratagem. El costo limpio por agente sale de **E1** y de `a_nodes_per_move`. Documentado en §6.
  - **No se encontraron bugs en nuestro código** (Minimax, Alpha-Beta, Expectimax, heurísticas).
- **Mejoras implementadas:**
  - **Assert defensivo** en `next_action` de ambos agentes (`action is not None`): documenta el invariante y da un error claro si alguna vez se invoca sobre un estado terminal.
  - **README.md** del directorio `Isolation/` actualizado: lista los archivos nuevos de MATE y los artefactos generados.
  - **Corrección de consistencia** en DocumentacionMATE §4 (decía "no se guarda .pkl", contradecía §3.8/§5.6).
- **Mejoras propuestas (las 3 se IMPLEMENTARON después — ver entrada del 2026-06-01 "Mejoras aplicadas"):**
  1. **Diseño de seeds apareado** en `run_matchup`: jugar cada seed **dos veces** (una por lado) para controlar la posición inicial → menor varianza.
  2. **Tiempo por agente** en los matchups (medir cada agente por separado).
  3. **Subir N por celda** para análisis confiable desagregado.
  > Nota: primero se decidió **no implementarlas** (2026-06-01) y luego el equipo pidió **aplicarlas**; quedaron implementadas y los resultados se recalcularon (ver entrada "Mejoras aplicadas").
- **Cómo se verificó:** scripts de verificación ad-hoc (`poetry run python -c ...`): equivalencia AB sobre 292 estados, probe del terminal de Stratagem, split de `results.csv` por lado.
- **Pendiente / próximos pasos:** se implementaron las 3 mejoras; luego Paso 8 (redactar sección MATE del informe PDF).

### [2026-06-01] Pasos 6 (final) + 7 — Corrida final N alto, gráficos y `.pkl` — Juan
- **Estado:** ✅ completado (números finales)
- **Qué se hizo:** se subió N en el notebook (`N_RANDOM=100`, `N_SELF=100`, `N_STRAT=50`, `N_HEUR=30`), se extendió **E1 a d=1..4**, se reformuló **E3 para medir ambas técnicas vs Stratagem a d=2 y d=3** (juego parejo en profundidad), y se agregaron las celdas de **gráficos** (E1 nodos/tiempo, E2/E3 win rate, E5 win rate vs d, E6 heatmap → `plots/*.png`) y el guardado + recarga de **`mate_best_config.pkl`**. Re-ejecución headless completa (`jupyter nbconvert --execute --inplace`), ~10 min, **sin errores**. `results.csv` quedó con **878 filas** (E1=48, E2=200, E3=200, E4=100, E5=150, E6=180).
- **Archivos tocados:** `isolation.ipynb` (reconstruido limpio: 8 demos dadas + 24 celdas de experimentos/gráficos/pkl), `results.csv`, `plots/` (4 PNG), `mate_best_config.pkl` (todos generados). No se tocó ningún archivo del simulador.
- **Resultados finales:**
  - **E1 (Alpha-Beta):** reducción de nodos que **crece con la profundidad**: d2 **80%**, d3 **74%**, **d4 91%** (13.785 → 1.235 nodos). Tiempo/jugada a d4: 0.30 s (minimax) vs 0.08 s (AB). ✓ impacto contundente.
  - **E2 (vs Random, N=100):** Minimax **98%**, Expectimax **94%**. ✓ ambos dominan.
  - **E3 (vs Stratagem, N=50):** **d=2:** Expectimax 42% > Minimax 32%. **d=3:** **Minimax 38% > Expectimax 24%**. La técnica que conviene **depende de la profundidad**.
  - **E4 (Minimax vs Expectimax directo, N=100):** 48% / 52% — virtualmente parejo a d=2.
  - **E5 (profundidad vs Stratagem):** Minimax sube monótono con d: **28% → 32% → 38%** (d=1,2,3). ✓ buscar más hondo ayuda.
  - **E6 (torneo de heurísticas):** **`solo_mov_diff` (solo h2) gana** (0.667 prom.); `mov+acorralar` la peor (0.311). La diferencia de movilidad sola es la señal más fuerte.
  - **`mate_best_config.pkl`:** `{tecnica: minimax, profundidad: 3, pesos: {h2:1.0} (solo_mov_diff), métricas: {win_vs_stratagem_d3: 0.38, win_vs_random: 0.98, e6_winrate: 0.667}}`. Se recarga OK.
- **✅ HALLAZGO RESUELTO (cierra la nota de advertencia del Paso 6 pasada rápida):** la inversión Expectimax > Minimax vs Stratagem era un **artefacto de profundidad**. A **profundidad igualada (d=3, como Stratagem)** la **hipótesis original se confirma**: Minimax (38%) > Expectimax (24%). Más aún, profundizar **empeora** a Expectimax vs un rival determinista (42% → 24%) y **mejora** a Minimax (32% → 38%) — evidencia limpia de que el modelo de rival uniforme de Expectimax es incorrecto frente a Stratagem (DocumentacionMATE §3.3). Conclusión para el informe: *"¿cuál técnica es mejor?"* → **depende del oponente y de la profundidad**; vs estocástico (Random) ambos dominan; vs determinista (Stratagem) **a igual profundidad gana Minimax**.
- **Decisiones / desvíos respecto del plan:**
  - **E3 a dos profundidades** (en vez de una): decisión acordada para distinguir el efecto de la profundidad del efecto de la técnica. Fue clave para resolver el hallazgo.
  - **Reconstrucción limpia del notebook:** un primer intento de modificación falló porque el matching por `"BASE_SEED"` pisó la celda de helpers (que usa `base_seed=BASE_SEED`), dejando `make_board` indefinido. Se restauró desde git y se reconstruyó con un único builder que **solo hace append** (sin matching). Lección: editar celdas de notebook por contenido es frágil; preferir append o ids.
- **Pendiente / próximos pasos:** Paso 8 — redactar la **sección MATE del informe PDF** apoyándose en `DocumentacionMATE.md` (que ya incorpora los resultados) y los `plots/`. El notebook ya corre de punta a punta (equivale a "Restart & Run All").

### [2026-06-01] Paso 6 — Experimentos E1–E6 en el notebook (PASADA RÁPIDA) — Juan
- **Estado:** ✅ completado (pipeline validado con N chico; números finales pendientes de subir N)
- **Qué se hizo:** se agregaron al `isolation.ipynb` dado **17 celdas** (sin tocar las 8 demos previas) que implementan los experimentos E1–E6 con un framework reproducible:
  - **celda de configuración** con todas las constantes (`N_RANDOM`, `N_STRAT`, `DEPTH`, `AB_DEPTHS`, …) para escalar fácil;
  - **helpers** `run_matchup` (alterna quién arranca, siembra seed por partida, registra win/plies/tiempo/nodos), `summarize`, `make_board`, y fábricas de agentes (`mk_minimax`, `mk_expectimax`, `mk_random`, `mk_stratagem`);
  - **E1** impacto Alpha-Beta (nodos/tiempo con vs sin poda, d=1..3); **E2** vs Random; **E3** vs Stratagem; **E4** Minimax vs Expectimax; **E5** barrido de profundidad vs Stratagem; **E6** torneo de heurísticas;
  - registro completo a **`results.csv`** (236 filas, columna `experiment` para filtrar).
- **Archivos tocados:** `isolation.ipynb` (se le agregaron celdas), `results.csv` (nuevo, generado). **Cambio de entorno:** se agregaron **`pandas` y `matplotlib`** a `pyproject.toml` (necesarios para tablas/CSV y los gráficos que pide la consigna) vía `poetry add "pandas@^2.2.0" "matplotlib@^3.9.0"` (pandas 3.x exige Python ≥3.11; el proyecto está en ~3.10). No se modificó ningún archivo del simulador.
- **Cómo se verificó:** `poetry run jupyter nbconvert --to notebook --execute --inplace isolation.ipynb` corrió **de punta a punta sin errores** en ~2 min y dejó `results.csv` reproducible (seeds fijas).
- **Resultados de la pasada rápida (PRELIMINARES, N chico):**
  - **E1:** Alpha-Beta reduce nodos ~**80% a d=2** y ~**74% a d=3** (a d=1 no hay poda). ✓ confirma el impacto esperado.
  - **E2:** Minimax **100%** y Expectimax **96.7%** vs Random (N=30). ✓ ambos dominan.
  - **E3 (¡sorpresa!):** **Expectimax 50% vs Minimax 16.7%** contra Stratagem (N=12). Esto **contradice la hipótesis a priori** (DocumentacionMATE §3.3, que predecía Minimax ≥ Expectimax vs un rival determinista).
  - **E4:** Minimax 45% (⇒ Expectimax 55%) en enfrentamiento directo (N=20) — leve ventaja Expectimax, consistente con E3.
  - **E5:** win rate **idéntico (16.7%) a d=1,2,3** vs Stratagem — casi seguro artefacto de N=12 + determinismo (mismas seeds ⇒ se ganan las mismas 2/12 partidas); no concluir nada hasta subir N.
  - **E6:** la heurística **`solo_mov_diff` (solo h2) gana más** (0.767 prom.), `mov+acorralar` la peor (0.233). Sugiere que la movilidad sola es muy fuerte; a confirmar con N mayor.
- **Decisiones / desvíos respecto del plan:**
  - **Pasada rápida primero** (decisión acordada con el equipo): validar el pipeline con N chico antes de gastar cómputo en N≥100. Las constantes están parametrizadas para subir N de una.
  - **`pandas`+`matplotlib` agregados al entorno** (ver arriba): cambio necesario y acotado, justificado por el registro CSV y el apoyo visual exigido.
- **⚠️ HALLAZGO A INVESTIGAR / NOTA DE ADVERTENCIA:** el resultado E3/E4 (Expectimax ≥ Minimax **vs Stratagem**) es el **opuesto** a lo que documentamos como hipótesis. No es necesariamente un bug: nuestro Minimax asume que el rival minimiza **nuestra** heurística, pero Stratagem maximiza **la suya** (heurística distinta), así que el supuesto adversarial de Minimax también es "incorrecto" para este rival concreto, y a profundidad 2 (< 3 de Stratagem) puede caer en líneas defensivas subóptimas. **Antes de escribir conclusiones en el informe hay que confirmarlo con N≥100** (y, idealmente, igualando profundidad con Stratagem). Si se confirma, la conclusión del informe se ajusta a la evidencia (no al revés).
- **Pendiente / próximos pasos:** (a) subir N (≥100 vs Random/self, ≥50 vs Stratagem) para números finales y confirmar/ajustar el hallazgo E3/E4; (b) Paso 7 — gráficos (E1 nodos/tiempo vs profundidad, win rate por matchup, win rate vs d, heatmap de E6) + guardar `mate_best_config.pkl`.

### [2026-06-01] Paso 5 — ExpectimaxAgent (nodos de azar) — Juan
- **Estado:** ✅ completado
- **Qué se hizo:** se creó `Isolation/expectimax_agent.py` con la clase `ExpectimaxAgent(Agent)`. Misma estructura que Minimax pero los nodos del **rival** son **nodos de azar**: `Σ σ(s,a)·V(Suc(s,a))` con `σ` **uniforme** (1/n sobre las n acciones legales), según la lám. 8 del teórico. El agente sigue maximizando en sus nodos. Acepta la misma `eval_fn` que `MinimaxAgent` (default = `h2_mobility_diff`) y la misma instrumentación de nodos (`nodes_last_move`/`total_nodes`) para comparar costo.
- **Archivos tocados:** `Isolation/expectimax_agent.py` (nuevo). **No** se modificó ningún archivo dado.
- **Decisiones / desvíos respecto del plan:**
  - **Sin Alpha-Beta en Expectimax:** los nodos de azar promedian todas las ramas (no hay corte por cota como en MIN), por eso esta clase no expone el flag de poda. Documentado en el docstring.
  - **Modelo de rival σ uniforme:** es el modelo correcto para `RandomAgent` e incorrecto para `Stratagem` (determinista). Esta es justamente la hipótesis que medirán E2–E4; si Expectimax rinde peor que Minimax vs Stratagem, **no es un bug** (DocumentacionMATE §3.3, §5).
  - **Hallazgo de integración:** `Stratagem` tiene el nombre de su parámetro `__init__` **ofuscado**, así que hay que instanciarlo **posicional** (`Stratagem(2)`), no con `player=2`. Anotado en el smoke test para que los experimentos del Paso 6 no tropiecen con esto.
- **Cómo se verificó:** `poetry run python expectimax_agent.py`. (1) `ExpectimaxAgent(d=2)` vs `RandomAgent`, N=30 alternando lados → **29/30 = 97% win rate**. (2) Corre **sin errores** frente a `Stratagem` (2 partidas). Salió `OK Paso 5`. (Nota: el tiempo/jugada vs Stratagem lo domina el propio Minimax d=3 de Stratagem, ~0.24 s/jugada.)
- **Pendiente / próximos pasos:** Paso 6 — experimentos E1–E6 en `isolation.ipynb` usando `match.py` + los agentes, con registro a `results.csv` (N≥100, lados alternados, seeds).

### [2026-06-01] Paso 4 — Funciones de evaluación (h1–h4 + weighted_eval) — Juan
- **Estado:** ✅ completado
- **Qué se hizo:** se creó `Isolation/evaluation.py` con las cuatro componentes heurísticas, todas con firma `(board, player) -> float` y perspectiva del jugador:
  - `h1_mobility` — casillas adyacentes libres del jugador (movilidad propia);
  - `h2_mobility_diff` — movilidad propia − rival (suma cero);
  - `h3_center` — − distancia Manhattan al centro (= `Stratagem.Y`);
  - `h4_surround` — celdas destruidas alrededor del rival − alrededor propio (= `Stratagem.X`).

  Más `weighted_eval(weights)` que devuelve una `eval_fn(board, player)` combinada `Σ wᵢ·hᵢ` (valida claves; ignora pesos 0), inyectable a los agentes vía `eval_fn`, y un registro `HEURISTICS` para los experimentos del Paso 6. La movilidad se mide con `free_adjacent` (casillas adyacentes libres), **no** con `len(get_possible_actions)`.
- **Archivos tocados:** `Isolation/evaluation.py` (nuevo) y **refactor** de `Isolation/minimax_agent.py`. **No** se modificó ningún archivo dado.
- **Decisiones / desvíos respecto del plan:**
  - **Centralización de `free_adjacent`:** la primitiva de movilidad y la default del agente se movieron a `evaluation.py` (módulo de heurísticas, su lugar natural). `minimax_agent.py` ahora importa `h2_mobility_diff` de ahí y lo usa como default, **eliminando la duplicación** que tenía el Paso 2. Dependencia `minimax_agent → evaluation` (sin ciclos). El comportamiento del agente es idéntico (su default sigue siendo la diferencia de movilidad).
  - **Componentes alineadas con `Stratagem`** (h3=Y, h4=X) para tener un punto de comparación honesto y poder buscar ponderaciones que lo superen (consigna 2). h1/h2 cubren la señal de movilidad, la más correlacionada con ganar en Isolation.
- **Cómo se verificó:** `poetry run python evaluation.py` con boards de juguete de valor esperado conocido: centro (2,2) puntúa más que esquina en h3; movilidad 7 vs 2 → `h2_mobility_diff=±5`; rival rodeado por 3 destruidas → `h4_surround=±3`; `weighted_eval` lineal y respeta pesos; rechaza heurísticas desconocidas. Salió `OK Paso 4`. Además se **re-verificó `minimax_agent.py`** tras el refactor: sigue dando 97% vs Random y 83% de poda (sin cambios).
- **Pendiente / próximos pasos:** Paso 5 — `ExpectimaxAgent` (nodos de azar con σ uniforme del rival), reutilizando `search.py` y aceptando la misma `eval_fn`.

### [2026-06-01] Paso 3 — Alpha-Beta + ordenamiento de movimientos — Juan
- **Estado:** ✅ completado
- **Qué se hizo:** se agregó la poda Alpha-Beta a `minimax_agent.py` como un **flag** (`use_alpha_beta`) sobre el mismo núcleo de búsqueda. `next_action` enruta a `_alphabeta(...)` o `_minimax(...)` según el flag. La poda usa la ventana `(alpha, beta)` estándar (corte β en nodos MAX, corte α en nodos MIN). Se añadió **ordenamiento de movimientos** (`_ordered_successors`, flag `move_ordering=True` por defecto): en nodos MAX explora primero los sucesores de mayor evaluación y en nodos MIN los de menor, para maximizar la poda. El conteo de nodos usa el mismo `tick()` por llamada que `_minimax`, lo que hace la comparación directa.
- **Archivos tocados:** `Isolation/minimax_agent.py` (mismo archivo del Paso 2; se quitó el `NotImplementedError` y se implementó la poda). **No** se modificó ningún archivo dado.
- **Decisiones / desvíos respecto del plan:**
  - **Ordenamiento por evaluación del hijo** (no por un orden fijo): es barato y no afecta la corrección (el valor final es idéntico), solo aumenta la poda. Documentado que el orden no cambia el valor, solo el costo.
  - **Por qué la prueba de equivalencia compara VALOR, no siempre la acción:** a igual profundidad y estado, Alpha-Beta garantiza el **mismo valor** que Minimax, pero ante **empates de valor** el ordenamiento puede elegir otra acción igualmente óptima. Por eso: (a) se exige `valor(AB) == valor(Minimax)` siempre, y (b) se exige `acción(AB) == acción(Minimax)` **solo con `move_ordering=False`** (mismo orden de sucesores y mismo desempate → misma acción). En la raíz no hay corte (β=+∞), así que con orden desactivado la acción es idéntica por construcción.
  - **`nodos(AB) ≤ nodos(Minimax)` siempre:** Minimax recorre el árbol completo; Alpha-Beta visita un subconjunto. La desigualdad vale para cualquier ordenamiento.
- **Cómo se verificó:** `poetry run python minimax_agent.py`. (1) Paso 2 sigue OK (97% vs Random). (2) **Equivalencia** en 7 estados (2–4 aperturas aleatorias, d=3): mismo valor en todos; `acción(AB sin orden) == acción(Minimax)` en todos; **nodos Minimax = 37.988 vs Alpha-Beta = 6.554 → poda del 83%**. Salió `OK Paso 3`.
- **Pendiente / próximos pasos:** Paso 4 — `evaluation.py` con las heurísticas h1–h4 + `weighted_eval(weights)` inyectable como `eval_fn` a los agentes. (El experimento E1 del Paso 6 medirá nodos/tiempo con vs sin poda a varias profundidades de forma sistemática.)

### [2026-06-01] Paso 2 — MinimaxAgent (profundidad fija, sin poda) — Juan
- **Estado:** ✅ completado
- **Qué se hizo:** se creó `Isolation/minimax_agent.py` con la clase `MinimaxAgent(Agent)` que implementa `V_max,min(s, d)` del teórico (lám. 13) **sin poda**, apoyándose en `search.py` (`successors`, `is_terminal`, `utility`, `other`, `NodeCounter`). El método recursivo `_minimax(board, player_to_move, depth)` devuelve `(mejor_acción, valor)` (misma estructura que `Stratagem`); `next_action` elige el `argmax` en la raíz. Se instrumentó el conteo de nodos (`nodes_last_move` y `total_nodes`) para el análisis de Alpha-Beta del Paso 3 (E1). Se agregó una heurística por defecto (diferencia de movilidad) vía `free_adjacent`, inyectable: `eval_fn(board, player)` reemplaza la default cuando se pase (Paso 4).
- **Archivos tocados:** `Isolation/minimax_agent.py` (nuevo). **No** se modificó ningún archivo dado.
- **Decisiones / desvíos respecto del plan:**
  - **Flag `use_alpha_beta` honesto:** está cableado desde el Paso 2 pero, si se pide `True`, levanta `NotImplementedError("Alpha-Beta se implementa en el Paso 3")` en vez de ignorarlo silenciosamente. Así cada commit refleja exactamente lo implementado.
  - **Conteo de nodos = nodos visitados** (un `tick()` al entrar a cada llamada de `_minimax`, incluyendo terminales y hojas de corte). Es una definición simple y consistente que hará evidente la poda en el Paso 3 (AB visita estrictamente menos nodos).
  - **Heurística por defecto provisional** en el agente (diferencia de movilidad) para que el agente sea funcional sin depender aún de `evaluation.py`; la biblioteca h1–h4 del Paso 4 se inyecta por `eval_fn` sin tocar esta clase.
- **Cómo se verificó:** `poetry run python minimax_agent.py`. `MinimaxAgent(d=2)` vs `RandomAgent`, N=30 alternando quién arranca (15 cada lado, seeds 0–29) → **29/30 = 97% win rate**, sin excepciones de acción inválida, en ~2 s. Salió `OK Paso 2`.
- **Pendiente / próximos pasos:** Paso 3 — agregar poda Alpha-Beta (con ordenamiento de movimientos) bajo el flag `use_alpha_beta`, y test de equivalencia: a igual profundidad y seed, `argmax(AB) == argmax(Minimax)` con `nodos(AB) ≤ nodos(Minimax)`.

### [2026-06-01] Paso 1 — Núcleo de búsqueda — Martín
- **Estado:** ✅ completado
- **Qué se hizo:** se creó `Isolation/search.py` con `successors(board, player)`, `is_terminal(board, player_to_move)`, `utility(winner, agent_player)` (±1, sin empates), `other(player)` y la clase `NodeCounter` (para medir poda en Alpha-Beta).
- **Archivos tocados:** `Isolation/search.py` (nuevo).
- **Decisiones / desvíos respecto del plan:** ninguno.
- **Cómo se verificó:** `poetry run python search.py`. (1) `successors` devuelve **91** sucesores en apertura (confirma branching factor ~100/ply) y no muta el tablero original; (2) tablero con jugador 1 acorralado en esquina → `is_terminal` da `(True, 2)` y `utility` da −1/+1 según perspectiva; (3) `other` OK. Salió `OK Paso 1`.
- **Pendiente / próximos pasos:** Paso 2 — `MinimaxAgent` (profundidad fija, sin poda) apoyándose en `search.py`.

### [2026-06-01] Paso 0 — Setup y utilidades base — Martín
- **Estado:** ✅ completado
- **Qué se hizo:** se creó `Isolation/match.py` con `play_match(agent_p1, agent_p2, seed, render)`, que corre una partida y **devuelve** `{winner, plies, avg_move_time, seed}` (a diferencia de `play.py`, que solo imprime). Siembra `random.seed(seed)` para reproducibilidad.
- **Archivos tocados:** `Isolation/match.py` (nuevo). **No** se modificó ningún archivo dado.
- **Decisiones / desvíos respecto del plan:** se descartó el parche a `Board.clone()`: en 4×4 funciona bien y, con `seed`, la ejecución es determinista igual. (Plan §1, T0 y Paso 0 actualizados acorde.)
- **Cómo se verificó:** `poetry install` + `poetry run python match.py`. Dos corridas `RandomAgent(1)` vs `RandomAgent(2)` con `seed=42` dieron resultado idéntico (`winner=2, plies=4`). Salió `OK Paso 0`.
- **Pendiente / próximos pasos:** Paso 1 — `search.py` (`successors`, `is_terminal`, `utility`, `NodeCounter`).

---

## Dónde retomar

- **Próximo paso a ejecutar:** (a) **re-commit** de los artefactos regenerados tras aplicar las mejoras (`match.py`, `isolation.ipynb`, `results.csv`, `plots/`, docs MATE); (b) **Paso 8 — redactar la sección MATE del informe PDF** (≤ 8–9 págs.) apoyándose en `DocumentacionMATE.md` (ya con los resultados del diseño apareado) y los gráficos de `plots/`. El código, el notebook (corre de punta a punta), `results.csv`, `plots/` y `mate_best_config.pkl` ya están listos.
- **Listo para usar (TODO el pipeline):** `match.py`, `search.py`, `minimax_agent.py`, `expectimax_agent.py`, `evaluation.py`. `isolation.ipynb` corre E1–E6 + gráficos + `.pkl` de punta a punta (`poetry run jupyter nbconvert --to notebook --execute --inplace isolation.ipynb`). Artefactos: `results.csv` (1568 filas, diseño apareado), `plots/*.png` (4), `mate_best_config.pkl`. Recordar: `Stratagem` se instancia **posicional**. Entorno: `poetry install --no-root` (+ `pandas`/`matplotlib`).
- **Bloqueos actuales:** ninguno. El hallazgo E3 **quedó resuelto** (artefacto de profundidad; la hipótesis se confirma a d=3) — ver entrada de bitácora de Pasos 6/7 final.
- **Notas para el equipo:** seguir el orden de `PlanificacionMATE.md` §3 (Minimax sin poda → Alpha-Beta → heurísticas en paralelo → Expectimax → experimentos). Recordar sembrar `random` por partida y alternar lados (ver §4 del plan). Correr scripts con `poetry run python <archivo>` desde `Isolation/`.
