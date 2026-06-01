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
| 0 | Setup: parche `clone()` + `play_match()` con seed/swap/stats | `board.py`, `match.py` | ⬜ | — | — |
| 1 | Núcleo de búsqueda (sucesores, nodos, utilidad terminal) | `search.py` | ⬜ | — | — |
| 2 | MinimaxAgent profundidad fija (sin poda) | `minimax_agent.py` | ⬜ | — | — |
| 3 | Alpha-Beta + ordenamiento de movimientos | `minimax_agent.py` | ⬜ | — | — |
| 4 | Funciones de evaluación h1–h4 + `weighted_eval` | `evaluation.py` | ⬜ | — | — |
| 5 | ExpectimaxAgent (nodos de azar) | `expectimax_agent.py` | ⬜ | — | — |
| 6 | Framework de experimentos + logging CSV/JSON | `experiments.py` | ⬜ | — | — |
| 7 | Correr experimentos E1–E6 (N ≥ 100) | `results/` | ⬜ | — | — |
| 8 | Gráficos, tablas y modelos `.pkl` | `persistence.py`, `models/`, `mate.ipynb` | ⬜ | — | — |
| 9 | Notebook entregable + sección MATE del informe | `mate.ipynb`, informe PDF | ⬜ | — | — |

**Progreso:** 0 / 10 pasos completados.

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

### [—] Aún no hay avances registrados
- El proyecto está en estado inicial. La planificación (`PlanificacionMATE.md`) y la
  documentación de decisiones (`DocumentacionMATE.md`) están listas; falta arrancar el Paso 0.

---

## Dónde retomar

- **Próximo paso a ejecutar:** **Paso 0 — Setup y utilidades base** (`board.py` clone + `match.py`).
- **Bloqueos actuales:** ninguno.
- **Notas para el equipo:** seguir el orden de `PlanificacionMATE.md` §3 (Minimax sin poda → Alpha-Beta → heurísticas en paralelo → Expectimax → experimentos). Recordar sembrar `random` por partida y alternar lados (ver §4 del plan).
