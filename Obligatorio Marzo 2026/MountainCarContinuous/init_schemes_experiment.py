"""Comparación de esquemas de INICIALIZACIÓN de Q (responde a la devolución de la cátedra).

La cátedra aprobó la inicialización optimista y pidió: justificar por qué los valores
iniciales son **distintos de 0** y **no aleatorios**, y **probar también la alternativa
aleatoria** (que es "menos arbitraria" que elegir un valor optimista fijo).

IMPORTANTE (terminología): la estrategia de exploración es **ε-greedy en TODOS los casos**;
lo único que cambia entre variantes es el **valor inicial de la tabla Q**. Por eso el caso
de referencia se llama `init=0`, NO "ε-greedy puro" (ε-greedy no implica inicializar en 0).

Compara 4 esquemas × 5 seeds (recompensa real, sin shaping):
  - init=0            : Q arranca en 0 (la "trampa de no hacer nada").
  - aleatoria U(0,1)  : Q ~ Uniforme(0,1) — aleatoria de magnitud pequeña.
  - aleatoria U(0,20) : Q ~ Uniforme(0,20) — aleatoria a la escala del optimismo.
  - optimista=10      : Q constante = 10 (nuestra elección del núcleo).

Salidas: `init_schemes_comparison.json` + `plots/init_schemes_curves.png`
         + `plots/init_schemes_box_success.png`.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd

import experiments as ex

HERE = Path(__file__).parent
EPISODES = 1500

VARIANTS = [
    ("init=0",            {"optimistic_init": 0.0}),
    ("aleatoria U(0,1)",  {"optimistic_init": 0.0, "random_init": (0.0, 1.0)}),
    ("aleatoria U(0,20)", {"optimistic_init": 0.0, "random_init": (0.0, 20.0)}),
    ("optimista=10",      {"optimistic_init": 10.0}),
]


def main():
    (HERE / "plots").mkdir(exist_ok=True)
    all_curves, all_finals = [], []
    for name, ov in VARIANTS:
        print(f"== {name} ({ov}) ==")
        curves, finals = ex.run_config(name, ov, episodes=EPISODES)
        all_curves.append(curves)
        all_finals.append(finals)

    curves = pd.concat(all_curves, ignore_index=True)
    finals = pd.concat(all_finals, ignore_index=True)
    finals.to_json(HERE / "init_schemes_comparison.json", orient="records", indent=2)

    ex.plot_curves_band(
        curves, HERE / "plots" / "init_schemes_curves.png",
        title="Esquemas de inicialización de Q (misma ε-greedy, recompensa real, 5 seeds)",
    )
    ex.plot_box(
        finals, "test_success", HERE / "plots" / "init_schemes_box_success.png",
        title="Éxito en test por esquema de inicialización (5 seeds)",
        ylabel="Tasa de éxito en test",
    )

    print("\n=== Resumen (mediana / varianza sobre 5 seeds) ===")
    print(ex.summary(finals).to_string(index=False))
    print("\nGuardado: init_schemes_comparison.json + 2 gráficos en plots/")


if __name__ == "__main__":
    main()
