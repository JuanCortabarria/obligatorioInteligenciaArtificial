"""Reward shaping como EXTRA (Ng-Harada-Russell 1999) — experimento con datos reales.

El shaping NO forma parte del núcleo (modificar la recompensa equivale a cambiar el
ambiente; la cátedra lo permite solo como adicional). Acá lo evaluamos de forma honesta
para responder dos preguntas concretas, con varias semillas y análisis de varianza:

  1. ¿El shaping **rescata** al Q-Learning vanilla (opt=0), que solo da 0 %?
  2. ¿El shaping **acelera** al agente que ya resuelve por inicialización optimista (opt=10)?

Compara 4 variantes × 5 seeds (recompensa real para medir; el shaping solo entra en el
TD target durante el entrenamiento):

  - vanilla (opt=0)            : Q-Learning estándar — la "trampa", 0 %.
  - shaping (opt=0)            : ¿el shaping solo destraba la exploración?
  - optimista (opt=10) ⭐       : nuestro enfoque del núcleo (sin tocar la recompensa).
  - optimista+shaping (opt=10) : ¿sumar shaping acelera la convergencia?

Salidas: `shaping_comparison.json` + `plots/shaping_curves.png` + `plots/shaping_box_conv.png`.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd

import experiments as ex

HERE = Path(__file__).parent
EPISODES = 1500   # suficiente para ver convergencia (el optimista resuelve por ~350-520)

VARIANTS = [
    ("vanilla (opt=0)",            {"optimistic_init": 0.0, "reward_shaping": False}),
    ("shaping (opt=0)",            {"optimistic_init": 0.0, "reward_shaping": True}),
    ("optimista (opt=10)",         {"optimistic_init": 10.0, "reward_shaping": False}),
    ("optimista+shaping (opt=10)", {"optimistic_init": 10.0, "reward_shaping": True}),
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
    finals.to_json(HERE / "shaping_comparison.json", orient="records", indent=2)

    ex.plot_curves_band(
        curves, HERE / "plots" / "shaping_curves.png",
        title="Reward shaping (EXTRA): el shaping rescata al vanilla y acelera al optimista — recompensa real, 5 seeds",
    )
    ex.plot_box(
        finals, "conv_ep", HERE / "plots" / "shaping_box_conv.png",
        title="Episodio de convergencia por variante (5 seeds)",
        ylabel="Episodio de convergencia (éxito móvil ≥ 90%)",
    )

    print("\n=== Resumen (mediana / varianza sobre 5 seeds) ===")
    print(ex.summary(finals).to_string(index=False))
    print("\nGuardado: shaping_comparison.json + plots/shaping_curves.png + plots/shaping_box_conv.png")


if __name__ == "__main__":
    main()
