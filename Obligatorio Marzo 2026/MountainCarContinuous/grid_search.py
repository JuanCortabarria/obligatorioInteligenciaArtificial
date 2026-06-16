"""Exploración de hiperparámetros para Q-Learning sobre MountainCarContinuous-v0.

Enfoque (según la devolución de la cátedra):
  - Se trabaja con la **recompensa REAL** del ambiente (sin reward shaping).
  - Se exploran **diversos** valores de α, γ, epsilon-decay, **discretizaciones** y la
    **inicialización optimista**, con estrategia one-at-a-time (OAT) desde una config base.
  - Cada configuración se corre con **varias semillas** (`experiments.SEEDS`) para
    **analizar la varianza**, y se reporta con **boxplots** y **bandas de error**.

La inicialización optimista es la dimensión clave: sin ella, Q-Learning cae en la
"trampa de no hacer nada" (0% de éxito) porque moverse cuesta energía y la meta casi
nunca se alcanza explorando. Con ella, Q-Learning resuelve el ambiente sin tocar la
recompensa (Sutton & Barto §2.6).

Salidas:
  grid_search_results.json          — todas las corridas (config × seed) para el informe.
  plots/grid_search_box_success.png — boxplot de éxito en test por config.
  plots/grid_search_box_steps.png   — boxplot de pasos en test por config.
  plots/grid_search_optinit_curves.png — curvas con banda de error del barrido de opt-init.
"""

import json
from pathlib import Path

import pandas as pd

import experiments as ex

HERE = Path(__file__).parent
PLOTS = HERE / "plots"
EPISODES = 1500

# OAT: cada entrada cambia UNA cosa respecto de la base (experiments.DEFAULT).
GRID = {
    "base (bins40, opt=10)":  {},
    "opt=0 (trampa)":         {"optimistic_init": 0.0},
    "opt=1":                  {"optimistic_init": 1.0},
    "opt=50":                 {"optimistic_init": 50.0},
    "bins20 (gruesa)":        {"bins": 20, "n_actions": 3},
    "bins60 (fina)":          {"bins": 60, "n_actions": 7},
    "alpha=0.05":             {"alpha": 0.05},
    "alpha=0.3":              {"alpha": 0.3},
    "gamma=0.95":             {"gamma": 0.95},
    "gamma=0.999":            {"gamma": 0.999},
    "decay=0.995":            {"epsilon_decay": 0.995},
}


def main():
    PLOTS.mkdir(exist_ok=True)
    all_curves, all_finals = [], []

    for name, overrides in GRID.items():
        print(f"== {name} ({overrides}) ==")
        curves, finals = ex.run_config(name, overrides, episodes=EPISODES)
        all_curves.append(curves)
        all_finals.append(finals)
        s = ex.summary(finals).iloc[0]
        print(f"   test_succ(mediana)={s['test_success_mediana']:.0%}  "
              f"reward(mediana)={s['test_reward_mediana']:.1f}  "
              f"steps(mediana)={s['test_steps_mediana']:.0f}  "
              f"conv(mediana)={s['conv_ep_mediana']:.0f}")

    curves = pd.concat(all_curves, ignore_index=True)
    finals = pd.concat(all_finals, ignore_index=True)

    # Registro completo (todas las corridas, config × seed).
    finals.to_json(HERE / "grid_search_results.json", orient="records", indent=2)

    # Selección ROBUSTA (no por mediana sola, que engañaría): se prioriza que TODAS las
    # seeds resuelvan (mayor `test_success_min`), luego MENOR varianza (`test_reward_std`),
    # y como desempate menos pasos. El análisis de varianza es justamente lo que distingue
    # una config "buena de casualidad" (alta mediana pero algún seed en 0%) de una robusta.
    summ = ex.summary(finals).sort_values(
        ["test_success_min", "test_reward_std", "test_steps_mediana"],
        ascending=[False, True, True])
    print("\n=== Resumen (mediana/varianza sobre seeds) ===")
    print(summ.to_string(index=False))
    best = summ.iloc[0]["config"]
    print(f"\nMejor configuración ROBUSTA (todas las seeds resuelven + menor varianza + menos pasos): {best}")

    by_median = ex.summary(finals).sort_values(
        ["test_success_mediana", "test_steps_mediana"], ascending=[False, True]).iloc[0]["config"]
    if by_median != best:
        print(f"Nota: elegir solo por MEDIANA habría seleccionado '{by_median}', "
              f"que sin embargo tiene mayor varianza (alguna seed falla). Por eso se mira la varianza.")

    # Boxplots (varianza entre seeds) + banda de error del barrido de opt-init.
    ex.plot_box(finals, "test_success", PLOTS / "grid_search_box_success.png",
                "Grid search — éxito en test por configuración (5 seeds)", "Tasa de éxito en test")
    ex.plot_box(finals, "test_steps", PLOTS / "grid_search_box_steps.png",
                "Grid search — pasos en test por configuración (5 seeds)", "Pasos promedio (menor = mejor)")
    opt_curves = curves[curves["config"].isin(
        ["opt=0 (trampa)", "opt=1", "base (bins40, opt=10)", "opt=50"])]
    ex.plot_curves_band(opt_curves, PLOTS / "grid_search_optinit_curves.png",
                        "Efecto de la inicialización optimista (curvas con banda de error)")

    print(f"\nGuardado: grid_search_results.json + 3 gráficos en {PLOTS}")


if __name__ == "__main__":
    main()
