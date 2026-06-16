"""Comparación Dyna-Q vs Q-Learning sobre MountainCarContinuous-v0 (recompensa REAL).

Dyna-Q (Sutton & Barto §8.2) = Q-Learning + un modelo aprendido del ambiente + `n` pasos
de *planning* (actualizaciones simuladas) por cada paso real. La hipótesis del libro es
que el planning permite **converger en menos episodios reales** (amortiza cada experiencia),
a costa de más cómputo por episodio.

Se compara `planning_steps ∈ {0, 5, 25}` (0 = Q-Learning puro) sobre la **config base que
funciona** (inicialización optimista, sin shaping), variando SOLO el planning para aislar
su efecto. Cada variante se corre con **varias semillas** (`experiments.SEEDS`) para medir
la varianza; comparación **apareada** (mismas semillas para todas).

Salidas:
  dyna_q_comparison.json             — todas las corridas (planning × seed).
  plots/dyna_q_box_convergence.png   — boxplot del episodio de convergencia por planning.
  plots/dyna_q_box_steps.png         — boxplot de pasos en test por planning.
  plots/dyna_q_curves.png            — curvas de aprendizaje con banda de error.
"""

from pathlib import Path

import pandas as pd

import experiments as ex

HERE = Path(__file__).parent
PLOTS = HERE / "plots"
EPISODES = 1000          # presupuesto donde se distingue la velocidad de convergencia
PLANNING = [0, 5, 25]    # 0 = Q-Learning puro


def main():
    PLOTS.mkdir(exist_ok=True)
    all_curves, all_finals = [], []

    for n in PLANNING:
        name = "Q-Learning (n=0)" if n == 0 else f"Dyna-Q (n={n})"
        print(f"== {name} ==")
        curves, finals = ex.run_config(name, {"planning_steps": n}, episodes=EPISODES)
        all_curves.append(curves)
        all_finals.append(finals)
        s = ex.summary(finals).iloc[0]
        print(f"   test_succ(mediana)={s['test_success_mediana']:.0%}  "
              f"steps(mediana)={s['test_steps_mediana']:.0f}  "
              f"conv(mediana)={s['conv_ep_mediana']:.0f}  "
              f"tiempo(mediana)={finals['train_time_s'].median():.0f}s")

    curves = pd.concat(all_curves, ignore_index=True)
    finals = pd.concat(all_finals, ignore_index=True)
    finals.to_json(HERE / "dyna_q_comparison.json", orient="records", indent=2)

    print("\n=== Resumen (mediana sobre seeds) ===")
    print(ex.summary(finals).to_string(index=False))
    print("\nTiempo de entrenamiento (mediana, s) por variante:")
    print(finals.groupby("config")["train_time_s"].median().to_string())

    ex.plot_box(finals, "conv_ep", PLOTS / "dyna_q_box_convergence.png",
                "Dyna-Q vs Q-Learning — episodio de convergencia (5 seeds)",
                "Episodio de convergencia (menor = converge antes)")
    ex.plot_box(finals, "test_steps", PLOTS / "dyna_q_box_steps.png",
                "Dyna-Q vs Q-Learning — pasos en test (5 seeds)", "Pasos promedio (menor = mejor)")
    ex.plot_curves_band(curves, PLOTS / "dyna_q_curves.png",
                        "Dyna-Q vs Q-Learning — curvas de aprendizaje (banda de error)")

    print(f"\nGuardado: dyna_q_comparison.json + 3 gráficos en {PLOTS}")


if __name__ == "__main__":
    main()
