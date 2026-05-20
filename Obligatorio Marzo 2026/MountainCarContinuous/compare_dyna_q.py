"""
Comparación experimental Q-Learning vs Dyna-Q con distintos planning_steps.

Diseño del experimento (alineado con la consigna: "realizar un análisis y
experimentación similar a su trabajo con Q-Learning"):

  - Misma config base que el mejor Q-Learning del grid search
    (bins=20, n_actions=3, α=0.1, γ=0.99, ε_decay=0.995, shaping potential-based coef=300).
  - Variamos planning_steps n ∈ {0, 5, 25, 50}.
    n=0 ≡ Q-Learning puro (sanity check: debe reproducir resultados de §2.4).
  - 400 episodios por config, mismo seed (42) para reproducibilidad.

Hipótesis del libro (Sutton & Barto §8.2): Dyna-Q converge en **menos episodios
reales** que Q-Learning porque cada transición observada se amortiza en n updates
simulados. El costo: tiempo *por episodio* sube ~lineal en n.

Métricas:
  - convergence_ep: primer episodio donde rolling-50 success ≥ 0.9.
  - test_success_rate / test_avg_reward / test_avg_steps (50 ep greedy).
  - train_time_s: tiempo total de entrenamiento.
  - |model|: tamaño final del modelo aprendido (estados-acción únicos visitados).

Salidas:
  plots/dyna_q_comparison.png        — curvas de aprendizaje superpuestas.
  plots/dyna_q_convergence_vs_time.png — barras: convergence_ep y train_time por n.
  models/dyna_q_best.pkl             — el mejor agente Dyna-Q (entregable).
  dyna_q_comparison.json             — resultados crudos.
"""

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym

from discretization import Discretizer
from dyna_q_agent import DynaQAgent


HERE = Path(__file__).parent
PLOTS_DIR = HERE / "plots"
MODELS_DIR = HERE / "models"

PLANNING_VALUES = [0, 5, 25, 50]
EPISODES = 400
SEED = 42
CONVERGENCE_WINDOW = 50
CONVERGENCE_THRESHOLD = 0.9


def run_one(n: int) -> tuple[dict, dict, DynaQAgent]:
    env = gym.make("MountainCarContinuous-v0")
    disc = Discretizer(n_bins_x=20, n_bins_v=20, n_actions=3)
    agent = DynaQAgent(
        discretizer=disc,
        planning_steps=n,
        alpha=0.1,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        reward_shaping=True,
        shaping_coef=300.0,
        seed=SEED,
    )
    t0 = time.time()
    history = agent.train_agent(
        env, episodes=EPISODES, verbose_every=0, env_seed=SEED
    )
    train_time = time.time() - t0

    successes = np.array(history["success"])
    rolling = np.convolve(
        successes, np.ones(CONVERGENCE_WINDOW) / CONVERGENCE_WINDOW, mode="valid"
    )
    above = np.where(rolling >= CONVERGENCE_THRESHOLD)[0]
    convergence_ep = int(above[0] + CONVERGENCE_WINDOW - 1) if len(above) else None

    test_metrics = agent.test_agent(env, episodes=50)
    env.close()

    result = {
        "planning_steps": n,
        "train_time_s": round(train_time, 2),
        "train_success_rate_last100": float(np.mean(successes[-100:])),
        "train_avg_reward_last100": float(np.mean(history["rewards"][-100:])),
        "convergence_ep": convergence_ep,
        "test_success_rate": test_metrics["success_rate"],
        "test_avg_reward": test_metrics["avg_reward"],
        "test_avg_steps": test_metrics["avg_steps"],
        "model_size": len(agent.model),
    }
    return result, history, agent


def plot_curves(histories: dict, out_path: Path):
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    window = 50

    for n, hist in histories.items():
        rewards = np.array(hist["rewards"])
        successes = np.array(hist["success"])
        ma_r = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ma_s = np.convolve(successes, np.ones(window) / window, mode="valid")
        xs = np.arange(len(ma_r)) + window - 1
        label = f"n={n}" + (" (Q-Learning)" if n == 0 else "")
        axes[0].plot(xs, ma_r, label=label, linewidth=1.5)
        axes[1].plot(xs, ma_s, label=label, linewidth=1.5)

    axes[0].set_ylabel(f"Reward (media móvil {window})")
    axes[0].set_title("Dyna-Q vs Q-Learning — curvas de aprendizaje (eje X = episodios reales)")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].set_ylabel(f"Tasa de éxito (ventana {window})")
    axes[1].set_xlabel("Episodio")
    axes[1].set_ylim(-0.05, 1.05); axes[1].grid(alpha=0.3); axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_convergence_and_time(results: list[dict], out_path: Path):
    ns = [r["planning_steps"] for r in results]
    conv = [r["convergence_ep"] if r["convergence_ep"] is not None else 0 for r in results]
    times = [r["train_time_s"] for r in results]
    steps = [r["test_avg_steps"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].bar([str(n) for n in ns], conv, color="C0")
    axes[0].set_xlabel("planning_steps n"); axes[0].set_ylabel("Convergencia (episodio reales)")
    axes[0].set_title("¿Cuántos episodios reales hasta 90% éxito?")
    axes[0].grid(alpha=0.3, axis="y")

    axes[1].bar([str(n) for n in ns], times, color="C3")
    axes[1].set_xlabel("planning_steps n"); axes[1].set_ylabel("Tiempo total (s)")
    axes[1].set_title("Costo computacional (400 episodios)")
    axes[1].grid(alpha=0.3, axis="y")

    axes[2].bar([str(n) for n in ns], steps, color="C2")
    axes[2].set_xlabel("planning_steps n"); axes[2].set_ylabel("Steps test (greedy)")
    axes[2].set_title("Calidad final de la política")
    axes[2].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    PLOTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    results = []
    histories = {}
    agents = {}

    for n in PLANNING_VALUES:
        print(f"\n=== planning_steps = {n} ===")
        result, history, agent = run_one(n)
        results.append(result)
        histories[n] = history
        agents[n] = agent
        print(
            f"  done in {result['train_time_s']}s | "
            f"conv@{result['convergence_ep']} | "
            f"test_succ={result['test_success_rate']:.0%} | "
            f"test_reward={result['test_avg_reward']:.2f} | "
            f"test_steps={result['test_avg_steps']:.1f} | "
            f"|model|={result['model_size']}"
        )

    # Criterio para "mejor Dyna-Q":
    #   1. Mayor test_success_rate.
    #   2. Menor test_avg_steps (política más eficiente — consistente con el
    #      criterio del grid search de Q-Learning).
    #   3. Como desempate, menor convergence_ep.
    # No queremos seleccionar n=0 (eso es Q-Learning puro, no Dyna-Q).
    def score(r):
        return (
            -r["test_success_rate"],
            r["test_avg_steps"],
            r["convergence_ep"] if r["convergence_ep"] is not None else 10_000,
        )

    dyna_only = [r for r in results if r["planning_steps"] > 0]
    sorted_results = sorted(dyna_only, key=score)
    best = sorted_results[0]
    best_n = best["planning_steps"]
    agents[best_n].save(MODELS_DIR / "dyna_q_best.pkl")
    print(f"\nMejor Dyna-Q: n={best_n} → guardado en {MODELS_DIR / 'dyna_q_best.pkl'}")

    plot_curves(histories, PLOTS_DIR / "dyna_q_comparison.png")
    plot_convergence_and_time(results, PLOTS_DIR / "dyna_q_convergence_vs_time.png")

    with open(HERE / "dyna_q_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResultados JSON: {HERE / 'dyna_q_comparison.json'}")


if __name__ == "__main__":
    main()
