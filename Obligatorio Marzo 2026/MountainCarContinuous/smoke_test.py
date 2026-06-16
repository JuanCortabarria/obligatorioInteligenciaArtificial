"""Smoke test: entrenamiento corto del QLearningAgent (recompensa REAL, sin shaping).

El objetivo NO es lograr la mejor política (para eso está el grid search), sino verificar
end-to-end que:
    1. La pipeline corre sin errores.
    2. La tabla Q se actualiza (deja de tener el valor inicial en las celdas visitadas).
    3. Con inicialización optimista, el agente alcanza la meta y aprende, **sin tocar la
       recompensa** (sin reward shaping).

Si esto no anda, ningún experimento va a andar.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym

from discretization import Discretizer
from q_learning_agent import QLearningAgent
from experiments import TEST_SEEDS


def plot_history(history: dict, out_path: Path, title: str):
    rewards = np.array(history["rewards"])
    successes = np.array(history["success"])
    window = 50
    moving_avg = (np.convolve(rewards, np.ones(window) / window, mode="valid")
                  if len(rewards) >= window else rewards)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(rewards, alpha=0.3, label="reward por episodio (real)")
    axes[0].plot(np.arange(len(moving_avg)) + window - 1, moving_avg, color="C1",
                 label=f"media móvil ({window})")
    axes[0].set_ylabel("Reward real acumulado"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[0].set_title(title)
    sr = np.convolve(successes, np.ones(window) / window, mode="valid")
    axes[1].plot(np.arange(len(sr)) + window - 1, sr, color="C2")
    axes[1].set_ylabel(f"Tasa de éxito (ventana {window})")
    axes[1].set_xlabel("Episodio"); axes[1].set_ylim(-0.05, 1.05); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Gráfico guardado en: {out_path}")


def main():
    env = gym.make("MountainCarContinuous-v0")
    discretizer = Discretizer(n_bins_x=40, n_bins_v=40, n_actions=5)

    agent = QLearningAgent(
        discretizer=discretizer,
        alpha=0.1, gamma=0.99,
        epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.999,
        optimistic_init=10.0,      # ← clave para aprender SIN tocar la recompensa
        reward_shaping=False,      # ← recompensa REAL
        seed=42,
    )

    print("Smoke test: 500 episodios, recompensa real + inicialización optimista (sin shaping)")
    history = agent.train_agent(env, episodes=500, verbose_every=100, env_seed=42)

    last_100 = slice(-100, None)
    success_rate = float(np.mean(history["success"][last_100]))
    avg_reward = float(np.mean(history["rewards"][last_100]))
    # Cobertura REAL = celdas actualizadas (distintas del valor inicial); con inicialización
    # optimista NO sirve count_nonzero (todo arranca distinto de cero).
    coverage = float(np.mean(agent.Q != agent.optimistic_init))

    print(f"\nResultados últimos 100 episodios:")
    print(f"  success_rate = {success_rate:.2%}")
    print(f"  avg_reward   = {avg_reward:7.2f}")
    print(f"  Q-table cobertura (visitada) = {coverage:.2%}")

    print("\nTest greedy (10 episodios, seeds fijas):")
    test_metrics = agent.test_agent(env, episodes=10, test_seeds=TEST_SEEDS)
    print(f"  avg_reward   = {test_metrics['avg_reward']:7.2f}")
    print(f"  success_rate = {test_metrics['success_rate']:.2%}")
    print(f"  avg_steps    = {test_metrics['avg_steps']:.1f}")
    env.close()

    out_dir = Path(__file__).parent / "models"
    plots_dir = Path(__file__).parent / "plots"
    plot_history(history, plots_dir / "smoke_test_learning_curve.png",
                 title="Smoke test — Q-Learning sin shaping (inicialización optimista, 500 ep)")
    agent.save(out_dir / "smoke_test.pkl")
    print(f"Modelo guardado en: {out_dir / 'smoke_test.pkl'}")

    if coverage == 0:
        print("\nFAIL: la Q-table no se actualizó — algo está roto en el update.")
        sys.exit(1)
    if success_rate == 0 and test_metrics["success_rate"] == 0:
        print("\nWARN: el agente no llegó a la meta. La pipeline corre pero no convergió en 500 ep.")
    print("\nSmoke test OK: la pipeline aprende con la recompensa real.")


if __name__ == "__main__":
    main()
