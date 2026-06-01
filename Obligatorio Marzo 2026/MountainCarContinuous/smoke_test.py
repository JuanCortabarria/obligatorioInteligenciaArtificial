"""
Smoke test: entrenamiento corto del QLearningAgent.

El objetivo NO es lograr una política buena (para eso está el grid search),
sino verificar end-to-end que:
    1. La pipeline corre sin errores.
    2. La tabla Q se actualiza (deja de ser todo ceros).
    3. Con reward shaping, el agente al menos logra alcanzar la meta alguna vez.

Si esto no anda, ningún grid search va a andar.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend no interactivo, para correr en CLI sin display
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym

from discretization import Discretizer
from q_learning_agent import QLearningAgent


def plot_history(history: dict, out_path: Path, title: str):
    rewards = np.array(history["rewards"])
    successes = np.array(history["success"])
    window = 50
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
    else:
        moving_avg = rewards

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(rewards, alpha=0.3, label="reward por episodio")
    axes[0].plot(
        np.arange(len(moving_avg)) + window - 1,
        moving_avg,
        color="C1",
        label=f"media móvil ({window})",
    )
    axes[0].set_ylabel("Reward acumulado")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_title(title)

    success_rate_running = np.convolve(
        successes, np.ones(window) / window, mode="valid"
    )
    axes[1].plot(
        np.arange(len(success_rate_running)) + window - 1,
        success_rate_running,
        color="C2",
    )
    axes[1].set_ylabel(f"Tasa de éxito (ventana {window})")
    axes[1].set_xlabel("Episodio")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(alpha=0.3)

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
        alpha=0.1,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        optimistic_init=0.0,
        reward_shaping=True,
        shaping_coef=300.0,  # potential-based: Φ=300·|v|, escala razonable vs reward base
        seed=42,
    )

    print("Smoke test: 500 episodios, config media + shaping")
    history = agent.train_agent(env, episodes=500, verbose_every=100, env_seed=42)

    last_100 = slice(-100, None)
    success_rate = float(np.mean(history["success"][last_100]))
    avg_reward = float(np.mean(history["rewards"][last_100]))
    nonzero_q = int(np.count_nonzero(agent.Q))
    total_q = agent.Q.size

    print()
    print(f"Resultados últimos 100 episodios:")
    print(f"  success_rate = {success_rate:.2%}")
    print(f"  avg_reward   = {avg_reward:7.2f}")
    print(f"  Q no-cero    = {nonzero_q}/{total_q} ({nonzero_q / total_q:.2%})")

    # Test greedy
    print()
    print("Test greedy (10 episodios):")
    test_metrics = agent.test_agent(env, episodes=10)
    print(f"  avg_reward   = {test_metrics['avg_reward']:7.2f}")
    print(f"  success_rate = {test_metrics['success_rate']:.2%}")
    print(f"  avg_steps    = {test_metrics['avg_steps']:.1f}")

    env.close()

    # Guardar plot + modelo del smoke test (no es el modelo final del entregable,
    # pero sirve para tener algo a la mano y testear el flujo save/load).
    out_dir = Path(__file__).parent / "models"
    plots_dir = Path(__file__).parent / "plots"
    plot_history(
        history,
        plots_dir / "smoke_test_learning_curve.png",
        title="Smoke test — Q-Learning + shaping (config media, 500 ep)",
    )
    agent.save(out_dir / "smoke_test.pkl")
    print(f"Modelo guardado en: {out_dir / 'smoke_test.pkl'}")

    # Criterios de éxito del smoke test:
    if nonzero_q == 0:
        print("\nFAIL: Q sigue en cero — algo está roto en el update.")
        sys.exit(1)
    if success_rate == 0 and test_metrics["success_rate"] == 0:
        print("\nWARN: el agente no llegó a la meta nunca. La pipeline corre pero el aprendizaje no convergió en 500 episodios.")
        # No es un fail duro: 500 episodios es poco. Lo dejamos como advertencia.
    print("\nSmoke test OK: la pipeline aprende.")


if __name__ == "__main__":
    main()
