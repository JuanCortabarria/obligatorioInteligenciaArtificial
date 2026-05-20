"""
Entrenamiento final del mejor modelo Q-Learning encontrado en el grid search.

Toma la config ganadora (alpha_0.05: bins=40, n_actions=5, alpha=0.05, resto BASE)
y la entrena con más episodios para que la Q-table esté bien convergida.
Este es el modelo final que entregamos como `models/q_learning_best.pkl`.

Nota histórica: en la primera versión del grid search, ganaba `bins_gruesa_20`
con steps=72.8. Tras corregir el bug del reward shaping en estados terminales
(ver §2.6 del informe), aquella config quedó en 95% de éxito — había estado
"ganando por casualidad" porque el bonus incorrecto compensaba la baja
resolución. Con el shaping correcto, la ganadora es `alpha_0.05`.
"""

from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym

from discretization import Discretizer
from q_learning_agent import QLearningAgent


HERE = Path(__file__).parent
SEED = 42
EPISODES = 2000


def main():
    env = gym.make("MountainCarContinuous-v0")
    disc = Discretizer(n_bins_x=40, n_bins_v=40, n_actions=5)
    agent = QLearningAgent(
        discretizer=disc,
        alpha=0.05,
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
        env, episodes=EPISODES, verbose_every=200, env_seed=SEED
    )
    train_time = time.time() - t0

    # Métricas finales
    last_100 = slice(-100, None)
    test_metrics = agent.test_agent(env, episodes=50)
    env.close()

    print(f"\nTrain time: {train_time:.1f}s")
    print(f"Final train success rate (last 100): {np.mean(history['success'][last_100]):.2%}")
    print(f"Final train avg reward (last 100):   {np.mean(history['rewards'][last_100]):.2f}")
    print(f"Test (50 ep greedy) success rate:    {test_metrics['success_rate']:.2%}")
    print(f"Test avg reward:                     {test_metrics['avg_reward']:.2f}")
    print(f"Test avg steps:                      {test_metrics['avg_steps']:.1f}")
    print(f"Q-table cobertura:                   {np.count_nonzero(agent.Q) / agent.Q.size:.2%}")

    # Save final model
    agent.save(HERE / "models" / "q_learning_best.pkl")
    print(f"\nModelo guardado: {HERE / 'models' / 'q_learning_best.pkl'}")

    # Plot final
    rewards = np.array(history["rewards"])
    successes = np.array(history["success"])
    window = 100
    ma = np.convolve(rewards, np.ones(window) / window, mode="valid")
    sr = np.convolve(successes, np.ones(window) / window, mode="valid")

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(rewards, alpha=0.25, label="reward/ep")
    axes[0].plot(np.arange(len(ma)) + window - 1, ma, color="C1", label=f"media móvil ({window})")
    axes[0].set_ylabel("Reward acumulado"); axes[0].grid(alpha=0.3); axes[0].legend()
    axes[0].set_title(f"Q-Learning final — bins=40, n_actions=5, α=0.05, shaping potential-based ({EPISODES} ep)")
    axes[1].plot(np.arange(len(sr)) + window - 1, sr, color="C2")
    axes[1].set_ylabel(f"Tasa éxito ({window})"); axes[1].set_xlabel("Episodio"); axes[1].set_ylim(-0.05, 1.05); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "plots" / "q_learning_best_curve.png", dpi=120)
    plt.close(fig)
    print(f"Plot: {HERE / 'plots' / 'q_learning_best_curve.png'}")


if __name__ == "__main__":
    main()
