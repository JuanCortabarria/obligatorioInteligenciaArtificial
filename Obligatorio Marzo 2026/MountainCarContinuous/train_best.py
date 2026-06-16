"""Entrenamiento de los modelos finales entregables (recompensa REAL, sin shaping).

Entrena y guarda los dos modelos con la recompensa original del ambiente:
  - `models/q_learning_best.pkl`: el Q-Learning **robusto** ganador del grid search
    (`bins=40, α=0.3, γ=0.99, ε_decay=0.999, inicialización optimista=10`).
  - `models/dyna_q_best.pkl`: el mejor Dyna-Q de la comparación (`planning_steps=5`,
    sobre la config base `α=0.1`), que converge antes y da mejor política.

La clave para aprender SIN modificar la recompensa es la **inicialización optimista**
(Sutton & Barto §2.6), no el reward shaping: sin optimismo Q-Learning cae en la "trampa de
no hacer nada" (0% de éxito); con él resuelve el ambiente con la recompensa real.
"""

from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym

from experiments import DEFAULT, build_agent, TEST_SEEDS

HERE = Path(__file__).parent
SEED = 0
EPISODES = 2500


def train_and_report(name, cfg, plot_path=None):
    env = gym.make("MountainCarContinuous-v0")
    agent = build_agent(cfg, SEED)
    t0 = time.time()
    history = agent.train_agent(env, episodes=EPISODES, verbose_every=0, env_seed=SEED)
    dt = time.time() - t0
    last100 = slice(-100, None)
    test = agent.test_agent(env, episodes=50, test_seeds=TEST_SEEDS)
    env.close()

    print(f"\n=== {name} ({dt:.1f}s) ===")
    print(f"  Train success (últimos 100): {np.mean(history['success'][last100]):.2%}")
    print(f"  Test success (50 ep):        {test['success_rate']:.2%}")
    print(f"  Test avg reward (real):      {test['avg_reward']:.2f}")
    print(f"  Test avg steps:              {test['avg_steps']:.1f}")
    # Cobertura REAL = celdas actualizadas (distintas del valor inicial). Con
    # inicialización optimista NO sirve count_nonzero (todo arranca != 0).
    coverage = float(np.mean(agent.Q != agent.optimistic_init))
    print(f"  Q-table cobertura (visitada): {coverage:.2%}")

    if plot_path is not None:
        rewards = np.array(history["rewards"]); succ = np.array(history["success"])
        w = 100
        ma = np.convolve(rewards, np.ones(w) / w, mode="valid")
        sr = np.convolve(succ, np.ones(w) / w, mode="valid")
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        axes[0].plot(rewards, alpha=0.25, label="reward/ep (real)")
        axes[0].plot(np.arange(len(ma)) + w - 1, ma, color="C1", label=f"media móvil ({w})")
        axes[0].set_ylabel("Reward real acumulado"); axes[0].grid(alpha=0.3); axes[0].legend()
        axes[0].set_title(f"{name} SIN shaping ({EPISODES} ep)")
        axes[1].plot(np.arange(len(sr)) + w - 1, sr, color="C2")
        axes[1].set_ylabel(f"Tasa éxito ({w})"); axes[1].set_xlabel("Episodio")
        axes[1].set_ylim(-0.05, 1.05); axes[1].grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(plot_path, dpi=120); plt.close(fig)
        print(f"  Plot: {plot_path}")

    return agent


def main():
    (HERE / "models").mkdir(exist_ok=True)
    (HERE / "plots").mkdir(exist_ok=True)

    # 1) Q-Learning robusto (grid winner): α=0.3, n=0.
    ql_cfg = {**DEFAULT, "alpha": 0.3, "planning_steps": 0}
    ql = train_and_report("Q-Learning final (α=0.3)", ql_cfg,
                          plot_path=HERE / "plots" / "q_learning_best_curve.png")
    ql.save(HERE / "models" / "q_learning_best.pkl")
    print(f"  Guardado: {HERE / 'models' / 'q_learning_best.pkl'}")

    # 2) Dyna-Q (mejor de la comparación): base α=0.1, n=5.
    dyna_cfg = {**DEFAULT, "planning_steps": 5}
    dyna = train_and_report("Dyna-Q final (n=5)", dyna_cfg)
    dyna.save(HERE / "models" / "dyna_q_best.pkl")
    print(f"  Guardado: {HERE / 'models' / 'dyna_q_best.pkl'}")


if __name__ == "__main__":
    main()
