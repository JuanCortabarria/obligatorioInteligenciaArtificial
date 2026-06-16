"""Experimento 'episodios vs exploración' (responde a la devolución de la cátedra).

Demuestra que el cuello de botella de Q-Learning en MountainCarContinuous NO es la
cantidad de episodios sino la **exploración**: con ε-greedy el agente casi nunca llega a
la meta (no tiene de dónde aprender), y subir los episodios (hasta 10.000) no lo arregla.
La **inicialización optimista** (técnica estándar de Q-Learning, Sutton & Barto §2.6) sí lo
resuelve, porque fuerza exploración sistemática.

Compara Q-Learning **vanilla** (`opt=0`) vs **con inicialización optimista** (`opt=10`) a
distintos presupuestos de episodios, con varias semillas. Mide la **tasa de éxito en test**
y la **cantidad de veces que el agente llegó a la meta durante el entrenamiento**.

Salidas: `episodes_vs_exploration.json` + `plots/episodes_vs_exploration.png`.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from discretization import Discretizer
from q_learning_agent import QLearningAgent
import experiments as ex

HERE = Path(__file__).parent
SEEDS = [0, 1, 2]
EPISODES = [1500, 5000, 10000]
sns.set_theme(style="whitegrid")


def run(opt, episodes, seed):
    env = gym.make("MountainCarContinuous-v0")
    disc = Discretizer(40, 40, 5)
    ag = QLearningAgent(discretizer=disc, alpha=0.1, gamma=0.99,
                        epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.999,
                        optimistic_init=opt, reward_shaping=False, seed=seed)
    h = ag.train_agent(env, episodes=episodes, verbose_every=0, env_seed=seed)
    test = ag.test_agent(env, episodes=10, test_seeds=ex.TEST_SEEDS)
    n_goals = int(np.sum(h["success"]))
    env.close()
    return test["success_rate"], n_goals


def main():
    (HERE / "plots").mkdir(exist_ok=True)
    rows = []
    for opt, label in [(0.0, "Q-Learning vanilla (opt=0)"),
                       (10.0, "Q-Learning + init optimista (opt=10)")]:
        for ep in EPISODES:
            for s in SEEDS:
                t0 = time.time()
                ts, ng = run(opt, ep, s)
                rows.append(dict(variante=label, episodios=ep, seed=s,
                                 test_success=ts, metas_train=ng))
                print(f"{label:38s} ep={ep:6d} seed={s} -> test={ts:.0%} metas={ng} ({time.time()-t0:.0f}s)")
    df = pd.DataFrame(rows)
    df.to_json(HERE / "episodes_vs_exploration.json", orient="records", indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.barplot(data=df, x="episodios", y="test_success", hue="variante", errorbar="sd", ax=axes[0])
    axes[0].set_title("Éxito en test vs cantidad de episodios")
    axes[0].set_ylabel("Tasa de éxito en test"); axes[0].set_xlabel("Episodios de entrenamiento")
    axes[0].set_ylim(0, 1.05); axes[0].legend(fontsize=8)
    sns.barplot(data=df, x="episodios", y="metas_train", hue="variante", errorbar="sd", ax=axes[1])
    axes[1].set_title("Veces que llegó a la meta durante el entrenamiento")
    axes[1].set_ylabel("# metas alcanzadas (entrenamiento)"); axes[1].set_xlabel("Episodios de entrenamiento")
    axes[1].set_yscale("symlog"); axes[1].legend(fontsize=8)
    fig.suptitle("Más episodios NO resuelven el Q-Learning vanilla (casi nunca llega a la meta); "
                 "la inicialización optimista sí")
    fig.tight_layout()
    fig.savefig(HERE / "plots" / "episodes_vs_exploration.png", dpi=120)
    plt.close(fig)

    print("\n=== Resumen (media sobre seeds) ===")
    print(df.groupby(["variante", "episodios"])[["test_success", "metas_train"]].mean().round(2).to_string())
    print(f"\nGuardado: episodes_vs_exploration.json + plots/episodes_vs_exploration.png")


if __name__ == "__main__":
    main()
