"""Infraestructura de experimentos multi-seed para LOST (recompensa REAL, sin shaping).

Motivación (devolución de la cátedra): la experimentación debe (1) explorar diversos
hiperparámetros y discretizaciones, (2) **repetir cada configuración con varias semillas
para analizar la varianza** de los resultados, y (3) visualizarse con **bandas de error o
boxplots**. Este módulo centraliza eso para que `grid_search.py`, `compare_dyna_q.py` y
`train_best.py` lo reutilicen.

Decisiones de rigor:
  - Cada corrida siembra el agente (`seed=s` → `random`+`numpy`) y el env (`env_seed=s`),
    de forma independiente y reproducible por semilla.
  - El **test** se evalúa sobre un conjunto FIJO de episodios sembrados (`TEST_SEEDS`),
    igual para todas las configuraciones, así la métrica refleja la política y no la
    suerte del reset (comparación justa).
  - **NO se usa reward shaping** (el núcleo trabaja con la recompensa real del ambiente).
"""

import time

import numpy as np
import pandas as pd
import gymnasium as gym

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from discretization import Discretizer
from q_learning_agent import QLearningAgent
from dyna_q_agent import DynaQAgent

ENV_ID = "MountainCarContinuous-v0"

# Semillas de ENTRENAMIENTO (presupuesto ligero: 5) → miden la varianza.
SEEDS = [0, 1, 2, 3, 4]
# Episodios de TEST fijos (mismos para todas las configs) → evaluación comparable.
TEST_SEEDS = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]

# Configuración base (recompensa real). La clave para que Q-Learning aprenda sin shaping
# es la INICIALIZACIÓN OPTIMISTA (Sutton & Barto §2.6): sin ella cae en la "trampa de no
# hacer nada" (moverse cuesta energía y la meta casi nunca se alcanza → 0% de éxito).
DEFAULT = dict(
    bins=40, n_actions=5,
    alpha=0.1, gamma=0.99,
    epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.999,
    optimistic_init=10.0,
    reward_shaping=False,
    planning_steps=0,          # 0 = Q-Learning; >0 = Dyna-Q
)

sns.set_theme(style="whitegrid")


def build_agent(cfg: dict, seed: int):
    """Construye el agente (Q-Learning o Dyna-Q) para una semilla, sin shaping."""
    disc = Discretizer(cfg["bins"], cfg["bins"], cfg["n_actions"])
    kwargs = dict(
        alpha=cfg["alpha"], gamma=cfg["gamma"],
        epsilon_start=cfg["epsilon_start"], epsilon_min=cfg["epsilon_min"],
        epsilon_decay=cfg["epsilon_decay"], optimistic_init=cfg["optimistic_init"],
        reward_shaping=cfg["reward_shaping"], seed=seed,
    )
    if cfg.get("planning_steps", 0) > 0:
        return DynaQAgent(discretizer=disc, planning_steps=cfg["planning_steps"], **kwargs)
    return QLearningAgent(discretizer=disc, **kwargs)


def _rolling(x, w=50):
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return None
    return np.convolve(x, np.ones(w) / w, mode="valid")


def run_config(name: str, overrides: dict, seeds=SEEDS, episodes=1500,
               eval_episodes=10, curve_every=10):
    """Corre una configuración con `seeds` semillas. Devuelve `(df_curvas, df_finales)`.

    `df_finales`: una fila por semilla (train_success, test_success, test_reward,
    test_steps, conv_ep, train_time_s). `df_curvas`: reward (media móvil 50) por episodio
    y semilla, submuestreado cada `curve_every` para los gráficos con banda de error.
    """
    cfg = {**DEFAULT, **overrides}
    curves, finals = [], []
    for s in seeds:
        env = gym.make(ENV_ID)
        agent = build_agent(cfg, s)
        t0 = time.time()
        h = agent.train_agent(env, episodes=episodes, verbose_every=0, env_seed=s)
        dt = time.time() - t0

        succ = np.array(h["success"])
        rew = np.array(h["rewards"])

        # Convergencia: primer episodio con éxito en ventana móvil de 50 ≥ 90%.
        roll = _rolling(succ, 50)
        conv = episodes
        if roll is not None:
            idx = np.where(roll >= 0.9)[0]
            if len(idx) > 0:
                conv = int(idx[0] + 49)

        test = agent.test_agent(env, episodes=eval_episodes, test_seeds=TEST_SEEDS)
        env.close()

        finals.append(dict(
            config=name, seed=s,
            train_success=float(np.mean(succ[-100:])),
            test_success=test["success_rate"],
            test_reward=test["avg_reward"],
            test_steps=test["avg_steps"],
            conv_ep=conv,
            train_time_s=round(dt, 1),
        ))

        rma = _rolling(rew, 50)
        if rma is not None:
            for e in range(0, len(rma), curve_every):
                curves.append(dict(config=name, seed=s, episode=e + 49, reward_ma=float(rma[e])))

    return pd.DataFrame(curves), pd.DataFrame(finals)


def summary(df_finals: pd.DataFrame) -> pd.DataFrame:
    """Resumen por configuración: mediana y desvío entre semillas (análisis de varianza)."""
    g = df_finals.groupby("config")
    out = pd.DataFrame({
        "n_seeds": g.size(),
        "test_success_mediana": g["test_success"].median(),
        "test_success_min": g["test_success"].min(),
        "test_reward_mediana": g["test_reward"].median(),
        "test_reward_std": g["test_reward"].std().round(2),
        "test_steps_mediana": g["test_steps"].median().round(1),
        "conv_ep_mediana": g["conv_ep"].median(),
    }).reset_index()
    return out


# ---- Gráficos (seaborn): bandas de error y boxplots ----

def plot_curves_band(df_curves: pd.DataFrame, out_path, title):
    """Curvas de aprendizaje con BANDA DE ERROR (dispersión entre semillas)."""
    plt.figure(figsize=(11, 6))
    sns.lineplot(data=df_curves, x="episode", y="reward_ma", hue="config", errorbar="sd")
    plt.title(title)
    plt.xlabel("Episodio")
    plt.ylabel("Reward real (media móvil 50)  ±  desvío entre seeds")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_box(df_finals: pd.DataFrame, value, out_path, title, ylabel):
    """Boxplot de una métrica final por configuración (muestra la varianza entre seeds)."""
    plt.figure(figsize=(11, 6))
    sns.boxplot(data=df_finals, x="config", y=value, color="#9ecae1")
    sns.stripplot(data=df_finals, x="config", y=value, color="black", size=4, alpha=0.6)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


if __name__ == "__main__":
    # Smoke test del módulo: 2 configs, 2 seeds, pocos episodios → verifica que corre.
    import os
    os.makedirs("plots", exist_ok=True)
    c1, f1 = run_config("QL opt=10", {}, seeds=[0, 1], episodes=300)
    c2, f2 = run_config("QL opt=0", {"optimistic_init": 0.0}, seeds=[0, 1], episodes=300)
    curves = pd.concat([c1, c2], ignore_index=True)
    finals = pd.concat([f1, f2], ignore_index=True)
    print(summary(finals).to_string(index=False))
    plot_curves_band(curves, "plots/_smoke_curves.png", "Smoke")
    plot_box(finals, "test_success", "plots/_smoke_box.png", "Smoke", "test success")
    print("OK experiments.py (smoke)")
