"""
Grid search de hiperparámetros para Q-Learning sobre MountainCarContinuous-v0.

Estrategia: **one-at-a-time** (OAT) — partimos de una config base validada por
el smoke test, y variamos un solo hiperparámetro a la vez. Esto es más
interpretable que un producto cartesiano completo (que sería 3·3·3·2·2·2 = 216 corridas)
y permite atribuir mejoras/empeoramientos a un cambio específico.

Métricas que registramos por corrida:
    - success_rate(últimos 100): fracción de éxitos al final del entrenamiento.
    - avg_reward(últimos 100): reward acumulado promedio (sin shaping).
    - convergence_ep: primer episodio a partir del cual success_rate(50) ≥ 0.9.
                     Si nunca llega, queda None.
    - test_success_rate / test_avg_reward / test_avg_steps: política greedy
                     evaluada en 20 episodios al final.

Salidas:
    plots/grid_search_curves.png   — todas las curvas de aprendizaje superpuestas.
    plots/grid_search_summary.png  — barras de comparación de métricas finales.
    models/q_learning_grid_best.pkl — el mejor agente del grid (entrenado con
                                       EPISODES=800 episodios). NO es el entregable
                                       final — para eso `train_best.py` re-entrena
                                       la config ganadora con 2000 episodios y
                                       guarda en `models/q_learning_best.pkl`.
    grid_search_results.json       — tabla con los resultados (para el informe).
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
from q_learning_agent import QLearningAgent


HERE = Path(__file__).parent
PLOTS_DIR = HERE / "plots"
MODELS_DIR = HERE / "models"

# ---- Config base: la del smoke test, que ya validamos que aprende. ----
BASE = {
    "bins": 40,
    "n_actions": 5,
    "alpha": 0.1,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_min": 0.05,
    "epsilon_decay": 0.995,
    "optimistic_init": 0.0,
    "reward_shaping": True,
    "shaping_coef": 300.0,
}

# Cada entrada del grid es {name: dict_of_overrides}. La config efectiva es BASE | overrides.
GRID = {
    "base":                     {},
    "shaping_off":              {"reward_shaping": False},
    "bins_gruesa_20":           {"bins": 20, "n_actions": 3},
    "bins_fina_100":            {"bins": 100, "n_actions": 10},
    "alpha_0.3":                {"alpha": 0.3},
    "alpha_0.05":               {"alpha": 0.05},
    "gamma_0.95":               {"gamma": 0.95},
    "eps_decay_0.999":          {"epsilon_decay": 0.999},
    "eps_decay_0.99":           {"epsilon_decay": 0.99},
    "optimistic_init_1.0":      {"optimistic_init": 1.0},
    "shaping_coef_100":         {"shaping_coef": 100.0},
    "shaping_coef_600":         {"shaping_coef": 600.0},
}

EPISODES = 800
SEED = 42


def run_one(name: str, overrides: dict) -> dict:
    cfg = {**BASE, **overrides}
    env = gym.make("MountainCarContinuous-v0")

    disc = Discretizer(
        n_bins_x=cfg["bins"], n_bins_v=cfg["bins"], n_actions=cfg["n_actions"]
    )
    agent = QLearningAgent(
        discretizer=disc,
        alpha=cfg["alpha"],
        gamma=cfg["gamma"],
        epsilon_start=cfg["epsilon_start"],
        epsilon_min=cfg["epsilon_min"],
        epsilon_decay=cfg["epsilon_decay"],
        optimistic_init=cfg["optimistic_init"],
        reward_shaping=cfg["reward_shaping"],
        shaping_coef=cfg["shaping_coef"],
        seed=SEED,
    )

    t0 = time.time()
    history = agent.train_agent(
        env, episodes=EPISODES, verbose_every=0, env_seed=SEED
    )
    train_time = time.time() - t0

    successes = np.array(history["success"])
    rewards = np.array(history["rewards"])
    last_100 = slice(-100, None)

    # Convergencia: primer episodio donde rolling success(50) ≥ 0.9.
    window = 50
    convergence_ep = None
    if len(successes) >= window:
        rolling = np.convolve(successes, np.ones(window) / window, mode="valid")
        above = np.where(rolling >= 0.9)[0]
        if len(above) > 0:
            convergence_ep = int(above[0] + window - 1)

    # Test greedy
    test_metrics = agent.test_agent(env, episodes=20)
    env.close()

    result = {
        "name": name,
        "config": cfg,
        "train_time_s": round(train_time, 2),
        "train_success_rate_last100": float(np.mean(successes[last_100])),
        "train_avg_reward_last100": float(np.mean(rewards[last_100])),
        "convergence_ep_50w_0.9": convergence_ep,
        "test_success_rate": test_metrics["success_rate"],
        "test_avg_reward": test_metrics["avg_reward"],
        "test_avg_steps": test_metrics["avg_steps"],
        "Q_nonzero_pct": float(np.count_nonzero(agent.Q) / agent.Q.size),
    }
    return result, history, agent


def plot_all_curves(histories: dict, out_path: Path):
    """Una curva por run, todas en el mismo eje. Reward media móvil 50."""
    fig, ax = plt.subplots(figsize=(12, 7))
    window = 50
    for name, hist in histories.items():
        rewards = np.array(hist["rewards"])
        if len(rewards) >= window:
            ma = np.convolve(rewards, np.ones(window) / window, mode="valid")
            xs = np.arange(len(ma)) + window - 1
            ax.plot(xs, ma, label=name, linewidth=1.2)
        else:
            ax.plot(rewards, label=name, linewidth=1.2)
    ax.set_xlabel("Episodio")
    ax.set_ylabel("Reward (media móvil 50)")
    ax.set_title("Grid search — curvas de aprendizaje")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_summary(results: list[dict], out_path: Path):
    """Barras: test success rate y test avg reward para cada run."""
    names = [r["name"] for r in results]
    succ = [r["test_success_rate"] for r in results]
    rew = [r["test_avg_reward"] for r in results]
    steps = [r["test_avg_steps"] for r in results]

    fig, axes = plt.subplots(3, 1, figsize=(11, 12))
    axes[0].barh(names, succ, color="C2")
    axes[0].set_xlim(0, 1.05)
    axes[0].set_xlabel("Test success rate (20 episodios greedy)")
    axes[0].grid(alpha=0.3, axis="x")

    axes[1].barh(names, rew, color="C0")
    axes[1].set_xlabel("Test avg reward (sin shaping)")
    axes[1].grid(alpha=0.3, axis="x")
    axes[1].axvline(0, color="k", linewidth=0.5)

    axes[2].barh(names, steps, color="C3")
    axes[2].set_xlabel("Test avg steps (menor = mejor)")
    axes[2].grid(alpha=0.3, axis="x")

    fig.suptitle("Grid search — resumen de métricas en test greedy")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    PLOTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    results = []
    histories = {}
    agents = {}

    for name, overrides in GRID.items():
        print(f"\n=== Run: {name} ===")
        print(f"  config: BASE | {overrides}")
        result, history, agent = run_one(name, overrides)
        results.append(result)
        histories[name] = history
        agents[name] = agent
        print(
            f"  done in {result['train_time_s']}s | "
            f"train_succ={result['train_success_rate_last100']:.2%} | "
            f"test_succ={result['test_success_rate']:.2%} | "
            f"test_reward={result['test_avg_reward']:.2f} | "
            f"test_steps={result['test_avg_steps']:.1f} | "
            f"conv@{result['convergence_ep_50w_0.9']}"
        )

    # Criterio de "mejor":
    #   1. Mayor test_success_rate (queremos políticas que resuelvan el problema).
    #   2. Entre las exitosas, MENOR test_avg_steps (política más eficiente:
    #      llega a la meta en menos pasos). En MountainCar, esto correlaciona
    #      directamente con un mejor reward acumulado por la penalización de
    #      acciones, y captura mejor "calidad de la política" que el reward
    #      crudo (que tiene varianza alta por las acciones óptimas).
    def score(r):
        # Tuple en orden: maximizar succ, minimizar steps (→ -steps), maximizar reward
        return (r["test_success_rate"], -r["test_avg_steps"], r["test_avg_reward"])

    sorted_results = sorted(results, key=score, reverse=True)
    best = sorted_results[0]
    best_name = best["name"]
    best_agent = agents[best_name]

    # Guardar en path DISTINTO del entregable final, para no sobreescribir
    # el modelo bien entrenado de train_best.py (2000 episodios) si se corre
    # este script después.
    best_agent.save(MODELS_DIR / "q_learning_grid_best.pkl")
    print(f"\nMejor run: {best_name}")
    print(f"  → guardado en {MODELS_DIR / 'q_learning_grid_best.pkl'}")
    print(f"  Nota: este modelo tiene 800 episodios. Para el entregable final corré")
    print(f"  `python3 train_best.py` que re-entrena con 2000 ep en q_learning_best.pkl.")

    # Plots y JSON
    plot_all_curves(histories, PLOTS_DIR / "grid_search_curves.png")
    plot_summary(sorted_results, PLOTS_DIR / "grid_search_summary.png")
    with open(HERE / "grid_search_results.json", "w") as f:
        json.dump(sorted_results, f, indent=2)
    print(f"\nResultados JSON: {HERE / 'grid_search_results.json'}")
    print(f"Plots: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
