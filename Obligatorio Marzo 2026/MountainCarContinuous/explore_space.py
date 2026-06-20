"""Exploración del espacio de OBSERVACIÓN y de ACCIÓN (apoyo visual para el informe).

La cátedra recomienda graficar el espacio de observación (2D: posición/velocidad) y
el de acción (1D) para ver **qué se explora y qué no** y para **justificar la
discretización**. Este script genera, a partir del modelo entregable
`models/q_learning_best.pkl`, dos figuras:

  1. plots/observation_space_coverage.png
     - Panel A: qué celdas del espacio (x, v) se **visitaron alguna vez** durante el
       entrenamiento (celda actualizada ⇔ Q ≠ inicialización optimista) vs. cuáles
       **nunca** se visitaron → cobertura del grid.
     - Panel B: **por dónde pasa la política óptima** (densidad de visitas en rollouts
       greedy), con una trayectoria de ejemplo superpuesta.

  2. plots/action_space_distribution.png
     - Panel A: con qué frecuencia la **política** elige cada acción discreta
       (argmax sobre las celdas visitadas).
     - Panel B: qué acciones se **ejecutan** en rollouts óptimos (greedy).

Imprime además las métricas reales (cobertura %, % por acción) para citarlas en el informe.
"""

from pathlib import Path

import numpy as np
import gymnasium as gym

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from q_learning_agent import QLearningAgent

HERE = Path(__file__).parent
PLOTS = HERE / "plots"
ENV_ID = "MountainCarContinuous-v0"
GOAL_X = 0.45  # posición de la meta (Gymnasium)
# Mismos episodios fijos que se usan para evaluar (comparable con el resto del informe).
TEST_SEEDS = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]

sns.set_theme(style="whitegrid")


def greedy_rollouts(agent, env, seeds, max_steps=1000):
    """Corre episodios con la política greedy (sin exploración) y acumula:
    visitas por celda de estado, conteo por acción y una trayectoria de ejemplo."""
    disc = agent.discretizer
    state_visits = np.zeros(disc.state_shape, dtype=float)
    action_counts = np.zeros(disc.n_actions, dtype=float)
    example_traj = []
    for i, s in enumerate(seeds):
        obs, _ = env.reset(seed=int(s))
        ep_points = []
        for _ in range(max_steps):
            st = disc.get_state(obs)
            a_idx = int(np.argmax(agent.Q[st]))
            state_visits[st] += 1
            action_counts[a_idx] += 1
            ep_points.append((float(obs[0]), float(obs[1])))
            obs, _, terminated, truncated, _ = env.step(disc.action_from_idx(a_idx))
            if terminated or truncated:
                break
        if i == 0:
            example_traj = ep_points
    return state_visits, action_counts, np.array(example_traj)


def main():
    PLOTS.mkdir(exist_ok=True)
    agent = QLearningAgent.load(HERE / "models" / "q_learning_best.pkl")
    disc = agent.discretizer

    # --- Cobertura del entrenamiento: celda "visitada" ⇔ alguna Q(s,·) cambió respecto
    # de la inicialización optimista (estados nunca visitados quedan en el valor inicial).
    visited_full = np.any(agent.Q != agent.optimistic_init, axis=2)
    visited = visited_full[: disc.n_bins_x, : disc.n_bins_v]
    coverage = float(visited.mean())

    # --- Rollouts greedy: por dónde pasa la política óptima + qué acciones ejecuta.
    env = gym.make(ENV_ID)
    state_visits, action_counts, traj = greedy_rollouts(agent, env, TEST_SEEDS)
    env.close()
    sv = state_visits[: disc.n_bins_x, : disc.n_bins_v]

    extent = [disc.X_MIN, disc.X_MAX, disc.V_MIN, disc.V_MAX]

    # ============ Figura 1: espacio de observación ============
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    axes[0].imshow(visited.T.astype(float), origin="lower", extent=extent, aspect="auto",
                   cmap="Greens", vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_title(f"Explorado durante el entrenamiento\n"
                      f"(verde = visitado: {coverage:.0%} del grid · "
                      f"{1 - coverage:.0%} nunca visitado)")
    axes[0].set_xlabel("Posición x"); axes[0].set_ylabel("Velocidad v")
    axes[0].axvline(GOAL_X, color="red", ls="--", lw=1.2, label=f"meta x={GOAL_X}")
    axes[0].axhline(0, color="gray", ls=":", lw=0.8, alpha=0.6)
    axes[0].legend(loc="upper left")

    dens = np.log1p(sv.T)
    im1 = axes[1].imshow(dens, origin="lower", extent=extent, aspect="auto",
                         cmap="magma", interpolation="nearest")
    if traj.size:
        axes[1].plot(traj[:, 0], traj[:, 1], color="cyan", lw=1.2, alpha=0.9,
                     label="trayectoria óptima (ej.)")
        axes[1].scatter([traj[0, 0]], [traj[0, 1]], color="white", s=30, zorder=5, label="inicio")
    axes[1].set_title("Por dónde pasa la política óptima\n(densidad de visitas en rollouts greedy, escala log)")
    axes[1].set_xlabel("Posición x"); axes[1].set_ylabel("Velocidad v")
    axes[1].axvline(GOAL_X, color="cyan", ls="--", lw=1.0, alpha=0.7)
    axes[1].legend(loc="upper left")
    fig.colorbar(im1, ax=axes[1], label="log(1 + visitas)")

    fig.suptitle("Espacio de observación (x, v): cobertura de exploración y recorrido óptimo",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(PLOTS / "observation_space_coverage.png", dpi=120)
    plt.close(fig)

    # ============ Figura 2: espacio de acción (1D) ============
    pi = np.argmax(agent.Q, axis=2)[: disc.n_bins_x, : disc.n_bins_v]
    pol_counts = np.array([(pi[visited] == k).sum() for k in range(disc.n_actions)], dtype=float)
    labels = [f"{a:+.1f}" for a in disc.actions]
    pol_pct = 100 * pol_counts / pol_counts.sum()
    exec_pct = 100 * action_counts / action_counts.sum()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    sns.barplot(x=labels, y=pol_pct, ax=axes[0], color="#4c72b0")
    axes[0].set_title("Acción de la política (argmax) sobre celdas visitadas")
    axes[0].set_xlabel("Fuerza discreta"); axes[0].set_ylabel("% de celdas")
    sns.barplot(x=labels, y=exec_pct, ax=axes[1], color="#dd8452")
    axes[1].set_title("Acciones ejecutadas en rollouts óptimos (greedy)")
    axes[1].set_xlabel("Fuerza discreta"); axes[1].set_ylabel("% de pasos")
    fig.suptitle(f"Espacio de acción (1D): distribución de la fuerza elegida "
                 f"(acciones disponibles: {[float(a) for a in disc.actions]})", fontsize=12)
    fig.tight_layout()
    fig.savefig(PLOTS / "action_space_distribution.png", dpi=120)
    plt.close(fig)

    # --- Métricas reales para el informe ---
    print(f"Cobertura del espacio de estados (visitado en training): {coverage:.1%}")
    print(f"  celdas visitadas: {int(visited.sum())} / {visited.size}")
    print("\nDistribución de acciones de la POLÍTICA (sobre celdas visitadas):")
    for lab, p in zip(labels, pol_pct):
        print(f"  {lab}: {p:5.1f}%")
    print("\nDistribución de acciones EJECUTADAS en rollouts óptimos:")
    for lab, p in zip(labels, exec_pct):
        print(f"  {lab}: {p:5.1f}%")
    extremes_pol = pol_pct[0] + pol_pct[-1]
    extremes_exec = exec_pct[0] + exec_pct[-1]
    zero_idx = list(disc.actions).index(0.0) if 0.0 in disc.actions else None
    print(f"\n% acciones extremas (±1) — política: {extremes_pol:.1f}%  ·  ejecutadas: {extremes_exec:.1f}%")
    if zero_idx is not None:
        print(f"% acción 0.0 (no empujar) — política: {pol_pct[zero_idx]:.1f}%  ·  ejecutadas: {exec_pct[zero_idx]:.1f}%")
    print(f"\nGuardado: {PLOTS/'observation_space_coverage.png'} y {PLOTS/'action_space_distribution.png'}")


if __name__ == "__main__":
    main()
