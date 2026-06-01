"""
Visualización de la política aprendida y la value function del mejor modelo.

Genera dos mapas en el espacio de estado (x, v):
  1. Política: en cada celda, qué acción elige el agente (color = acción discreta).
  2. Value function: V(s) = max_a Q(s, a). Muestra qué tan "valioso" cree el
     agente que es cada estado.

Verifica que la política aprendida es razonable: aplicar fuerza positiva cuando
se va hacia la derecha (v > 0) y negativa cuando se va hacia la izquierda (v < 0),
para acumular momento en cada oscilación.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from q_learning_agent import QLearningAgent


HERE = Path(__file__).parent


def main():
    agent = QLearningAgent.load(HERE / "models" / "q_learning_best.pkl")
    disc = agent.discretizer

    # V(s) = max_a Q(s, a) — la "valoración" del estado bajo la política greedy.
    V = np.max(agent.Q, axis=2)
    # π(s) = argmax_a Q(s, a) — qué acción elige el agente en cada estado.
    pi = np.argmax(agent.Q, axis=2)

    # Máscara de "estado visitado": al menos una entrada Q(s, ·) cambió respecto
    # de la inicialización optimistic_init. Estados no visitados quedan en valor
    # default y argmax devuelve ruido — los enmascaramos para no engañar al lector.
    visited = np.any(agent.Q != agent.optimistic_init, axis=2)

    # Recortamos el "+1" de digitize porque las celdas extremas casi no se visitan.
    V = V[: disc.n_bins_x, : disc.n_bins_v]
    pi = pi[: disc.n_bins_x, : disc.n_bins_v]
    visited = visited[: disc.n_bins_x, : disc.n_bins_v]

    # Eje x = posición, eje y = velocidad. La matriz tiene shape (n_bins_x, n_bins_v),
    # transponemos para que x quede en horizontal.
    extent = [disc.X_MIN, disc.X_MAX, disc.V_MIN, disc.V_MAX]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # Value function
    im0 = axes[0].imshow(
        V.T,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
    )
    axes[0].set_xlabel("Posición x")
    axes[0].set_ylabel("Velocidad v")
    axes[0].set_title("V(s) = max_a Q(s, a)")
    axes[0].axvline(0.45, color="red", linestyle="--", linewidth=1.2, label="meta x=0.45")
    axes[0].axhline(0, color="white", linestyle=":", linewidth=0.8, alpha=0.5)
    axes[0].legend(loc="upper left")
    fig.colorbar(im0, ax=axes[0], label="V(s)")

    # Política — solo se muestra en estados visitados (NaN en el resto).
    pi_real = disc.actions[pi].astype(float)
    pi_real_masked = np.where(visited, pi_real, np.nan)
    im1 = axes[1].imshow(
        pi_real_masked.T,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="coolwarm",
        interpolation="nearest",
        vmin=-1,
        vmax=1,
    )
    axes[1].set_facecolor("#dddddd")  # estados no visitados quedan en gris claro
    axes[1].set_xlabel("Posición x")
    axes[1].set_ylabel("Velocidad v")
    axes[1].set_title(f"π(s) en estados visitados (gris = no visitado)\nacciones disponibles: {disc.actions}")
    axes[1].axvline(0.45, color="black", linestyle="--", linewidth=1.2, label="meta x=0.45")
    axes[1].axhline(0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
    axes[1].legend(loc="upper left")
    fig.colorbar(im1, ax=axes[1], label="Fuerza aplicada")

    fig.suptitle(
        f"Política y value function aprendidas — q_learning_best "
        f"(bins={disc.n_bins_x}, n_actions={disc.n_actions})",
        fontsize=12,
    )
    fig.tight_layout()
    out_path = HERE / "plots" / "q_learning_best_policy.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Plot: {out_path}")

    # Diagnóstico de la política, restringido a *estados visitados*.
    # La estrategia óptima clásica de MountainCar es "pump-and-go":
    #   - v > 0 (yendo derecha)  → empujar +1 para acumular momento.
    #   - v < 0 (yendo izquierda) → empujar -1 para acumular momento del otro lado.
    # En estados no-visitados Q(s,·) ≈ optimistic_init para todas las acciones,
    # entonces argmax es ruido (siempre devuelve la primera). Filtrarlos da
    # una medición honesta.
    visit_pct = float(np.mean(visited))
    print(f"\nCobertura del espacio de estados (visitados durante training): {visit_pct:.1%}")

    # Slices de la mitad superior (v > 0) y mitad inferior (v < 0) — saltamos
    # la fila central (v ≈ 0) donde la "dirección correcta" es ambigua.
    mid = disc.n_bins_v // 2
    upper = slice(mid + 1, disc.n_bins_v)  # v > 0
    lower = slice(0, mid)                   # v < 0

    right_visited = visited[:, upper]
    left_visited = visited[:, lower]
    right_actions = pi_real[:, upper]
    left_actions = pi_real[:, lower]

    if right_visited.sum() == 0 or left_visited.sum() == 0:
        print("  → No hay suficientes estados visitados para diagnosticar.")
        return

    pct_pos_when_right = float(np.mean(right_actions[right_visited] > 0))
    pct_neg_when_left = float(np.mean(left_actions[left_visited] < 0))

    print(f"Diagnóstico (solo estados visitados):")
    print(f"  Cuando v > 0: % acciones positivas = {pct_pos_when_right:.1%}  "
          f"(n={int(right_visited.sum())} celdas)")
    print(f"  Cuando v < 0: % acciones negativas = {pct_neg_when_left:.1%}  "
          f"(n={int(left_visited.sum())} celdas)")
    if pct_pos_when_right > 0.5 and pct_neg_when_left > 0.5:
        print("  → Política coherente con la estrategia 'pump-and-go' esperada.")
    else:
        print("  → La política aprendida se aparta de la heurística simple — verificar con video.")


if __name__ == "__main__":
    main()
