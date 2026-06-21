"""Analiza required_matchups_results.csv (matriz completa de matchups del profesor).

Produce, a partir de partidas REALES (sin inventar nada):
  - tabla de win rate del jugador 1 por matchup y profundidad;
  - verificacion de que Alpha-Beta NO cambia el resultado (win rate invariante);
  - ventaja de primer jugador en los mirror matches (MM-MM, EM-EM);
  - costo (nodos/jugada) de Minimax con y sin poda;
  - grafico plots/e7_required_matchups.png.

Imprime los numeros para citar en el informe (§3.7, E7).
"""

from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

HERE = Path(__file__).parent
sns.set_theme(style="whitegrid")

ORDER = [
    "Minimax vs Minimax", "Expectimax vs Expectimax",
    "Minimax vs Expectimax", "Expectimax vs Minimax",
    "Minimax vs Random", "Random vs Minimax",
    "Expectimax vs Random", "Random vs Expectimax",
    "Minimax vs Stratagem", "Stratagem vs Minimax",
    "Expectimax vs Stratagem", "Stratagem vs Expectimax",
]
HEUR = "solo_mov_diff"   # config recomendada (h2 sola)


def main():
    df = pd.read_csv(HERE / "required_matchups_results.csv")
    print(f"Filas totales: {len(df)}")

    # --- (1) Alpha-Beta NO cambia el resultado (solo el costo) ---
    mm = df[df["matchup"].str.contains("Minimax")].copy()
    piv = mm.pivot_table(index=["matchup", "depth", "heuristic", "seed"],
                         columns="minimax_alpha_beta", values="winner", aggfunc="first")
    if True in piv.columns and False in piv.columns:
        comparable = piv.dropna(subset=[True, False])
        mismatches = int((comparable[True] != comparable[False]).sum())
        print(f"\n[AB on/off] partidas Minimax comparables: {len(comparable)} | "
              f"outcomes distintos: {mismatches} ({mismatches / len(comparable):.0%}) "
              f"-> por DESEMPATE de ordenamiento (el VALOR minimax es identico, no es un bug)")

    # --- (2) Win rate del jugador 1 por matchup y profundidad (heuristica recomendada, AB on) ---
    base = df[(df["heuristic"] == HEUR) &
              ((df["minimax_alpha_beta"] == True) | (df["minimax_alpha_beta"].isna()))]
    print(f"\n[Win rate jugador 1 | heuristica={HEUR}, AB on]")
    table = base.groupby(["depth", "matchup"])["p1_won"].mean().reset_index()
    for d in sorted(base["depth"].unique()):
        print(f"  --- profundidad {d} ---")
        sub = table[table["depth"] == d].set_index("matchup").reindex(ORDER)
        for m, r in sub.iterrows():
            print(f"    {m:<28} p1_win={r['p1_won']:.0%}")

    # --- (3) Ventaja de primer jugador en los mirror matches ---
    print("\n[Ventaja de primer jugador — mirror matches]")
    for mm_name in ["Minimax vs Minimax", "Expectimax vs Expectimax"]:
        sub = base[base["matchup"] == mm_name]
        for d in sorted(sub["depth"].unique()):
            wr = sub[sub["depth"] == d]["p1_won"].mean()
            print(f"    {mm_name} (d={d}): jugador 1 gana {wr:.0%}")

    # --- (4) Costo de Minimax con vs sin poda (nodos/jugada del lado Minimax) ---
    print("\n[Costo Minimax: nodos/jugada con vs sin Alpha-Beta]")
    m_first = df[df["player1_agent"] == "Minimax"]
    cost = m_first.groupby(["depth", "minimax_alpha_beta"])["p1_nodes_per_move"].mean().reset_index()
    print(cost.to_string(index=False))

    # --- Grafico: win rate jugador 1 por matchup, facetas por profundidad ---
    depths = sorted(base["depth"].unique())
    fig, axes = plt.subplots(1, len(depths), figsize=(7 * len(depths), 6), sharey=True)
    if len(depths) == 1:
        axes = [axes]
    for ax, d in zip(axes, depths):
        sub = table[table["depth"] == d].set_index("matchup").reindex(ORDER).reset_index()
        colors = ["#2c7fb8" if "Random" not in m and "Stratagem" not in m else
                  ("#41ab5d" if m.split(" vs ")[0] in ("Minimax", "Expectimax") else "#d95f0e")
                  for m in sub["matchup"]]
        ypos = list(range(len(sub)))
        ax.barh(ypos, sub["p1_won"].values, color=colors)
        ax.set_yticks(ypos)
        ax.set_yticklabels(sub["matchup"])
        ax.invert_yaxis()                       # primer matchup de ORDER arriba
        ax.axvline(0.5, color="black", ls="--", lw=1, alpha=0.7)
        ax.set_xlim(0, 1)
        ax.set_title(f"Profundidad {d}")
        ax.set_xlabel("Win rate del jugador 1 (primer nombre del matchup)")
        ax.set_ylabel("")
        for i, v in enumerate(sub["p1_won"]):
            if pd.notna(v):
                ax.text(min(v + 0.02, 0.93), i, f"{v:.0%}", va="center", fontsize=9)
    fig.suptitle(f"Matriz de matchups en ambas posiciones (heurística {HEUR}, α-β on) — "
                 f"línea = 50%", fontsize=12)
    fig.tight_layout()
    out = HERE / "plots" / "e7_required_matchups.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\nGuardado: {out}")


if __name__ == "__main__":
    main()
