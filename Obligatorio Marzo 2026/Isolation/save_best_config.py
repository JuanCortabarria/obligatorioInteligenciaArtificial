"""Regenera `mate_best_config.pkl` a partir de `results.csv` (experimentos E2/E3/E6).

MATE no entrena un modelo: este artefacto serializa la **configuracion computada** por la
experimentacion (tecnica + profundidad + ponderacion ganadoras), de forma **reproducible y
recargable**. Los numeros se **derivan de `results.csv`** (no se hardcodean), replicando la
logica de la celda del notebook que elige:
  - la **tecnica** por su win rate vs Stratagem a la mayor profundidad medida (E3);
  - la **ponderacion** por el torneo round-robin de heuristicas (E6).
"""

import pickle

import pandas as pd

WEIGHT_SETS = {
    "solo_mov_diff": {"h2": 1.0},
    "mov+centro": {"h2": 1.0, "h3": 0.5},
    "mov+acorralar": {"h2": 1.0, "h4": 1.0},
    "balanceada": {"h1": 1.0, "h2": 2.0, "h3": 0.5, "h4": 1.0},
}


def main():
    df = pd.read_csv("results.csv")

    # E3 — mejor tecnica vs Stratagem a la mayor profundidad medida.
    e3 = df[df["experiment"] == "E3"].copy()
    e3["tech"] = e3["agent"].str.split("(").str[0]
    e3["d"] = e3["agent"].str.extract(r"d=(\d+)").astype(int)
    dmax = int(e3["d"].max())
    wr_strat = e3[e3["d"] == dmax].groupby("tech")["a_won"].mean()
    best_tech = wr_strat.idxmax()

    # E2 — win rate vs Random de la tecnica ganadora.
    wr_rand = df[df["experiment"] == "E2"].groupby("agent")["a_won"].mean()

    # E6 — mejor ponderacion (round-robin: cada heuristica como agente y como rival).
    e6 = df[df["experiment"] == "E6"]
    heurs = sorted(set(e6["agent"]) | set(e6["opponent"]))

    def heur_winrate(h):
        as_agent = e6.loc[e6["agent"] == h, "a_won"]
        as_opponent = 1 - e6.loc[e6["opponent"] == h, "a_won"]
        return float(pd.concat([as_agent, as_opponent]).mean())

    wr_heur = {h: heur_winrate(h) for h in heurs}
    best_weights_name = max(wr_heur, key=wr_heur.get)

    best_config = {
        "tecnica": best_tech.lower(),
        "profundidad": dmax,
        "pesos": WEIGHT_SETS[best_weights_name],
        "pesos_nombre": best_weights_name,
        "metricas": {
            "win_rate_vs_stratagem_dmax": round(float(wr_strat.max()), 3),
            "win_rate_vs_random": round(float(wr_rand.get(best_tech, float("nan"))), 3),
            "e6_mejor_winrate_promedio": round(float(wr_heur[best_weights_name]), 3),
        },
    }

    with open("mate_best_config.pkl", "wb") as f:
        pickle.dump(best_config, f)
    with open("mate_best_config.pkl", "rb") as f:
        loaded = pickle.load(f)
    print("mate_best_config.pkl guardado y recargado OK:")
    print(loaded)


if __name__ == "__main__":
    main()
