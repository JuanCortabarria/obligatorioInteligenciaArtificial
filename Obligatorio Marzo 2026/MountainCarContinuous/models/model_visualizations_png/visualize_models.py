import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

MODEL_FILES = {
    "smoke_test": Path("models/smoke_test.pkl"),
    "q_learning_best": Path("models/q_learning_best.pkl"),
    "dyna_q_best": Path("models/dyna_q_best.pkl"),
}

OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(exist_ok=True)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_q_visualizations(model_name, data):
    Q = data["Q"]

    max_q = np.max(Q, axis=2)
    best_action = np.argmax(Q, axis=2)

    plt.figure(figsize=(7, 6))
    plt.imshow(max_q, origin="lower", aspect="auto")
    plt.colorbar(label="Máximo valor Q")
    plt.xlabel("Bin de velocidad / variable 2")
    plt.ylabel("Bin de posición / variable 1")
    plt.title(f"{model_name}: mejor valor Q por estado")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{model_name}_max_q_heatmap.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 6))
    plt.imshow(best_action, origin="lower", aspect="auto")
    plt.colorbar(label="Acción elegida")
    plt.xlabel("Bin de velocidad / variable 2")
    plt.ylabel("Bin de posición / variable 1")
    plt.title(f"{model_name}: política aprendida")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{model_name}_policy.png", dpi=200)
    plt.close()

def save_model_coverage(model_name, data):
    model = data.get("model")

    if not isinstance(model, dict):
        return

    transitions_per_state = np.zeros((41, 41))

    for key in model.keys():
        try:
            state, action = key
            x, v = state
            transitions_per_state[x, v] += 1
        except Exception:
            pass

    plt.figure(figsize=(7, 6))
    plt.imshow(transitions_per_state, origin="lower", aspect="auto")
    plt.colorbar(label="Cantidad de acciones/modelos aprendidos")
    plt.xlabel("Bin de velocidad / variable 2")
    plt.ylabel("Bin de posición / variable 1")
    plt.title(f"{model_name}: cobertura del modelo interno")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{model_name}_model_coverage.png", dpi=200)
    plt.close()

for model_name, model_path in MODEL_FILES.items():
    data = load_pickle(model_path)
    save_q_visualizations(model_name, data)
    save_model_coverage(model_name, data)

print("Visualizaciones generadas correctamente.")
