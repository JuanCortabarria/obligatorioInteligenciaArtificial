"""
Utilidades de discretización para MountainCarContinuous-v0.

El ambiente nos da observaciones y acciones continuas:
    - obs = (x, v)   con x ∈ [-1.2, 0.6], v ∈ [-0.07, 0.07]
    - action = a ∈ [-1, 1]   (fuerza aplicada al carro)

Para usar Q-Learning tabular necesitamos pasar todo a un espacio finito
de estados y acciones. Esta clase encapsula esa conversión y deja los
parámetros (cantidad de bins por dimensión, cantidad de acciones) configurables
para poder comparar configuraciones distintas en el grid search.
"""

import numpy as np


class Discretizer:
    # Límites duros del ambiente MountainCarContinuous-v0 (definidos por Gymnasium).
    X_MIN, X_MAX = -1.2, 0.6
    V_MIN, V_MAX = -0.07, 0.07
    A_MIN, A_MAX = -1.0, 1.0

    def __init__(self, n_bins_x: int = 40, n_bins_v: int = 40, n_actions: int = 5):
        self.n_bins_x = n_bins_x
        self.n_bins_v = n_bins_v
        self.n_actions = n_actions

        # np.linspace incluye los extremos. np.digitize devuelve índices en [0, n_bins].
        self.x_space = np.linspace(self.X_MIN, self.X_MAX, n_bins_x)
        self.v_space = np.linspace(self.V_MIN, self.V_MAX, n_bins_v)
        self.actions = np.linspace(self.A_MIN, self.A_MAX, n_actions)

    @property
    def state_shape(self):
        # +1 porque np.digitize puede devolver índice = len(bins) (valores > último bin).
        return (self.n_bins_x + 1, self.n_bins_v + 1)

    @property
    def q_shape(self):
        return (*self.state_shape, self.n_actions)

    def get_state(self, obs):
        x, v = obs
        x_bin = int(np.digitize(x, self.x_space))
        v_bin = int(np.digitize(v, self.v_space))
        return x_bin, v_bin

    def action_from_idx(self, idx: int) -> np.ndarray:
        # El env espera un np.array shape (1,) con el valor de la fuerza.
        return np.array([self.actions[idx]], dtype=np.float32)

    def get_action_idx(self, a: float) -> int:
        # Inverso aproximado: encuentra la acción discreta más cercana al valor continuo.
        return int(np.argmin(np.abs(self.actions - a)))

    def config(self) -> dict:
        return {
            "n_bins_x": self.n_bins_x,
            "n_bins_v": self.n_bins_v,
            "n_actions": self.n_actions,
        }
