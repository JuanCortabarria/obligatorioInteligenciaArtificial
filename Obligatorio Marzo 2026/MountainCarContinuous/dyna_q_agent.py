"""
Dyna-Q tabular: Q-Learning + modelo aprendido + planning.

Implementación del algoritmo "Tabular Dyna-Q" (Sutton & Barto, Reinforcement
Learning: An Introduction, 2nd ed., Sec. 8.2, Figure 8.2):

    Inicializar Q(s, a) y Model(s, a) arbitrariamente
    Loop por siempre:
        (a) S ← estado actual no terminal
        (b) A ← ε-greedy(S, Q)
        (c) Tomar acción A; observar R, S'
        (d) Q(S, A) ← Q(S, A) + α [R + γ max_a Q(S', a) − Q(S, A)]
        (e) Model(S, A) ← R, S'         (asumiendo entorno determinista)
        (f) Repetir n veces (planning):
            S ← estado previamente observado al azar
            A ← acción previamente tomada en S al azar
            R, S' ← Model(S, A)
            Q(S, A) ← Q(S, A) + α [R + γ max_a Q(S', a) − Q(S, A)]

La hipótesis central del cap. 8.2 es que **un step real + n planning steps**
permite converger en muchos menos episodios reales que Q-Learning puro,
amortizando cada transición observada en n updates simulados. El precio es
mayor costo computacional *por episodio*, pero menor *número de episodios
necesarios*.

Nuestro entorno (MountainCarContinuous) es determinista, así que la versión
con `Model(s, a) ← (r, s')` (overwrite) del libro aplica directamente.
Para entornos estocásticos habría que guardar distribuciones o usar
Dyna-Q+ (sec. 8.3) — fuera del alcance de la consigna.

Nota sobre reward shaping: el modelo guarda el reward **shaped** (lo que el
agente efectivamente vio), no el reward crudo. Esto preserva la semántica
del problema: planning vuelve a "vivir" la misma señal de aprendizaje que
la experiencia real generó, sin re-aplicar shaping (que ya está integrado
en el reward almacenado).
"""

import random
from pathlib import Path
import pickle

import numpy as np

from discretization import Discretizer
from q_learning_agent import QLearningAgent


class DynaQAgent(QLearningAgent):
    def __init__(
        self,
        discretizer: Discretizer,
        planning_steps: int = 5,
        **kwargs,
    ):
        super().__init__(discretizer=discretizer, **kwargs)
        self.planning_steps = planning_steps
        # Model: clave = (state_tuple, action_idx), valor = (shaped_r, next_state_tuple, terminated)
        # Notar: usamos tuples como claves porque ndarray no es hasheable.
        self.model: dict = {}

    # ----- planning helper -----

    def _q_update(self, state, action_idx: int, reward: float, next_state, terminated: bool) -> None:
        """Q-Learning update genérico — comparte la regla con experiencia real y planning."""
        future = 0.0 if terminated else float(np.max(self.Q[next_state]))
        td_target = reward + self.gamma * future
        td_error = td_target - self.Q[state][action_idx]
        self.Q[state][action_idx] += self.alpha * td_error

    def _planning(self) -> None:
        """Loop interno de planning (paso (f) del algoritmo)."""
        if not self.model or self.planning_steps <= 0:
            return
        # Materializamos la lista de claves una vez por step real. Es O(|model|),
        # aceptable para tamaños típicos (< 10k entradas en MountainCar).
        keys = list(self.model.keys())
        for _ in range(self.planning_steps):
            key = random.choice(keys)
            state, action_idx = key
            shaped_r, next_state, terminated = self.model[key]
            self._q_update(state, action_idx, shaped_r, next_state, terminated)

    # ----- override del entrenamiento -----

    def train_agent(
        self,
        env,
        episodes: int = 1000,
        max_steps: int = 10_000,
        verbose_every: int = 100,
        env_seed: int | None = None,
        reset_epsilon: bool = True,
    ):
        """Loop de Dyna-Q. Por cada step real, hacemos `planning_steps` updates simulados."""
        if reset_epsilon:
            self.epsilon = self.epsilon_start

        history = {"rewards": [], "steps": [], "success": [], "epsilon": []}

        for ep in range(episodes):
            reset_seed = env_seed if (ep == 0 and env_seed is not None) else None
            obs, _ = env.reset(seed=reset_seed)
            state = self.discretizer.get_state(obs)
            total_reward = 0.0
            reached_goal = False
            step = 0

            for step in range(max_steps):
                action_idx = self._epsilon_greedy(state)
                real_action = self.discretizer.action_from_idx(action_idx)

                next_obs, reward, terminated, truncated, _ = env.step(real_action)
                next_state = self.discretizer.get_state(next_obs)

                if terminated:
                    reached_goal = True

                shaped_reward = self._shape(reward, obs, next_obs, terminated)

                # (d) Q-update con experiencia real.
                self._q_update(state, action_idx, shaped_reward, next_state, terminated)

                # (e) Update del modelo. Sobrescribimos: en entorno determinista
                # la última observación de (s, a) es válida; si hubiera estocasticidad
                # esta línea promediaría implícitamente — para eso está Dyna-Q+.
                self.model[(state, action_idx)] = (shaped_reward, next_state, terminated)

                # (f) Planning: n updates simulados a partir del modelo.
                self._planning()

                state = next_state
                obs = next_obs
                total_reward += reward  # historia sin shaping (comparable)

                if terminated or truncated:
                    break

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            history["rewards"].append(total_reward)
            history["steps"].append(step + 1)
            history["success"].append(1 if reached_goal else 0)
            history["epsilon"].append(self.epsilon)

            if verbose_every and (ep + 1) % verbose_every == 0:
                last = slice(max(0, ep - 99), ep + 1)
                avg_r = float(np.mean(history["rewards"][last]))
                rate = float(np.mean(history["success"][last]))
                print(
                    f"Ep {ep + 1:5d} | ε={self.epsilon:.3f} | "
                    f"avg_reward(100)={avg_r:7.2f} | "
                    f"success_rate(100)={rate:.2%} | "
                    f"|model|={len(self.model)}"
                )

        return history

    # ----- persistencia (extiende la del padre para incluir planning_steps y model) -----

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "Q": self.Q,
            "discretizer_config": self.discretizer.config(),
            "hyperparams": {
                "alpha": self.alpha,
                "gamma": self.gamma,
                "epsilon_start": self.epsilon_start,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
                "optimistic_init": self.optimistic_init,
                "reward_shaping": self.reward_shaping,
                "shaping_coef": self.shaping_coef,
                "planning_steps": self.planning_steps,
            },
            "model": self.model,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path) -> "DynaQAgent":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        disc = Discretizer(**payload["discretizer_config"])
        agent = cls(discretizer=disc, **payload["hyperparams"])
        agent.Q = payload["Q"]
        agent.model = payload.get("model", {})
        agent.epsilon = agent.epsilon_min
        return agent
