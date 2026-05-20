"""
Agente Q-Learning tabular para MountainCarContinuous-v0.

Implementación off-policy TD control (Sutton & Barto, sección 6.5).
Actualización:
    Q(s, a) ← Q(s, a) + α · [ r + γ · max_a' Q(s', a')  −  Q(s, a) ]

Política de comportamiento: ε-greedy con ε decayente.
Política aprendida: greedy sobre Q (off-policy → permite explorar
agresivamente sin contaminar la política objetivo).
"""

import pickle
import random
from pathlib import Path

import numpy as np

from discretization import Discretizer


class QLearningAgent:
    def __init__(
        self,
        discretizer: Discretizer,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        optimistic_init: float = 0.0,
        reward_shaping: bool = False,
        shaping_coef: float = 100.0,
        seed: int | None = None,
    ):
        self.discretizer = discretizer
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.optimistic_init = optimistic_init
        self.reward_shaping = reward_shaping
        self.shaping_coef = shaping_coef

        # Inicialización optimista: si optimistic_init > 0, todas las acciones lucen
        # atractivas hasta que la experiencia las "descuente", favoreciendo exploración
        # (Sutton & Barto, sec. 2.6).
        self.Q = np.full(discretizer.q_shape, optimistic_init, dtype=np.float64)
        self.epsilon = epsilon_start

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    # ----- política -----

    def _epsilon_greedy(self, state) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.discretizer.n_actions - 1)
        return int(np.argmax(self.Q[state]))

    def next_action(self, obs):
        """Acción greedy (para test/demo, sin exploración)."""
        state = self.discretizer.get_state(obs)
        action_idx = int(np.argmax(self.Q[state]))
        return self.discretizer.action_from_idx(action_idx)

    # ----- reward shaping -----

    def _shape(self, reward: float, obs, next_obs) -> float:
        """
        Potential-based reward shaping (Ng, Harada & Russell, 1999):
            F(s, s') = γ · Φ(s') − Φ(s)
            shaped_reward = reward + F(s, s')
        con Φ(s) = shaping_coef · |v|.

        Esta forma de shaping tiene la propiedad teórica de no cambiar la
        política óptima — solo acelera el aprendizaje. Premia *aumentos*
        de |v| (γ·|v'| > |v|) y penaliza *bajadas*, lo que empuja al agente
        a acumular momento hacia la meta sin caer en la trampa de "oscilar
        para siempre acumulando bonus" (que ocurría con un shaping aditivo simple).

        Importante: el shaping NO se aplica al reward terminal (+100).
        """
        if not self.reward_shaping:
            return reward
        if reward >= 99.0:
            return reward
        _, v = obs
        _, v_next = next_obs
        phi = self.shaping_coef * abs(v)
        phi_next = self.shaping_coef * abs(v_next)
        return reward + self.gamma * phi_next - phi

    # ----- entrenamiento -----

    def train_agent(
        self,
        env,
        episodes: int = 1000,
        max_steps: int = 999,
        verbose_every: int = 100,
        env_seed: int | None = None,
    ):
        """
        Loop de entrenamiento. Devuelve diccionario con historia para graficar:
            - rewards: reward acumulado por episodio (sin shaping, para comparar entre configs)
            - steps: steps hasta done por episodio
            - success: 1 si llegó a la meta, 0 si no
            - epsilon: ε al final de cada episodio
        """
        history = {"rewards": [], "steps": [], "success": [], "epsilon": []}

        for ep in range(episodes):
            # Seedear solo el primer reset; los siguientes usan el RNG ya inicializado del env.
            # Esto da reproducibilidad sin matar la diversidad entre episodios.
            reset_seed = env_seed if (ep == 0 and env_seed is not None) else None
            obs, _ = env.reset(seed=reset_seed)
            state = self.discretizer.get_state(obs)
            total_reward = 0.0
            reached_goal = False

            for step in range(max_steps):
                action_idx = self._epsilon_greedy(state)
                real_action = self.discretizer.action_from_idx(action_idx)

                next_obs, reward, terminated, truncated, _ = env.step(real_action)
                done = terminated or truncated

                if terminated and reward >= 99.0:
                    reached_goal = True

                shaped_reward = self._shape(reward, obs, next_obs)
                next_state = self.discretizer.get_state(next_obs)

                # Update Q-Learning:
                # Q[s,a] += α · (r + γ · max_a' Q[s',a'] − Q[s,a])
                # Si done, el bootstrap futuro es 0.
                future = 0.0 if done else np.max(self.Q[next_state])
                td_target = shaped_reward + self.gamma * future
                td_error = td_target - self.Q[state][action_idx]
                self.Q[state][action_idx] += self.alpha * td_error

                state = next_state
                obs = next_obs
                total_reward += reward  # reward sin shaping para historia limpia

                if done:
                    break

            # Decay de ε al final de cada episodio.
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
                    f"avg_reward(100)={avg_r:7.2f} | success_rate(100)={rate:.2%}"
                )

        return history

    # ----- evaluación -----

    def test_agent(self, env, episodes: int = 10, max_steps: int = 999, render: bool = False) -> dict:
        """Evalúa la política greedy actual sin exploración."""
        rewards = []
        successes = 0
        steps_list = []
        for _ in range(episodes):
            obs, _ = env.reset()
            total_reward = 0.0
            for step in range(max_steps):
                action = self.next_action(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                if render:
                    env.render()
                if terminated or truncated:
                    if terminated and reward >= 99.0:
                        successes += 1
                    break
            rewards.append(total_reward)
            steps_list.append(step + 1)
        return {
            "avg_reward": float(np.mean(rewards)),
            "success_rate": successes / episodes,
            "avg_steps": float(np.mean(steps_list)),
            "rewards": rewards,
        }

    # ----- persistencia -----

    def save(self, path: str | Path) -> None:
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
            },
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str | Path) -> "QLearningAgent":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        disc = Discretizer(**payload["discretizer_config"])
        agent = cls(discretizer=disc, **payload["hyperparams"])
        agent.Q = payload["Q"]
        agent.epsilon = agent.epsilon_min  # tras cargar, no queremos seguir explorando por defecto
        return agent
