"""
Agente Q-Learning tabular para MountainCarContinuous-v0.

Implementa la regla de actualización off-policy TD control vista en el material
de clase (QL.pdf, slide 6) y en Sutton & Barto, sección 6.5:

    Inicializar Q(s, a) arbitrariamente, ∀ s ∈ S, a ∈ A(s)
    Repetir (por episodio):
        Inicializar s
        Repetir hasta done:
            Con probabilidad ε:
                a ← sample(A(s))             (* explorar *)
            sino:
                a ← argmax_a' Q(s, a')       (* explotar *)
            s', r, done ← step(a)
            Q(s, a) ← Q(s, a) + α · ( r + γ · max_a' Q(s', a') − Q(s, a) )
            s ← s'

Diferencias importantes con el pseudocódigo "puro":

1. **Distinción `terminated` vs `truncated`** (API Gymnasium > 0.26):
   El slide del curso trata done como un único flag. Gymnasium separa:
       - `terminated`: el episodio llegó a un estado terminal del MDP.
                       → bootstrap futuro = 0 (no hay más decisiones que tomar).
       - `truncated` : timeout u otro corte artificial. El estado *no* es
                       terminal del MDP.
                       → bootstrap futuro debe seguir siendo γ·max_a' Q(s', a').
   Tratar truncated como terminated (lo que hace el flag `done` agregado) es
   un bug clásico que sesga Q hacia abajo en problemas con timeout.

2. **Política de comportamiento ε-greedy con decay exponencial por episodio**.
3. **Reward shaping opcional, potential-based** (Ng-Harada-Russell 1999),
   que no cambia la política óptima.
4. **Inicialización opcionalmente optimista** (Sutton & Barto sec. 2.6).
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
        random_init: tuple[float, float] | None = None,
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
        self.random_init = random_init
        self.reward_shaping = reward_shaping
        self.shaping_coef = shaping_coef

        # Sembramos ANTES de inicializar Q para que la init aleatoria sea reproducible.
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Inicialización de Q(s, a). Tres esquemas (la estrategia de exploración —ε-greedy—
        # es la misma en todos; lo que cambia es el VALOR INICIAL de Q):
        #   - random_init=(low, high): valores aleatorios U(low, high) — alternativa "menos
        #     arbitraria" que elegir un valor optimista fijo (sugerencia de la cátedra).
        #   - optimistic_init>0: constante alta → las acciones no probadas "se ven mejores",
        #     forzando exploración estructurada (Sutton & Barto §2.6).
        #   - optimistic_init=0 (y sin random_init): inicialización en 0 (caso de referencia).
        if random_init is not None:
            low, high = random_init
            self.Q = np.random.uniform(low, high, size=discretizer.q_shape).astype(np.float64)
        else:
            self.Q = np.full(discretizer.q_shape, optimistic_init, dtype=np.float64)
        self.epsilon = epsilon_start

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

    def _shape(self, reward: float, obs, next_obs, terminated: bool) -> float:
        """
        Potential-based reward shaping (Ng, Harada & Russell, 1999):

            F(s, s') = γ · Φ(s') − Φ(s)
            shaped_reward = reward + F(s, s')

        con Φ(s) = shaping_coef · |v|. Premia *aumentos* de |v| y penaliza
        *bajadas*, empujando al agente a acumular momento.

        Teorema NHR-99: shaping potential-based no cambia la política óptima.
        Solo acelera la propagación de la señal de aprendizaje.

        Estado terminal: por convención NHR-99 en MDPs episódicos, Φ(s') = 0
        cuando s' es terminal. Esto SÍ se aplica acá: shaped_r = r + γ·0 − Φ(s)
        = r − Φ(s). NO devolvemos `reward` puro — eso sería equivalente a
        olvidar el bonus de potencial que el agente ya "acumuló" entrando a
        la meta, sobre-estimando el valor del estado pre-terminal.
        """
        if not self.reward_shaping:
            return reward
        _, v = obs
        _, v_next = next_obs
        phi = self.shaping_coef * abs(v)
        phi_next = 0.0 if terminated else self.shaping_coef * abs(v_next)
        return reward + self.gamma * phi_next - phi

    # ----- entrenamiento -----

    def train_agent(
        self,
        env,
        episodes: int = 1000,
        max_steps: int = 10_000,
        verbose_every: int = 100,
        env_seed: int | None = None,
        reset_epsilon: bool = True,
    ):
        """
        Loop de entrenamiento Q-Learning. Devuelve historial por episodio.

        Args:
            env: ambiente Gymnasium (típicamente MountainCarContinuous-v0).
            episodes: cantidad de episodios a correr.
            max_steps: tope duro de steps por episodio. Por defecto muy alto
                       porque confiamos en `truncated` del wrapper TimeLimit
                       del env (999 para este env). Si el env no tuviera
                       TimeLimit, esto previene loops infinitos.
            verbose_every: cada cuántos episodios imprimir progreso (0 = silencio).
            env_seed: seed para el PRIMER reset del env. Los resets posteriores
                      heredan el RNG ya inicializado. Si None, no se seedea.
            reset_epsilon: si True (default), ε se resetea a epsilon_start al
                           inicio del entrenamiento. Garantiza reproducibilidad
                           entre corridas independientes.

        Returns:
            dict con listas por episodio: rewards (sin shaping), steps,
            success (0/1), epsilon.
        """
        if reset_epsilon:
            self.epsilon = self.epsilon_start

        history = {"rewards": [], "steps": [], "success": [], "epsilon": []}

        for ep in range(episodes):
            # Seedear solo el primer reset; los siguientes usan el RNG ya inicializado.
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

                # En MountainCarContinuous-v0, terminated=True ⇔ alcanzó la meta.
                if terminated:
                    reached_goal = True

                shaped_reward = self._shape(reward, obs, next_obs, terminated)

                # Q-Learning update.
                # CRUCIAL: bootstrap futuro = 0 SOLO si terminated. Si truncated
                # (timeout), s' NO es terminal del MDP — bootstrappear normalmente.
                future = 0.0 if terminated else float(np.max(self.Q[next_state]))
                td_target = shaped_reward + self.gamma * future
                td_error = td_target - self.Q[state][action_idx]
                self.Q[state][action_idx] += self.alpha * td_error

                state = next_state
                obs = next_obs
                total_reward += reward  # reward sin shaping (historia comparable)

                if terminated or truncated:
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

    def test_agent(self, env, episodes: int = 10, max_steps: int = 10_000, render: bool = False,
                   test_seeds=None) -> dict:
        """Evalúa la política greedy actual sin exploración.

        Si se pasa `test_seeds` (lista de semillas), cada episodio de test arranca
        sembrado con una de esas semillas. Esto fija el conjunto de estados iniciales
        de evaluación, de modo que la métrica de test sea **comparable entre
        configuraciones** (refleja la política, no la suerte del reset). Si es None,
        el comportamiento es el anterior (resets sin semilla).
        """
        rewards = []
        successes = 0
        steps_list = []
        for i in range(episodes):
            if test_seeds is not None:
                obs, _ = env.reset(seed=int(test_seeds[i % len(test_seeds)]))
            else:
                obs, _ = env.reset()
            total_reward = 0.0
            step = 0
            for step in range(max_steps):
                action = self.next_action(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                if render:
                    env.render()
                if terminated or truncated:
                    if terminated:
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
                "random_init": self.random_init,
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
        agent.epsilon = agent.epsilon_min  # tras cargar, no exploramos por defecto
        return agent
