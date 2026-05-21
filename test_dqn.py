import numpy as np

import gymnasium as gym
import gymnasium_env

from agents.dqn_agent import DQNAgent


# =========================================================
# PREPROCESS OBSERVATION
# =========================================================

def flatten_observation(obs):

    if isinstance(obs, dict):

        flat = []

        for value in obs.values():
            flat.extend(
                np.array(value).flatten()
            )

        return np.array(
            flat,
            dtype=np.float32
        )

    return np.array(
        obs,
        dtype=np.float32
    )


# =========================================================
# ENV
# =========================================================

env = gym.make(
    "gymnasium_env/GridWorld-v0",
    render_mode="human"
)

obs, info = env.reset()

obs = flatten_observation(obs)

observation_dim = len(obs)

action_dim = env.action_space.n


# =========================================================
# LOAD MODEL
# =========================================================

agent = DQNAgent(
    observation_dim,
    action_dim
)

agent.load("dqn_model.pth")

agent.epsilon = 0.0


# =========================================================
# TEST LOOP
# =========================================================

done = False

while not done:

    action = agent.predict(obs)

    next_obs, reward, terminated, truncated, info = env.step(action)

    done = terminated or truncated

    env.render()

    obs = flatten_observation(next_obs)

env.close()