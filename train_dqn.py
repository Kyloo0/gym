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
    "gymnasium_env/GridWorld-v0"
)
env = gym.wrappers.TimeLimit(
    env,
    max_episode_steps=100
)

obs, info = env.reset()

obs = flatten_observation(obs)

observation_dim = len(obs)

action_dim = env.action_space.n


# =========================================================
# AGENT
# =========================================================

agent = DQNAgent(
    observation_dim,
    action_dim
)

EPISODES = 500

TARGET_UPDATE = 10


# =========================================================
# TRAIN LOOP
# =========================================================

for episode in range(EPISODES):

    obs, info = env.reset()

    obs = flatten_observation(obs)

    done = False

    total_reward = 0

    step_count = 0

    while not done:
        step_count += 1
        action = agent.predict(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_obs = flatten_observation(next_obs)

        agent.replay_buffer.push(
            obs,
            action,
            reward,
            next_obs,
            done
        )

        agent.train_step()
        obs = next_obs
        total_reward += reward
        done = terminated or truncated

    if episode % TARGET_UPDATE == 0:

        agent.update_target_network()

    agent.decay_epsilon()

    print(
        f"Episode {episode}"
        f" | Reward {total_reward}"
        f" | Epsilon {agent.epsilon:.3f}"
    )

agent.save("dqn_model.pth")

print("DQN model saved!")