import gymnasium as gym
import gymnasium_env

from stable_baselines3 import PPO

env = gym.make(
    "gymnasium_env/GridWorld-v0",
    render_mode="human"
)

model = PPO.load("ppo_gridworld")

obs, info = env.reset()

while True:
    action, _states = model.predict(obs)
    action = int(action)  # Convert from numpy array to int

    obs, reward, terminated, truncated, info = env.step(action)

    env.render()

    if terminated or truncated:
        obs, info = env.reset()