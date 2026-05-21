import gymnasium as gym
import gymnasium_env

from stable_baselines3 import PPO

env = gym.make("gymnasium_env/GridWorld-v0")

model = PPO("MultiInputPolicy", env, verbose=1)

model.learn(total_timesteps=100000)

model.save("ppo_gridworld")