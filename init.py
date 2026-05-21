import gymnasium as gym
import gymnasium_env

# Create our training environment - a cart with a pole that needs balancing
env = gym.make("gymnasium_env/GridWorld-v0", render_mode="human")
# Reset environment to start a new episode
observation, info = env.reset()

print(f"Starting observation: {observation}")

episode_over = False
total_reward = 0

for _ in range(1000):
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    env.render()

    if terminated or truncated:
        obs, info = env.reset()

print(f"Episode finished! Total reward: {total_reward}")
env.close()