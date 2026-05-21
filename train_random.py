import gymnasium as gym
import gymnasium_env

from agents.random_agent import RandomAgent


env = gym.make(
    "gymnasium_env/GridWorld-v0",
    render_mode="human"
)

agent = RandomAgent(
    env.action_space
)

EPISODES = 10

for episode in range(EPISODES):

    observation, info = env.reset()

    done = False

    total_reward = 0

    while not done:

        action = agent.predict(
            observation
        )

        observation, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        env.render()

        total_reward += reward

    print(
        f"Episode {episode} | Reward: {total_reward}"
    )

env.close()