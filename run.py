import gymnasium as gym


def run_mountain_car():
    # Create the MountainCar environment
    env = gym.make('CartPole-v1', render_mode="human")

    # Number of episodes to run
    num_episodes = 3

    for episode in range(num_episodes):
        # Reset the environment to start a new episode
        env.reset()
        for t in range(100):
            # Render the environment
            env.render()

            # Take a random action
            action = env.action_space.sample()

            # Execute the action and get the new state, reward, done, and info
            observation, reward, truncated, done, info = env.step(action)

            # Check if the episode is done
            if done:
                print(f"Episode {episode + 1} finished after {t + 1} timesteps")
                break
        else:
            print(f"Episode {episode + 1} reached the maximum timestep limit")

    # Close the environment
    env.close()


if __name__ == "__main__":
    run_mountain_car()
