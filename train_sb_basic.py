import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

def train_and_run_mountain_car():
    # Make the environment with vectorized wrappers for automatic reset
    env = make_vec_env('MountainCar-v0', n_envs=1, env_kwargs={"render_mode": "human"})

    # Instantiate the agent
    model = DQN('MlpPolicy', env, verbose=1)

    # Train the agent
    model.learn(total_timesteps=10000)

    # Evaluate the trained agent
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")

    # Enjoy trained agent
    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        env.render()

    env.close()


if __name__ == "__main__":
    train_and_run_mountain_car()
