import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Callable

import gymnasium as gym
import wandb
from stable_baselines3 import PPO, A2C, DQN, SAC, TD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecVideoRecorder, VecEnv
from wandb.integration.sb3 import WandbCallback

algorithms = {
    'PPO': PPO,
    'A2C': A2C,
    'DQN': DQN,
    'SAC': SAC,
    'TD3': TD3,
}


class MultiSegmentVecVideoRecorder(VecVideoRecorder):
    """
    Extended VecVideoRecorder that updates video_name/video_path
    each time recording starts, avoiding overwriting old files.
    """

    def __init__(
            self,
            venv: VecEnv,
            video_folder: str,
            record_video_trigger: Callable[[int], bool],
            video_length: int = 200,
            name_prefix: str = "rl-video",
            log_to_wandb: bool = True,
    ):
        super().__init__(
            venv=venv,
            video_folder=video_folder,
            record_video_trigger=record_video_trigger,
            video_length=video_length,
            name_prefix=name_prefix,
        )
        self.log_to_wandb = log_to_wandb

    def _start_recording(self) -> None:
        if self.recording:
            self._stop_recording()
        self.video_name = f"{self.name_prefix}-step-{self.step_id}-to-step-{self.step_id + self.video_length}.mp4"
        self.video_path = os.path.join(self.video_folder, self.video_name)
        self.recording = True

    def _stop_recording(self) -> None:
        """Stop current recording, save video, and log it to wandb."""
        super()._stop_recording()
        if self.log_to_wandb and wandb.run is not None and os.path.exists(self.video_path):
            wandb.log({"video": wandb.Video(self.video_path)})


class MountainCarRewardWrapper(gym.RewardWrapper):
    def __init__(self, env, velocity_scale=10.0):
        super().__init__(env)
        self.velocity_scale = velocity_scale

    def reward(self, reward):
        velocity = abs(self.env.unwrapped.state[1])
        reward += velocity * self.velocity_scale
        return reward


def make_env(env_name, seed=None):
    def _init():
        env = gym.make(env_name, render_mode="rgb_array")
        if seed is not None:
            env.reset(seed=seed)
        if 'MountainCar' in env_name:
            env = MountainCarRewardWrapper(env)
        return Monitor(env)
    return _init



def create_model(args, env, log_dir, policy):
    algorithm = algorithms[args.algorithm]
    if args.algorithm == 'A2C':
        model = algorithm(policy, env, verbose=1, tensorboard_log=log_dir, learning_rate=args.learning_rate,
                          gamma=args.gamma, seed=args.seed)
    else:
        model = algorithm(policy, env, verbose=1, tensorboard_log=log_dir, learning_rate=args.learning_rate,
                          gamma=args.gamma, seed=args.seed, batch_size=args.batch_size)
    return model


def main(args: argparse.Namespace):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_name = f'{args.algorithm}_{args.env_name}_{timestamp}'
    wandb.init(project=args.project, config=vars(args), sync_tensorboard=True, save_code=True, name=run_name,
               id=run_name, mode="online", group=args.env_name, job_type=args.algorithm, tags=args.wandb_tags)
    run_name = wandb.run.name
    base_path = Path(__file__).parent.parent.resolve()
    log_dir = f"{base_path}/logs/{run_name}"
    os.makedirs(log_dir, exist_ok=True)

    # env = DummyVecEnv([lambda: make_env(args.env_name)])
    env = make_vec_env(lambda: make_env(args.env_name, seed=args.seed), n_envs=args.n_envs)
    video_length = args.video_length
    env = MultiSegmentVecVideoRecorder(
        env,
        f"{log_dir}/videos",
        record_video_trigger=lambda x: x % args.record_freq == 0,
        video_length=video_length,
        name_prefix=args.env_name,
    )

    policy = "CnnPolicy" if 'CarRacing' in args.env_name else "MlpPolicy"

    model = create_model(args, env, log_dir, policy)

    eval_callback = EvalCallback(
        env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=args.eval_freq,
        deterministic=True
    )

    wandb_cb = WandbCallback(
        gradient_save_freq=args.gradient_save_freq,
        model_save_freq=args.model_save_freq,
        model_save_path=log_dir,
        verbose=2
    )

    model.learn(
        total_timesteps=args.max_steps,
        callback=[eval_callback, wandb_cb]
    )

    model.save(f"{log_dir}/final_model")
    wandb.save(f"{log_dir}/final_model.zip")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='PPO', help='RL algorithm',
                        choices=['DQN', 'A2C', 'PPO', 'SAC', 'TD3'])
    parser.add_argument('--project', type=str, default='ExperimentalSetup', help='Wandb project name')
    parser.add_argument('--env_name', type=str, default='LunarLander-v3', help='Environment name',
                        choices=['FrozenLake-v1', 'MountainCar-v0', 'MountainCarContinuous-v0', 'CartPole-v1',
                                 'Acrobot-v1', 'Pendulum-v1', 'LunarLander-v3', 'CarRacing-v3'])
    parser.add_argument('--seed', type=int, default=42, help='Seed for the pseudo random generators')
    parser.add_argument('--n_envs', type=int, default=4, help='Number of parallel environments')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate for the optimizer')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--max_steps', type=int, default=1e6, help='Maximum number of steps')
    parser.add_argument('--eval_freq', type=int, default=5000, help='Frequency of evaluations')
    parser.add_argument('--model_save_freq', type=int, default=5000, help='Frequency of saving the model')
    parser.add_argument('--gradient_save_freq', type=int, default=100, help='Frequency of saving the model')
    parser.add_argument('--record_freq', type=int, default=5000, help='Frequency of recording episodes')
    parser.add_argument('--video_length', type=int, default=1000, help='Length of the recording')
    parser.add_argument('--wandb_tags', type=str, nargs='+', default=[], help='Tags to denote runs')
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
