import argparse
import os
from datetime import datetime

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from wandb.integration.sb3 import WandbCallback

import wandb

algorithms = {
    'PPO': PPO,
    'A2C': A2C,
    'DQN': DQN,
}


def main(args: argparse.Namespace) -> None:

    print("HELLO")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")

    run_name = f'{args.algorithm}_{args.env_name}_{timestamp}'

    # Configure wandb
    wandb.init(project=args.project, config=vars(args), entity="tu-e", sync_tensorboard=True, name=run_name,
               monitor_gym=True, save_code=True, id=run_name, mode="online", group=args.env_name)

    # Create and wrap the environment
    env = make_vec_env(args.env_name, n_envs=args.num_envs)

    # Define the model
    algorithm = algorithms[args.algorithm]
    model = algorithm("MlpPolicy", env, learning_rate=args.learning_rate, gamma=args.gamma, verbose=1,
                      tensorboard_log="./tensorboard/", device="cuda")

    # Callbacks
    video_folder = f"./videos/{run_name}"
    os.makedirs(video_folder, exist_ok=True)
    eval_env = gym.make(args.env_name, render_mode="rgb_array")
    eval_env = RecordVideo(eval_env, video_folder=video_folder,
                           episode_trigger=lambda episode_id: episode_id % args.record_freq == 0,
                           name_prefix=f"{args.algorithm}_{args.env_name}")

    # Instantiate the VideoLoggerCallback
    video_logger_callback = VideoLoggerCallback(video_folder="./videos/", log_freq=args.record_freq)

    eval_callback = EvalCallback(eval_env, best_model_save_path='../logs/',
                                 log_path='../logs/', eval_freq=args.eval_freq,
                                 deterministic=True, render=False,
                                 n_eval_episodes=args.n_eval_episodes)

    # stop_callback = StopTrainingOnRewardThreshold(reward_threshold=args.reward_threshold, verbose=1)
    checkpoint_callback = CheckpointCallback(save_freq=args.save_freq, save_path='../logs/', name_prefix='model')
    wandb_callback = WandbCallback(gradient_save_freq=args.log_interval, verbose=2)

    # Start the training
    model.learn(total_timesteps=args.max_steps, log_interval=args.log_interval,
                callback=[eval_callback, checkpoint_callback, wandb_callback])

    # Close the environment
    env.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='PPO', choices=['DQN', 'A2C', 'PPO'], help='RL algorithm')
    parser.add_argument('--project', type=str, default='BasicExperiments', help='Wandb project name')
    parser.add_argument('--num_envs', type=int, default=10, help='Number of parallel environments')
    parser.add_argument('--env_name', type=str, default='CartPole-v1', help='Environment name')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the optimizer')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--policy_kwargs', type=str, default='dict(activation_fn=torch.nn.ReLU, net_arch=[64, 64])',
                        help='Policy kwargs for DQN')
    parser.add_argument('--max_steps', type=int, default=4e5, help='Maximum number of steps')
    parser.add_argument('--eval_freq', type=int, default=250, help='Frequency of evaluations')
    parser.add_argument('--save_freq', type=int, default=5e4, help='Frequency of saving the model')
    parser.add_argument('--n_eval_episodes', type=int, default=10, help='Number of episodes per evaluation')
    parser.add_argument('--log_interval', type=int, default=100, help='Log interval')
    parser.add_argument('--reward_threshold', type=float, default=400, help='Reward threshold to stop training')
    parser.add_argument('--record_freq', type=int, default=50, help='Frequency of recording episodes')
    return parser.parse_args()


class VideoLoggerCallback(BaseCallback):
    def __init__(self, video_folder: str, log_freq: int, verbose=1):
        super(VideoLoggerCallback, self).__init__(verbose)
        self.video_folder = video_folder
        self.log_freq = log_freq
        self.logged_videos = set()

    def _on_step(self) -> bool:
        # Check if it's time to log videos based on log_freq
        if self.n_calls % self.log_freq == 0:
            self._log_videos()
        return True

    def _log_videos(self):
        # Find all video files in the video_folder
        new_videos = [f for f in os.listdir(self.video_folder) if f.endswith('.mp4') and f not in self.logged_videos]
        for video_file in new_videos:
            # Log the video to wandb
            wandb.log({"videos": wandb.Video(os.path.join(self.video_folder, video_file), fps=4, format="mp4")})
            self.logged_videos.add(video_file)
            if self.verbose > 0:
                print(f"Logged video {video_file} to wandb.")


if __name__ == "__main__":
    main(parse_args())
