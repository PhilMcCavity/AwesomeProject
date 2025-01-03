import argparse
import os
from datetime import datetime

import torch
from tianshou.utils import WandbLogger

import wandb
from gymnasium import make
from gymnasium.wrappers import RecordVideo
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv
from tianshou.policy import DQNPolicy
from tianshou.trainer import offpolicy_trainer
# from tianshou.utils import WandbLogger
from tianshou.utils.net.common import Net
from torch.utils.tensorboard import SummaryWriter


def make_env(task, seed=0, record_video=False, record_interval=10, video_dir="./videos"):
    def _init():
        env = make(task, render_mode="rgb_array")
        if record_video:
            env = RecordVideo(env, video_dir, episode_trigger=lambda ep: ep % record_interval == 0, name_prefix=task)
        return env

    return _init


def get_envs(env_name, num_envs=8, record_interval=10, seed=0):
    envs = [make_env(env_name, i + seed, i == 0, record_interval, f"../videos") for i in range(num_envs)]
    train_envs = SubprocVectorEnv(envs)
    test_envs = SubprocVectorEnv([make_env(env_name, i + num_envs + seed) for i in range(num_envs)])
    return train_envs, test_envs


def setup_network_and_policy(env, learning_rate=1e-3, hidden_sizes=[128, 128]):
    state_shape = env.observation_space[0].shape or env.observation_space[0].n
    action_shape = env.action_space[0].shape or env.action_space[0].n

    net = Net(state_shape=state_shape, action_shape=action_shape, hidden_sizes=hidden_sizes, device='cuda').to('cuda')
    optim = torch.optim.Adam(net.parameters(), lr=learning_rate)
    policy = DQNPolicy(net, optim, estimation_step=3, target_update_freq=10)

    return policy


def main(cfg):
    train_envs, test_envs = get_envs(cfg.env_name, cfg.num_envs)

    policy = setup_network_and_policy(train_envs, cfg.learning_rate, cfg.hidden_sizes)

    # Setup the log directory for TensorBoard
    log_dir = "../logs"
    os.makedirs(log_dir, exist_ok=True)

    # Initialize the SummaryWriter for TensorBoard
    summary_writer = SummaryWriter(log_dir)
    summary_writer.add_text("args", str(args))

    train_collector = Collector(policy, train_envs, VectorReplayBuffer(total_size=50000, buffer_num=cfg.num_envs))
    test_collector = Collector(policy, test_envs)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"DQN_{cfg.env_name}_{timestamp}"

    # Initialize WandbLogger
    wandb_logger = WandbLogger(
    # wandb.init(
        project='FreshProject',
        train_interval=1,
        update_interval=1,
        name=run_name,  # Optional: if None, W&B will assign a random name
        run_id=run_name,  # Optional: if None, W&B will assign a random name
        # id=run_name,  # Optional: if None, W&B will assign a random name
        entity='tu-e',  # Optional: specify if you're part of a team/organization
        # group=cfg.env_name,  # Optional: group runs together
        config=cfg,  # Pass argparse.Namespace or dict
        monitor_gym=True,  # Automatically logs video if your env supports it
        # sync_tensorboard=True,  # Automatically upload tensorboard logs
    )

    # Load the SummaryWriter into WandbLogger
    wandb_logger.load(summary_writer)

    def train_callback(epoch, env_step):
        policy.set_eps(max(0.1, 1 - env_step / 25000))
        if epoch % cfg.record_interval == 0:
        # Here you would save and upload the video if it was recorded
        # Assuming videos are saved in './videos', you can upload them using WandB
            for filename in os.listdir("./videos"):
                if filename.endswith(".mp4"):
                    wandb.log({"video": wandb.Video(os.path.join("./videos", filename), caption=f"Epoch_{epoch}")})

    result = offpolicy_trainer(
        policy, train_collector, test_collector,
        max_epoch=cfg.max_epoch, step_per_epoch=cfg.step_per_epoch,
        step_per_collect=cfg.step_per_collect, episode_per_test=cfg.episode_per_test, batch_size=cfg.batch_size,
        train_fn=train_callback,
        test_fn=lambda epoch, env_step: policy.set_eps(0.05),
        logger=wandb_logger,
    )

    print(f'Finished training! Results: {result}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_envs', type=int, default=8, help='Number of parallel environments')
    parser.add_argument('--env_name', type=str, default='CartPole-v1', help='Environment name')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the optimizer')
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[128, 128, 128],
                        help='Sizes of the hidden layers in the network')
    parser.add_argument('--max_epoch', type=int, default=10, help='Maximum number of epochs')
    parser.add_argument('--step_per_epoch', type=int, default=1000, help='Steps per epoch')
    parser.add_argument('--step_per_collect', type=int, default=10, help='Steps per data collection')
    parser.add_argument('--episode_per_test', type=int, default=100, help='Episodes per test phase')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--record_interval', type=int, default=20, help='Interval of epochs to record videos')

    args = parser.parse_args()
    main(args)
