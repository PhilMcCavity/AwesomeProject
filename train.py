import argparse

import torch
from gymnasium import make
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv
from tianshou.policy import DQNPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils.net.common import Net


def make_env(task, seed=0):
    def _init():
        env = make(task)
        return env

    return _init


def get_envs(env_name, num_envs=8, seed=0):
    train_envs = SubprocVectorEnv([make_env(env_name, i + seed) for i in range(num_envs)])
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

    train_collector = Collector(policy, train_envs, VectorReplayBuffer(total_size=50000, buffer_num=cfg.num_envs))
    test_collector = Collector(policy, test_envs)

    result = offpolicy_trainer(
        policy, train_collector, test_collector,
        max_epoch=cfg.max_epoch, step_per_epoch=cfg.step_per_epoch,
        step_per_collect=cfg.step_per_collect, episode_per_test=cfg.episode_per_test, batch_size=cfg.batch_size,
        train_fn=lambda epoch, env_step: policy.set_eps(max(0.1, 1 - env_step / 25000)),
        test_fn=lambda epoch, env_step: policy.set_eps(0.05)
    )

    print(f'Finished training! Results: {result}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_envs', type=int, default=8, help='Number of parallel environments')
    parser.add_argument('--env_name', type=str, default='CartPole-v1', help='Environment name')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the optimizer')
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[128, 128],
                        help='Sizes of the hidden layers in the network')
    parser.add_argument('--max_epoch', type=int, default=30, help='Maximum number of epochs')
    parser.add_argument('--step_per_epoch', type=int, default=1000, help='Steps per epoch')
    parser.add_argument('--step_per_collect', type=int, default=10, help='Steps per data collection')
    parser.add_argument('--episode_per_test', type=int, default=100, help='Episodes per test phase')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')

    args = parser.parse_args()
    main(args)
