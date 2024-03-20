import argparse

from tianshou.highlevel.config import SamplingConfig
from tianshou.highlevel.env import (
    EnvFactoryRegistered,
    VectorEnvType,
)
from tianshou.highlevel.experiment import DQNExperimentBuilder, ExperimentConfig
from tianshou.highlevel.logger import LoggerFactoryDefault
from tianshou.highlevel.params.policy_params import DQNParams
from tianshou.highlevel.trainer import (
    EpochStopCallbackRewardThreshold,
    EpochTestCallbackDQNSetEps,
    EpochTrainCallbackDQNSetEps,
)
from tianshou.utils.logging import run_main

import wandb


def main() -> None:
    args = parse_args()
    experiment = (
        DQNExperimentBuilder(
            EnvFactoryRegistered(task="CartPole-v1", seed=0, venv_type=VectorEnvType.DUMMY),
            ExperimentConfig(
                persistence_enabled=True,
                watch=False,
                watch_render=1 / 35,
                watch_num_episodes=1,
            ),
            SamplingConfig(
                num_epochs=100,
                step_per_epoch=10000,
                batch_size=64,
                num_train_envs=10,
                num_test_envs=100,
                buffer_size=20000,
                step_per_collect=10,
                update_per_step=1 / 10,
            ),
        )
        .with_dqn_params(
            DQNParams(
                lr=1e-3,
                discount_factor=0.9,
                estimation_step=3,
                target_update_freq=320,
            ),
        )
        .with_model_factory_default(hidden_sizes=(64, 64))
        .with_epoch_train_callback(EpochTrainCallbackDQNSetEps(0.3))
        .with_epoch_test_callback(EpochTestCallbackDQNSetEps(0.0))
        # .with_epoch_stop_callback(EpochStopCallbackRewardThreshold(500))
        .with_logger_factory(LoggerFactoryDefault(logger_type="wandb", wandb_project="FreshProject"))
        .build()
    )
    experiment.run()
    wandb.join()


def parse_args():
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
    return parser.parse_args()


if __name__ == "__main__":
    run_main(main)
