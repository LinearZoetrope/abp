import time

import gym
import numpy as np
from tensorboardX import SummaryWriter
from abp import DQNAdaptive
from abp.utils import clear_summary_path


def run_task(evaluation_config, network_config, reinforce_config, log=True):
    env = gym.make(evaluation_config.env)
    state = env.reset()
    max_episode_steps = env._max_episode_steps

    agent = DQNAdaptive(name="TicTacToe",
                        choices=range(9),
                        network_config=network_config,
                        reinforce_config=reinforce_config)
    if log:
        training_summaries_path = evaluation_config.summaries_path + "/train"
        clear_summary_path(training_summaries_path)
        train_summary_writer = SummaryWriter(training_summaries_path)

        test_summaries_path = evaluation_config.summaries_path + "/test"
        clear_summary_path(test_summaries_path)
        test_summary_writer = SummaryWriter(test_summaries_path)

    for episode in range(evaluation_config.training_episodes):
        state = env.reset()
        total_reward = 0

        for steps in range(max_episode_steps):
            action, _ = agent.predict(state)

            state, reward, done, info = env.step(action)

            total_reward += reward

            reshaped_board = np.reshape(info['board'], (3, 3))

            sum_rows = np.sum(reshaped_board, axis=1)
            sum_cols = np.sum(reshaped_board, axis=0)
            sum_diagonal = np.trace(reshaped_board)
            sum_rev_diagonal = np.trace(np.flipud(reshaped_board))

            reward_type = 0

            for row in range(3):
                if sum_rows[row] == 3:
                    agent.reward(10)
                elif sum_rows[row] == -3:
                    agent.reward(-10)

            reward_type = 3

            for col in range(3):
                if sum_cols[col] == 3:
                    agent.reward(10)
                elif sum_cols[col] == -3:
                    agent.reward(-10)

            if sum_diagonal == 3:
                agent.reward(10)
            elif sum_diagonal == -3:
                agent.reward(-10)

            if sum_rev_diagonal == 3:
                agent.reward(10)
            elif sum_diagonal == -3:
                agent.reward(-10)

            if info['illegal_move']:
                agent.reward(-10)
            else:
                agent.reward(1)

            if done:
                agent.end_episode(state)
                if log:
                    train_summary_writer.add_scalar(tag="Episode Reward", scalar_value=total_reward,
                                                    global_step=episode + 1)
                    print('Episode Reward:', total_reward)
                break

    agent.disable_learning()

    # Test Episodes
    for episode in range(evaluation_config.test_episodes):
        state = env.reset()
        total_reward = 0
        for steps in range(max_episode_steps):
            action, q_values = agent.predict(state)

            if evaluation_config.render:
                env.render()
                time.sleep(10)

            state, reward, done, info = env.step(action)
            total_reward += reward

            if done:
                if log:
                    test_summary_writer.add_scalar(tag="Episode Reward", scalar_value=total_reward,
                                                   global_step=episode + 1)
                    print('Episode_Reward:', total_reward)
                break

    env.close()
