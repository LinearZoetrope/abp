from gym.envs.registration import registry, register, make, spec


register(
    id='TicTacToe-v0',
    entry_point='abp.envs.tic_tac_toe:TicTacToeEnv',
    max_episode_steps=1000, #TODO
    reward_threshold=40 #TODO
)