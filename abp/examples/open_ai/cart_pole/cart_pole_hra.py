import gym
from gym import wrappers
from abp.adaptives.hra import HRAAdaptive


def run_task(job_dir, render = True, training_episode = 500, test_episodes = 100, decay_steps = 250, model_path = None, restore_model = False):
    directory = "gym/hra/cartpole"
    env_spec = gym.make("CartPole-v0")

    threshold_angle = 0.087266463
    threshold_x = 1.5

    max_episode_steps = env_spec._max_episode_steps

    state = env_spec.reset()

    no_of_rewards = 4

    agent = HRAAdaptive(env_spec.action_space.n,
                        len(state),
                        no_of_rewards,
                        "Cart Pole", 
                        job_dir = job_dir,
                        decay_steps = decay_steps,
                        model_path = model_path,
                        restore_model = restore_model)

    #Episodes
    for epoch in range(training_episode):
        state = env_spec.reset()
        for steps in range(max_episode_steps):
            action = agent.predict(state)
            state, reward, done, info = env_spec.step(action)
            cart_position, cart_velocity, pole_angle, pole_velocity = state
            agent.reward(0, reward) # Reward for every step

            # Reward for pole angle increase or decrease
            if  -threshold_angle < pole_angle < threshold_angle:
                agent.reward(1, 1)
            else:
                agent.reward(1, -1)

            if steps < max_episode_steps and done:
                agent.reward(2, -40) # Reward for terminal state

            if -threshold_x < cart_position < threshold_x:
                agent.reward(3, 1)
            else:
                agent.reward(3, -1)

            agent.actual_reward(reward)

            if done:
                agent.end_episode(state)
                break


    agent.disable_learning()

    # After learning Episodes
    for epoch in range(test_episodes):
        state = env_spec.reset()
        for t in range(max_episode_steps):
            if render:
                env_spec.render()
            action = agent.predict(state)
            state, reward, done, info = env_spec.step(action)
            agent.test_reward(reward)

            if done:
                agent.end_episode(state)
                break

    env_spec.close()