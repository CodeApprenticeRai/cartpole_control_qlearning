import gym


env = gym.make('CartPole-v1', render_mode='human')

episode_max_length = 1000

observation = env.reset()

terminated_at_counter = 0 

for t in range(episode_max_length):
    env.render()
    random_action = env.action_space.sample()
    observation, reward, terminate_episode_signal, _, step_info = env.step(random_action)

    print((observation, reward, terminate_episode_signal), sep=", ")
    # input()

    if not (terminate_episode_signal):
        terminated_at_counter += 1
    else:
        break

print("Episode was terminated after {} timesteps".format(terminated_at_counter))

env.close()