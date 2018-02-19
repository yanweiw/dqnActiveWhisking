# Felix Yanwei Wang @ Northwestern University MSR, Feb 2018
# Acknowledgement: code borrows heavily from https://keon.io/deep-q-learning/#Cartpole-Game

from keras.models import load_model

episodes = 10
time_span = 10

# initialize environment and the agent
env = dqnEnv('models/lstm_tri_hex.h5')
agent = dqnAgent(100, 7)

# Iterate episods
for e in range(episodes):
    # reset state to start with
    curr_state = env.reset()

    # begin time sequence
    for t in range(time_span):
        action = agent.act(curr_state)
        reward, next_state = env.step(action)
        terminal = False
        if t >= time_span:
            terminal = True
        # remember the transition
        agent.remember(curr_state, action, reward, next_state, terminal)

        # advance state, total Qvalue as score
        state = next_state
        score = env.Qvalue

        if terminal:
            print('episodes: {}/{}, score: {}'.format(e, episodes, score))
            break

    # train dqn with gradient descent on past experience
    agent.replay(32)

print('Saving trained DQN model...')
model.save('models/dqn_episodes_%d.h5' % episodes)
