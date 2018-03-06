# Felix Yanwei Wang @ Northwestern University MSR, Feb 2018
# Acknowledgement: code borrows heavily from https://keon.io/deep-q-learning/#Cartpole-Game

from keras.models import load_model
# from dqnEnv import dqnEnv
import dqnEnv as de
import dqnAgent as da
import numpy as np

episodes = 20000
N_episodes = 50
max_exploration = 20

# initialize environment and the agent
env = de.dqnEnv('models/lstm_tri_hex.h5')
agent = da.dqnAgent(95, 7)

# Iterate episods
ob = np.zeros((50, max_exploration, 8))
rw = np.zeros((50, max_exploration, 1))
for e in range(1, episodes+1):
    if e % N_episodes == 1:
        ob = np.zeros((50, max_exploration, 8))
        rw = np.zeros((50, max_exploration, 1))
    # reset state to start with
    curr_state = env.reset()
    curr_state = np.asarray(curr_state).reshape(1, 95) # convert deque to nparray
    if e % N_episodes == 1:   # the first entry in N_episodes
        score = 0
        steps = 0.0
    # begin time sequence
    for t in range(1, max_exploration+1):
        action = agent.act(curr_state)
        reward, next_state, terminal = env.step(action)
        ob[(e - 1) % 50, t - 1, :] = env.shape, env.shapeX, env.shapeY, env.shapeT, env.shapeS, env.agentX, env.agentY, env.agentZ
        rw[(e - 1) % 50, t - 1] = reward
        next_state = np.asarray(next_state).reshape(1, 95) # convert deque to nparray
        # terminal = False
        if t == max_exploration:
            terminal = True
        # remember the transition
        agent.remember(curr_state, action, reward, next_state, terminal)

        # advance state, total Qvalue as score
        curr_state = next_state

        if terminal:
            score += env.qValue
            steps += t
            if e % N_episodes == 0:
                print('episodes: {}/{}, score: {}, steps: {}'.format(e, episodes, \
                                                                score / N_episodes, steps / N_episodes))
                np.save('data/agent_observation', ob)
                np.save('data/agent_reward', rw)
            break

    # train dqn with gradient descent on past experience
    if len(agent.memory) > 1000:
        agent.replay(32)

print('Saving trained DQN model...')
agent.model.save('models/dqn_episodes_%d.h5' % episodes)
