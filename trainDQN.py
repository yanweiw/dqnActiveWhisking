# Felix Yanwei Wang @ Northwestern University MSR, Feb 2018
# Acknowledgement: code borrows heavily from https://keon.io/deep-q-learning/#Cartpole-Game

from keras.models import load_model
# from dqnEnv import dqnEnv
import dqnEnv as de
import dqnAgent as da
import numpy as np

episodes = 5000
N_episodes = 50
max_exploration = 20

# initialize environment and the agent
env = de.dqnEnv('models/dnn_tri_hex.h5')
agent = da.dqnAgent(95, 11)

# Iterate episods
# ob = np.zeros((50, max_exploration, 8))
# rw = np.zeros((50, max_exploration, 1))
for e in range(1, episodes+1):  # the first entry in N_episodes
    if e % N_episodes == 1:
        ob = np.zeros((50, max_exploration, 8))
        rw = np.zeros((50, max_exploration, 1))
        score = 0.0
        steps = 0.0
    # reset state to start with
    env.reset()
    # begin time sequence
    for t in range(1, max_exploration+1):
        curr_state = np.asarray(env.state).reshape(1, 95) # convert deque to nparray
        action = agent.act(curr_state)
        reward, terminal = env.step(action)
        ob[(e - 1) % 50, t - 1, :] = env.shape, env.shapeX, env.shapeY, env.shapeT, env.shapeS, env.agentX, env.agentY, env.agentZ
        rw[(e - 1) % 50, t - 1] = reward
        next_state = np.asarray(env.state).reshape(1, 95) # convert deque to nparray
        if t == max_exploration:
            terminal = True
        # remember the transition
        agent.remember(curr_state, action, reward, next_state, terminal)
        # total Qvalue as score
        if terminal:
            score += env.qValue
            steps += t  # this is why t starts at 1 rather than 0
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
