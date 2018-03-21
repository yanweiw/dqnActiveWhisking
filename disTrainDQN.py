# Felix Yanwei Wang @ Northwestern University MSR, Feb 2018
# Acknowledgement: code borrowed in part from https://keon.io/deep-q-learning/#Cartpole-Game

from keras.models import load_model
from collections import deque
import disDqnEnv as dde
import disDqnAgent as dda
import numpy as np
import ray

# supress h5py warning
# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
# import h5py
# warnings.resetwarnings()

# @ray.remote
def updatedqn(agent, batch_size, memory):
    minibatch = random.sample(memory, batch_size)
    for state, action, reward, next_state, terminal in minibatch:
        target = reward # for terminal state
        if not terminal:
            target = reward + agent.gamma * np.amax(agent.model.predict(next_state)[0])
        target_f = agent.model.predict(state)
        target_f[0][action] = target
        agent.model.fit(state, target_f, epochs=1, verbose=0)
    if agent.epsilon > agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay
    return agent

def train():
    # episodes = 1000
    # N_episodes = 50
    # max_exploration = 20

    # initialize environment and the agent
    # env = de.dqnEnv('models/dnn_tri_hex.h5')
    agent = dda.dqnAgent(95, 11)
    memory = deque(maxlen=3000)
    classify = load_model('models/dnn_tri_hex.h5')

    # for e in range(1, 101):  # the first entry in N_episodes
        # if e % N_episodes == 1:
        #     ob = np.zeros((50, max_exploration, 8))
        #     rw = np.zeros((50, max_exploration, 1))
        #     ls = np.zeros((50, max_exploration, 1))
        #     score = 0.0
        #     steps = 0.0

    # envs = [dde.dqnEnv.remote() for _ in range(50)]
    # reset state to start with
    # env.reset.remote() for env in envs
    envs = [dde.dqnEnv.remote(classify) for _ in range(50)]
    # start 50 tasks
    remaining_ids = [env.explore.remote(agent) for env in envs]
    for _ in range(950):
        ready_ids, remaining_ids = ray.wait(remaining_ids)
        memory.extend(ray.get(ready_ids))
        if len(agent.memory) > 1000:
            # remaining_ids.append(updatedqn.remote())
            agent = updatedqn(agent, 32, memory)
        new_env = dde.dqnEnv.remote()
        remaining_ids.append(new_env.explore.remote(agent))


    #     # begin time sequence
    #     for t in range(1, max_exploration+1):
    #         curr_state = np.asarray(env.state).reshape(1, 95) # convert deque to nparray
    #         action = agent.act(curr_state)
    #         reward, terminal, loss = env.step(action)
    #         ob[(e - 1) % 50, t - 1, :] = env.shape, env.shapeX, env.shapeY, env.shapeT, env.shapeS, env.agentX, env.agentY, env.agentZ
    #         rw[(e - 1) % 50, t - 1] = reward
    #         ls[(e - 1) % 50, t - 1] = loss
    #         next_state = np.asarray(env.state).reshape(1, 95) # convert deque to nparray
    #         if t == max_exploration:
    #             terminal = True
    #         # remember the transition
    #         agent.remember(curr_state, action, reward, next_state, terminal)
    #         # total Qvalue as score
    #         if terminal:
    #             score += env.qValue
    #             steps += t  # this is why t starts at 1 rather than 0
    #             if e % N_episodes == 0:
    #                 print('episodes: {}/{}, score: {}, steps: {}'.format(e, episodes, \
    #                                                                 score / N_episodes, steps / N_episodes))
    #                 np.save('data/agent_observation', ob)
    #                 np.save('data/agent_reward', rw)
    #                 np.save('data/agent_loss', ls)
    #             break
    #
    #     # train dqn with gradient descent on past experience
    #     if len(agent.memory) > 1000:
    #         agent.replay(32)
    #
    # print('Saving trained DQN model...')
    # agent.model.save('models/dqn_episodes_%d.h5' % episodes)






# starts Ray
ray.init()
train()
