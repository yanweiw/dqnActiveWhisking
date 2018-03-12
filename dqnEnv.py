# Felix Yanwei Wang @ Northwestern University MSR, Feb 2018

from collections import deque
from keras.models import load_model
from keras import backend as K
import numpy as np
import simulation as sim


class dqnEnv:
    def __init__(self, dnnpath='models/dnn_tri_hex.h5'):
        self.min_width = sim.min_width     # head x ~ (0, 21)
        self.max_width = sim.max_width
        self.min_depth = sim.min_depth     # z ~ (1, 11)
        self.max_depth = sim.max_depth
        self.min_shape_pos = sim.min_x
        self.max_shape_pos = sim.max_x
        self.min_s = sim.min_s
        self.max_s = sim.max_s
        # initialize at center top
        self.agentX = 10#np.random.randint(self.min_width, self.max_width, dtype=np.uint8)
        self.agentY = 10#np.random.randint(self.min_width, self.max_width, dtype=np.uint8)
        self.agentZ = 8#np.random.randint(self.min_depth, self.max_depth, dtype=np.uint8)
        self.shape = np.random.choice([0, 1]) # whether tri or hex
        self.shapeX = np.random.uniform(self.min_shape_pos, self.max_shape_pos)
        self.shapeY = np.random.uniform(self.min_shape_pos, self.max_shape_pos)
        self.shapeT = np.random.uniform(0, 2 * np.pi)
        self.shapeS = np.random.uniform(self.min_s, self.max_s)
        self.qValue = 0 # initial prediction of accumulated rewards, as future rewards are negative
        self.dnn = load_model(dnnpath)
        # self.rnn.reset_states()   # dnn needs not to be reset
        self.state = deque(maxlen=5) # state is a deque of past five observations (numpy array of 1, 19)
        for i in range(5):
            self.state.append(np.zeros((1, 19)))


    def reset(self):
        '''
        Function to restart another sequence of simulations
        Return the first state produced by first observation after reset the stateful dnn
        '''
        self.agentX = 10#np.random.randint(self.min_width, self.max_width, dtype=np.uint8)
        self.agentY = 10#np.random.randint(self.min_width, self.max_width, dtype=np.uint8)
        self.agentZ = 8#np.random.randint(self.min_depth, self.max_depth, dtype=np.uint8)
        self.shape = np.random.choice([0, 1])
        self.shapeX = np.random.uniform(self.min_shape_pos, self.max_shape_pos)
        self.shapeY = np.random.uniform(self.min_shape_pos, self.max_shape_pos)
        self.shapeT = np.random.uniform(0, 2 * np.pi)
        self.shapeS = np.random.uniform(self.min_s, self.max_s)
        # self.rnn.reset_states()
        for i in range(5):
            self.state.append(np.zeros((1, 19)))
        # Fist observation in the sequence corresponding to a "stay action"
        self.step(0)
        self.qValue = 0 # maybe inital qValue should be renewed after inital step # Equality at birth


    def step(self, action):
        '''
        Function to step forward in time
        If an action tries to move agent out of bounds, make the action a "stay" action
        Return reward and state
        '''
        action = self.updateAgentPos(action) # modify action if agent doesn't move due to bounds
        config = [self.shape, self.shapeX, self.shapeY, self.shapeT, self.shapeS, \
                                    self.agentX, self.agentY, self.agentZ]
        observation, categories = sim.getDist(config) # (1, 19) np.array observation
        label = np.ones((1,1))
        if not self.shape:
            label = np.zeros((1,1))
        # evaluate loss as reward
        # loss = self.rnn.evaluate(observation.reshape(1, 19), label, batch_size=1, verbose=0)[0]
        loss = self.dnn.evaluate(observation, label, verbose=0)[0]
        # new_state = K.get_value(self.dnn.layers[0].states[1]) # (1, 100) np array
        # update reward and state
        self.state.append(observation)
        # self.state = new_state

        cate_prob = (categories + 0.01) / 19.04 # avoid 0s and thus NaN
        # calc Shannon entropy
        entropy = -np.sum(cate_prob * np.log2(cate_prob))
        # use agentZ as weight
        weight = (11 - self.agentZ) / 5.0

        terminal = False
        reward = weight * entropy - 1
        # if action == 0:
            # reward = -1 # discourage stay action
        # if np.all((observation==0), axis=1)[0] or np.all((observation==255), axis=1)[0]:
            # reward -= 0.5
        # total_ob = np.sum(observation)
        # if total_ob < 510 or total_ob > 4590:  # fewer than two whiskers (255 * 2) on shape or (255 * 18) off shape
            # reward -= 0.5                         # assuming loss for these cases are larger than 0.1
        if loss < 0.1:
            terminal = True
            reward = 30
        # if loss < 0.01:
            # terminal = True
            # reward = 10.0
        self.qValue += reward
        return reward, terminal, loss


    def updateAgentPos(self, action):
        '''
        Update agent position, subject to boundary checking
        action numerical value mapping:
        0: stay put
        1: agentX++
        2: agentY++
        3: agentX--
        4: agentY--
        5: agentZ++
        6: agentZ--
        7: agentZ--, agentX++
        8: agentZ--, agentY++
        9: agentZ--, agentX--
        10:agentZ--, agentY--
        '''
        if (action == 1) and (self.agentX + 1 < self.max_width):
            self.agentX += 1
        elif (action == 2) and (self.agentY + 1 < self.max_width):
            self.agentY += 1
        elif (action == 3) and (self.agentX - 1 >= self.min_width):
            self.agentX -= 1
        elif (action == 4) and (self.agentY - 1 >= self.min_width):
            self.agentY -= 1
        elif (action == 5) and (self.agentZ + 1 < self.max_depth):
            self.agentZ += 1
        elif (action == 6) and (self.agentZ - 1 >= self.min_depth):
            self.agentZ -= 1
        elif (action == 7) and (self.agentZ - 1 >= self.min_depth) and (self.agentX + 1 < self.max_width):
            self.agentZ -= 1
            self.agentX += 1
        elif (action == 8) and (self.agentZ - 1 >= self.min_depth) and (self.agentY + 1 < self.max_width):
            self.agentZ -= 1
            self.agentY += 1
        elif (action == 9) and (self.agentZ - 1 >= self.min_depth) and (self.agentX - 1 >= self.min_width):
            self.agentZ -= 1
            self.agentX -= 1
        elif (action == 10) and (self.agentZ - 1 >= self.min_depth) and (self.agentY - 1 >= self.min_width):
            self.agentZ -= 1
            self.agentY -= 1
        else:
            action = 0 # corresponding to action == 0
        return action
