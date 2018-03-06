# Felix Yanwei Wang @ Northwestern University MSR, Feb 2018

from collections import deque
from keras.models import load_model
from keras import backend as K
import numpy as np
import simulation as sim


class dqnEnv:
    def __init__(self, rnnpath='models/lstm_tri_hex.h5'):
        self.max_width = sim.max_width     # x ~ (0, 20)
        self.min_width = sim.min_width
        # self.height = 20    # y ~ (0, 20)
        self.max_depth = sim.max_depth     # z ~ (0, 20)
        self.min_depth = sim.min_depth
        self.max_shape_pos = sim.max_x
        self.min_shape_pos = sim.min_x
        self.max_s = sim.max_s
        self.min_s = sim.min_s
        self.agentX = 10#np.random.randint(self.min_width, self.max_width, dtype=np.uint8)
        self.agentY = 10#np.random.randint(self.min_width, self.max_width, dtype=np.uint8)
        self.agentZ = 8#np.random.randint(self.min_depth, self.max_depth, dtype=np.uint8)
        self.shape = np.random.choice([0, 1])
        self.shapeX = np.random.randint(self.min_shape_pos, self.max_shape_pos, dtype=np.uint8)
        self.shapeY = np.random.randint(self.min_shape_pos, self.max_shape_pos, dtype=np.uint8)
        self.shapeT = np.random.uniform(0, 2 * np.pi)
        self.shapeS = np.random.uniform(self.min_s, self.max_s)
        self.qValue = 0 # initial prediction of accumulated rewards, as future rewards are negative
        self.rnn = load_model(rnnpath)
        self.rnn.reset_states()
        self.state = deque(maxlen=5) # state is a deque of past five observations (numpy array of 1, 19)
        for i in range(5):
            self.state.append(np.zeros((1, 19)))


    def reset(self):
        '''
        Function to restart another sequence of simulations
        Return the first state produced by first observation after reset the stateful rnn
        '''
        self.agentX = 10#np.random.randint(self.min_width, self.max_width, dtype=np.uint8)
        self.agentY = 10#np.random.randint(self.min_width, self.max_width, dtype=np.uint8)
        self.agentZ = 8#np.random.randint(self.min_depth, self.max_depth, dtype=np.uint8)
        self.shape = np.random.choice([0, 1])
        self.shapeX = np.random.randint(self.min_shape_pos, self.max_shape_pos, dtype=np.uint8)
        self.shapeY = np.random.randint(self.min_shape_pos, self.max_shape_pos, dtype=np.uint8)
        self.shapeT = np.random.uniform(0, 2 * np.pi)
        self.shapeS = np.random.uniform(self.min_s, self.max_s)
        self.rnn.reset_states()
        for i in range(5):
            self.state.append(np.zeros((1, 19)))
        # self.qValue = 20
        # Fist observation in the sequence corresponding to a "stay action"
        self.qValue = 0 # maybe inital qValue should be renewed after inital step # Equality at birth
        return self.state


    def step(self, action):
        '''
        Function to step forward in time
        If an action tries to move agent out of bounds, make the action a "stay" action
        Return reward and state
        '''
        action = self.updateAgentPos(action) # modify action if agent doesn't move due to bounds
        config = [self.shape, self.shapeX, self.shapeY, self.shapeT, self.shapeS, \
                                    self.agentX, self.agentY, self.agentZ]
        observation = sim.getDist(config) # (1, 19)
        label = np.zeros((1,1))
        if not self.shape:
            label = np.ones((1,1))
        # evaluate loss as reward
        loss = self.rnn.evaluate(observation.reshape(1, 1, 19), label, batch_size=1, verbose=0)[0]
        # new_state = K.get_value(self.rnn.layers[0].states[1]) # (1, 100) np array
        self.state.append(observation)
        # update reward and state
        # self.state = new_state
        terminal = False
        reward = - loss
        # if action == 0:
            # reward = -1 # discourage stay action
        if loss < 0.50:
            terminal = True
            reward = 1.0
        self.qValue += reward
        return reward, self.state, terminal


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
        else:
            action = 0 # corresponding to action == 0
        return action
