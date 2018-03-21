import ray
import disDqnAgent as dda
import disDqnEnv as dde

ray.init()
agent = dda.dqnAgent(95, 11)
weights = agent.model.get_weights()

c1 = dde.dqnEnv.remote()
c1.reset.remote(weights)
results = c1.explore.remote(weights)

@ray.remote
class Counter(object):
    def __init__(self):
        self.value = 0

    def increment(self):
        self.value += 1
        return self.value
