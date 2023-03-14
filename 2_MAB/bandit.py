import numpy as np
import matplotlib.pyplot as plt


# We create a bandit class to simulate the bandit problem
class Bandit:
    def __init__(self, m):
        self.prob = np.random.uniform(size=m)
        self.best_idx = np.argmax(self.prob)
        self.best_prob = self.prob[self.best_idx]
        self.M = m
    
    def step(self,k):
        if np.random.rand() < self.prob[k]:
            return 1
        else:
            return 0