import numpy as np
    
# we then create a class for the agent to solve this problem

class Solver:
    def __init__(self,bandit):
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.M)
        self.regret = 0
        self.regret_list = []
        self.action_list = []
    
    def update_regret(self,m):
        self.regret += self.bandit.best_prob - self.bandit.prob[m]
        self.regret_list.append(self.regret)
    
    def run_one_step(self):
        raise NotImplemented

    def run(self,num_steps):
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.action_list.append(k)
            self.update_regret(k)
