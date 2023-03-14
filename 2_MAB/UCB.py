import numpy as np
from mab import Solver
from bandit import Bandit
from util import plot_results


# select the action with the highest upper confidence bound, 
# the upper confidence bound is defined as:
# UCB = estimation + sqrt(log(t)/(2*(counts+1)))  

class UCB(Solver):
    def __init__(self, bandit, coef,init_prob = 1.):
        super().__init__(bandit)
        self.total_count = 0
        self.estimates = np.array([init_prob] * self.bandit.M)
        self.coef = coef

    def run_one_step(self):
        self.total_count += 1
        ucb = self.estimates + self.coef * np.sqrt(np.log(self.total_count) / 2 / (self.counts + 1))
        k = np.argmax(ucb)
        r = self.bandit.step(k)
        self.estimates[k] += (1. / (self.counts[k] + 1) * (r - self.estimates[k]))

        return k

def run_ucb():
    np.random.seed(1)
    bandit = Bandit(10)
    coef = 1
    ucb_solver = UCB(bandit,coef=coef)
    ucb_solver.run(5000)
    # print("the total regrets: ", ucb_solver.regret_list)
    plot_results([ucb_solver],["UCB"])

run_ucb()