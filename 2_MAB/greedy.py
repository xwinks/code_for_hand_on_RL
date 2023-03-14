import numpy as np
from mab import Solver
from bandit import Bandit
from util import plot_results

class EpsilonGreedy(Solver):
    def __init__(self, bandit, epsilon=0.01,init_prob = 1.0):
        super().__init__(bandit)
        self.epsilon = epsilon
        self.estimation = np.array([init_prob] * self.bandit.M)
        
    def run_one_step(self):
        if np.random.random() < self.epsilon:
            k = np.random.randint(0,self.bandit.M)
        else:
            k = np.argmax(self.estimation)
        r = self.bandit.step(k)
        self.estimation[k] += 1.0/(self.counts[k]+1) * (r - self.estimation[k])
        return k


def run_greedy_method():
    # np.random.seed(1)
    M = 10
    bandit = Bandit(10)
    print("the best action is: ", bandit.best_idx, "the probability is: ", bandit.prob[bandit.best_idx])
    epsilons = [1e-4,0.01,0.1,0.25,0.5]
    eposilons_greedy_solver_names = ["epsilon={}".format(e) for e in epsilons]
    eposilons_greedy_solver_list = [EpsilonGreedy(bandit=bandit,epsilon=epsilon) for epsilon in epsilons]
    for solver in eposilons_greedy_solver_list:
        solver.run(5000)
    plot_results(eposilons_greedy_solver_list,eposilons_greedy_solver_names)

run_greedy_method()