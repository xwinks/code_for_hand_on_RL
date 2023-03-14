import numpy as np
import matplotlib.pyplot as plt

# plot the regrets with time
def plot_results(solvers,solver_names):
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regret_list))
        plt.plot(time_list,solver.regret_list,label=solver_names[idx])
    
    plt.xlabel("time")
    plt.ylabel("regret")
    plt.title("Regret with time")
    plt.legend()
    plt.show()
