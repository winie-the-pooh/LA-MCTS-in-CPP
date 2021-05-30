from functions.functions import *
from functions.mujoco_functions import *
from lamcts import MCTS

Dims = [1,2,3] + list(range(5,41,5))

for i in Dims:

    f = Ackley(dims = i)


    agent = MCTS(
                 lb = f.lb,              # the lower bound of each problem dimensions
                 ub = f.ub,              # the upper bound of each problem dimensions
                 dims = f.dims,          # the problem dimensions
                 ninits = f.ninits,      # the number of random samples used in initializations 
                 func = f,               # function object to be optimized
                 Cp = f.Cp,              # Cp for MCTS
                 leaf_size = f.leaf_size, # tree leaf size
                 kernel_type = f.kernel_type, #SVM configruation
                 gamma_type = f.gamma_type    #SVM configruation
                 )

    agent.search(iterations = 500)