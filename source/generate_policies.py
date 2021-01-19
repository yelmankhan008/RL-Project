import os
import cma
import time
import numpy as np
from scipy.stats import t
from scipy.special import softmax
from helper import *

# Load data
data = np.load("data.npy", allow_pickle=True)

# data_C is dictionary of candidate dat with 700,000 episodes.
data_C = {}
for i in range(700000):
    data_C[i] = data[i]

# data_S is dictionary of candidate dat with 300,000 episodes.
data_S = {}
for i in range(300000):
    data_S[i] = data[700000 + i]

# Used to get random seed values
seed_vals = np.arange(100000, 1000000)

start_time = time.time()

###################### RANDOM SEED GENERATOR ########################
np.random.seed(29857)
# np.random.seed(0)
#####################################################################

num_policy = 0
while num_policy < 100:
    Dc = {}
    Ds = {}

    # Different sizes used of candidate and safety sets. The size of candidate and safety sets are kept same
    indices_size = np.array([20000, 50000, 10000, 30000, 40000])

    # Choose a size at random from indices_size.
    size = np.random.choice(indices_size, 1)

    # Select random keys from data_C,data_S of the to create Dc,Ds for optimization
    indices_c = np.random.randint(len(data_C), size=size)
    indices_s = np.random.randint(len(data_S), size=size)

    # Populate Dc, Ds dicts using random generated indices
    Dc = {x: data_C[indices_c[x]] for x in range(len(indices_c))}
    Ds = {x: data_S[indices_s[x]] for x in range(len(indices_s))}
    size_Ds = len(Ds)

    # Initial theta is random and is flattens as CMAEvolutionStrategy requites a 1d array
    theta_e = np.random.rand(18*4).flatten()

    # Get random seed ofr CMAEvolutionStrategy
    cma_seed = np.random.choice(seed_vals, 1)

    # using pycma optimizer with sigma = 0.5, popsiz is 5 that is will return 5 theta values
    es = cma.CMAEvolutionStrategy(
        theta_e, 0.5, {'popsize': 5, 'seed': cma_seed})

    print(len(Dc), size_Ds)
    iter_ = 0

    # A lot of theta would crash out in a few iteration so I check till 20 and only consider those ones
    while (not es.stop()) and iter_ < 20:
        solutions = es.ask()
        es.tell(solutions, policy_objective(solutions, size_Ds, Dc))
        es.logger.add()
        # es.disp()
        iter_ += 1

    # Only consider the case where all 20 iterations happened.
    if iter_ == 20:
        theta_e_opt = es.result[0]
        print(theta_e_opt)
        # print(es.result[1])
        print(policy_safety_test(theta_e_opt, Ds))

        # Only create a policy if it passes the test
        if policy_safety_test(theta_e_opt, Ds):
            num_policy += 1
            if not os.path.exists('Policies'):
                os.makedirs('Policies')
            filename = f'Policies/policy{num_policy}.txt'
            with open(filename, 'w') as f:
                for val in theta_e_opt:
                    f.write(f'{val}\n')

            print(
                f'\nPolicy {num_policy} done! Time so far: {time.time() - start_time} seconds\n')
