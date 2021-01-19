import math
import numpy as np
from scipy.stats import t
from scipy.special import softmax


# The following two functions were used from Professor's AI Safety Tutorial.
# Link to the tutorial: https://aisafety.cs.umass.edu/tutorial4py.html
########################################
'''
This function returns the inverse of Student's t CDF using the degrees of freedom in nu for the corresponding
probabilities in p. It is a Python implementation of Matlab's tinv function: https://www.mathworks.com/help/stats/tinv.html
'''


def tinv(p, nu):
    return t.ppf(p, nu)


'''
This function computes the sample standard deviation of the vector v, with Bessel's correction
'''


def stddev(v):
    n = v.size
    variance = (np.var(v) * n) / (n-1)  # Variance with Bessel's correction
    return np.sqrt(variance)           # Compute the standard deviation
##########################################


'''
PDIS takes in data and theta to calculate PDIS(H, pi_e, pi_b) as in Eq 278 of notes, 
PDIS(D, pi_e, pi_b) as in Eq 279 of notes. 

pdis_avg represents the PDIS(D, pi_e, pi_b)
pdis represents the PDIS(H, pi_e, pi_b)
'''
# Eq 278 & 279 implementation


def PDIS(data_in, theta):
    # Get list of all episode sizes
    episodes = [int(value.shape[1]) for value in data_in.values()]
    # Genearate gamma to the power t array for the max episode size
    gamma = np.full(max(episodes), 0.95)**np.arange(0, max(episodes))
    # gamma = np.cumprod(np.ones(max(horizons))*0.95)/0.95

    # Keys of size number epsiodes
    keys = list(data_in.keys())
    # pdis_arr of size number of episodes. Hold Eq 278 output for each episode
    pdis_arr = np.empty(len(data_in.keys()))
    # Softmax to get policy values from the given theta
    pi_theta = softmax(theta, axis=1)

    # For each episode. each key has a 4x18 numy 2d array with index 0,1,2,3 being states, actions, rewards, pi_b
    for i in range(len(episodes)):
        # Get all the states for the episode
        pi_e_state = data_in[keys[i]][0, :].astype(int)
        # Get all the actions for the episode
        pi_e_action = data_in[keys[i]][1, :].astype(int)
        # Had issues with np.cumprod as vales were going to nan. so tried to do the procduct using log.
        pi_e = np.log(pi_theta[pi_e_state, pi_e_action])
        # Do the same for pi_b
        pi_b = np.log(data_in[keys[i]][3, :])
        # Subtract the logs and use exponential to get the equivalent of cumprod
        pi_e_by_b = np.exp(np.cumsum(pi_e-pi_b))
        # Get rewards for the episode
        R_t = data_in[keys[i]][2, :]
        # Get the product
        pdis_arr[i] = np.sum(gamma[:episodes[i]]*pi_e_by_b*R_t)

    # Gives PDIS(D, pi_e, pi_b)
    pdis_avg = np.sum(pdis_arr)/len(pdis_arr)

    return pdis_avg, pdis_arr


'''
policy_constraint takes in size of safety set, PDIS(D, pi_e, pi_b), PDIS(H, pi_e, pi_b) for episodes, mult.
Mult is used as a multipler as 2 is used for constraint, and 1 for safety. 
'''
# Eq 294 implementation


def policy_constraint(size_Ds, pdis_avg, pdis_arr, mult):
    # 1-delta is the confidence. 95% here
    delta = 0.05
    std_by_Ds = stddev(pdis_arr)/np.sqrt(size_Ds)
    t = tinv(1-delta, size_Ds-1)
    # condition holds the value representing Eq 294
    condition = pdis_avg - (mult * std_by_Ds * t)
    return condition


'''
policy_objective takes in multiple theta values passed through CMAEvolutionStrategy of the pycma library. Sends 5
Returns array of -PDIS(D, pi_e, pi_b) value for each theta if policy_constraint with mult 2 is greater the 1.41537 else return infinity.
'''


def policy_objective(thetas, size_Ds, Dc):
    c = 1.41537
    output = []
    for theta in thetas:
        theta = theta.reshape((18, 4))
        pdis_avg_c, pdis_arr_c = PDIS(Dc, theta)
        if policy_constraint(size_Ds, pdis_avg_c, pdis_arr_c, 2) >= c:
            output.append(-pdis_avg_c)
        else:
            output.append(float('inf'))
    return output


'''

'''
# Eq 285 implementation


def policy_safety_test(theta, Ds):
    c = 1.41537
    delta = 0.05
    theta = theta.reshape((18, 4))
    size_Ds = len(Ds)
    pdis_avg_s, pdis_arr_s = PDIS(Ds, theta)
    # Here std_by_Ds is not times 2 as in policy constraint.
    condition_safe = policy_constraint(size_Ds, pdis_avg_s, pdis_arr_s, 1)
    if condition_safe >= c:
        return True
    return False
