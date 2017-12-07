### MDP Value Iteration and Policy Iteratoin
# You might not need to use all parameters

import numpy as np
import gym
import time
from lake_envs import *

np.set_printoptions(precision=3)

def compute_value(P,V,state,action, gamma):
	value_EM = 0

	#immediate reward  R(s,a)
	value_EM += P[state][action][0][2]

	#discounted sum of future value:  sum of p(s'|s,a) * V(s')
	possible_results = P[state][action]
	for i in range(len(possible_results)):

	    (probability, nextstate, reward, terminal) = possible_results[i]
	    value_EM += gamma * probability * V[nextstate]      

	return value_EM


def value_iteration(P, nS, nA, gamma=0.9, max_iteration=20, tol=1e-3):
	"""
	Learn value function and policy by using value iteration method for a given
	gamma and environment.

	Parameters:
	----------
	P: dictionary
		It is from gym.core.Environment
		P[state][action] is tuples with (probability, nextstate, reward, terminal)
	nS: int
		number of states
	nA: int
		number of actions
	gamma: float
		Discount factor. Number in range [0, 1)
	max_iteration: int
		The maximum number of iterations to run before stopping. Feel free to change it.
	tol: float
		Determines when value function has converged.
	Returns:
	----------
	value function: np.ndarray
	policy: np.ndarray
	"""
	V = np.zeros(nS)
	policy = np.zeros(nS, dtype=int)
	############################
	# YOUR IMPLEMENTATION HERE #
	############################


	for k in range(max_iteration):
	    Value_Old = V.copy()

	    for i in range(nS):
		Value_CurState_Opt = -10
		for j in range(nA):
		    Value_CurState_New = compute_value(P, Value_Old, i, j, gamma)
		    if Value_CurState_New > Value_CurState_Opt:
		        Value_CurState_Opt = Value_CurState_New
		        policy[i] = j

         	V[i] = Value_CurState_Opt

	    #check tolerance
            var = np.linalg.norm(Value_Old - V)
            if var<tol:
		break


	return V, policy

def policy_evaluation(P, nS, nA, policy, gamma=0.9, max_iteration=100, tol=1e-3):
	"""Evaluate the value function from a given policy.
	Parameters
	----------
	P: dictionary
	    It is from gym.core.Environment
            P[state][action] is tuples with (probability, nextstate, reward, terminal)
	nS: int
	    number of states
	nA: int
	    number of actions
	gamma: float
	    Discount factor. Number in range [0, 1)
	policy: np.array
	    The policy to evaluate. Maps states to actions.
	max_iteration: int
	    The maximum number of iterations to run before stopping. Feel free to change it.
	tol: float
	    Determines when value function has converged.
	Returns
	-------
	value function: np.ndarray
	The value function from the given policy.
	"""
	############################
	# YOUR IMPLEMENTATION HERE #
	############################
	value_function = np.zeros(nS,dtype=float)
	new_value_function = value_function.copy()
	k = 0
	while k<=max_iteration or np.sum(np.sqrt(np.square(new_value_function-value_function)))>tol:
	    k += 1
	    value_function = new_value_function.copy()
	    for state in range(nS):
		
		action = policy[state]
		new_value_function[state] = compute_value(P,value_function,state,action, gamma)

	return new_value_function





 
def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
	"""Given the value function from policy improve the policy.

	Parameters
	----------
	P: dictionary
		It is from gym.core.Environment
		P[state][action] is tuples with (probability, nextstate, reward, terminal)
	nS: int
		number of states
	nA: int
		number of actions
	gamma: float
		Discount factor. Number in range [0, 1)
	value_from_policy: np.ndarray
		The value calculated from the policy
	policy: np.array
		The previous policy.

	Returns
	-------
	new policy: np.ndarray
		An array of integers. Each integer is the optimal action to take
		in that state according to the environment dynamics and the
		given value function.
	"""    
	############################
	# YOUR IMPLEMENTATION HERE #
	############################
        Q = np.zeros([nS,nA])
        for state in range(nS):
            for action in range(nA):
                
		Q[state][action] = compute_value(P,value_from_policy,state,action, gamma)
	    
        policy_New = np.argmax(Q, axis=1)
	

        return policy_New



def policy_iteration(P, nS, nA, gamma=0.9, max_iteration=20, tol=1e-3):
	"""Runs policy iteration.

	You should use the policy_evaluation and policy_improvement methods to
	implement this method.

	Parameters
	----------
	P: dictionary
		It is from gym.core.Environment
		P[state][action] is tuples with (probability, nextstate, reward, terminal)
	nS: int
		number of states
	nA: int
		number of actions
	gamma: float
		Discount factor. Number in range [0, 1)
	max_iteration: int
		The maximum number of iterations to run before stopping. Feel free to change it.
	tol: float
		Determines when value function has converged.
	Returns:
	----------
	value function: np.ndarray
	policy: np.ndarray
	"""
	V = np.zeros(nS)
	policy     = np.zeros(nS, dtype=int)
        policy_New = policy.copy()
	############################
	# YOUR IMPLEMENTATION HERE #
	############################

	
	k = 0
	while k<=max_iteration or np.sum(np.sqrt(np.square(policy_New-policy)))>tol:
  
	    k +=1
            policy = policy_New
            V = policy_evaluation(P, nS, nA, policy)
            policy_New = policy_improvement(P, nS, nA, V, policy)	
	 

	return V, policy



def example(env):
	"""Show an example of gym
	Parameters
		----------
		env: gym.core.Environment
			Environment to play on. Must have nS, nA, and P as
			attributes.
	"""
	env.seed(0); 
	from gym.spaces import prng; prng.seed(10) # for print the location
	# Generate the episode
	ob = env.reset()
	for t in range(100):
		env.render()
		a = env.action_space.sample()
		ob, rew, done, _ = env.step(a)
		if done:
			break
	assert done
	env.render();

def render_single(env, policy):
	"""Renders policy once on environment. Watch your agent play!

		Parameters
		----------
		env: gym.core.Environment
			Environment to play on. Must have nS, nA, and P as
			attributes.
		Policy: np.array of shape [env.nS]
			The action to take at a given state
	"""

	episode_reward = 0
	ob = env.reset()
	for t in range(100):
		env.render()
		time.sleep(0.5) # Seconds between frames. Modify as you wish.
		a = policy[ob]
		ob, rew, done, _ = env.step(a)
		episode_reward += rew
		if done:
			break
	assert done
	env.render();
	print "Episode reward: %f" % episode_reward


# Feel free to run your own debug code in main!
# Play around with these hyperparameters.
if __name__ == "__main__":
	env = gym.make("Deterministic-4x4-FrozenLake-v0")
	print env.__doc__
	print "Here is an example of state, action, reward, and next state"
	example(env)
		
	V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.9, max_iteration=20, tol=1e-3)

	print '\n Value Iteration'
	print '\nValue\n',V_vi,'\nPolicy\n',p_vi
	
	V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, max_iteration=20, tol=1e-3)

	print '\n Policy Iteration'
	print '\nValue\n',V_pi,'\nPolicy\n',p_pi
	
