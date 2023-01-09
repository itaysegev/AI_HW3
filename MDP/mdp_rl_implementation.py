from copy import deepcopy
import random
import numpy as np
from typing import List, Tuple, Dict


def expectation_calc(p: Tuple[int, int, int, int], curr_i, curr_j, U, mdp):
    sum = 0
    curr_state = (curr_i, curr_j)
    for k, action in enumerate(mdp.actions):
        next_state = mdp.step(curr_state, action)
        sum += p[k] * U[next_state[0]][next_state[1]]
    return sum


def max_diff(U_new, U, mdp):
    diff = np.zeros((mdp.num_row, mdp.num_col))
    for i in range(mdp.num_row):
        for j in range(mdp.num_col):
            if U[i][j] == "WALL":
                continue
            diff[i][j] = abs(U_new[i][j] - U[i][j])
    return np.amax(diff)



def value_iteration(mdp, U_init, epsilon=10 ** (-3)):
    # TODO:
    # Given the mdp, the initial utility of each state - U_init,
    #   and the upper limit - epsilon.
    # run the value iteration algorithm and
    # return: the U obtained at the end of the algorithms' run.
    #

    # ====== YOUR CODE: ======
    U_new = deepcopy(U_init)
    while True:
        delta = 0
        U = deepcopy(U_new)
        for i in range(mdp.num_row):
            for j in range(mdp.num_col):
                if mdp.board[i][j] == "WALL":
                    continue
                U_new[i][j] = float(mdp.board[i][j]) + mdp.gamma * max(
                    expectation_calc(mdp.transition_function[action], i, j, U,
                                     mdp) for action in mdp.actions)
        delta = max(delta, max_diff(U_new, U, mdp))
        if delta < epsilon * (1 - mdp.gamma) / mdp.gamma:
            return U
    # ========================


def get_policy(mdp, U):
    # TODO:
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================


def q_learning(mdp, init_state, total_episodes=10000, max_steps=999, learning_rate=0.7, epsilon=1.0,
                      max_epsilon=1.0, min_epsilon=0.01, decay_rate=0.8):
    # TODO:
    # Given the mdp and the Qlearning parameters:
    # total_episodes - number of episodes to run for the learning algorithm
    # max_steps - for each episode, the limit for the number of steps
    # learning_rate - the "learning rate" (alpha) for updating the table values
    # epsilon - the starting value of epsilon for the exploration-exploitation choosing of an action
    # max_epsilon - max value of the epsilon parameter
    # min_epsilon - min value of the epsilon parameter
    # decay_rate - exponential decay rate for exploration prob
    # init_state - the initial state to start each episode from
    # return: the Qtable learned by the algorithm
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================


def q_table_policy_extraction(mdp, qtable):
    # TODO:
    # Given the mdp and the Qtable:
    # return: the policy corresponding to the Qtable
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================


# BONUS

def policy_evaluation(mdp, policy):
    # TODO:
    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================


def policy_iteration(mdp, policy_init):
    # TODO:
    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================
