from copy import deepcopy
import random
import numpy as np
from typing import List, Tuple, Dict


def expectation_calc(p: Tuple[int, int, int, int], curr_i, curr_j, U, mdp):
    sum = 0
    curr_state = (curr_i, curr_j)
    if curr_state in mdp.terminal_states:
        return sum
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
    policy = deepcopy(U)
    for i in range(mdp.num_row):
        for j in range(mdp.num_col):
            if mdp.board[i][j] == "WALL":
                continue
            value_action_lst = [(expectation_calc(mdp.transition_function[action], i, j, U,
                                 mdp), action) for action in mdp.actions]
            policy[i][j] = max(value_action_lst, key=lambda tup: tup[0])[1]

    return policy
    # ========================


def outcome(state, curr_action, mdp):
    action_lst = [action for action in mdp.actions]
    weights = [100 * val for val in mdp.transition_function[curr_action]]
    curr_action = random.choices(population=action_lst, weights=weights, k=1)[0]
    new_state = mdp.step(state, curr_action)
    reward = float(mdp.board[state[0]][state[1]])
    done = state in mdp.terminal_states
    return new_state, reward, done


def state_calc(mdp):
    in_qtable = lambda state: not (mdp.board[state[0]][state[1]] == 'WALL')
    state_index_map = {}
    states_number = 0
    for i in range(mdp.num_row):
        for j in range(mdp.num_col):
            state = (i, j)
            if in_qtable(state):
                state_index_map[state] = states_number
                states_number += 1
    return state_index_map, states_number


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

    state_index_map, states_number = state_calc(mdp)
    action_list = [action for action in mdp.actions]
    # action_size = len(action_list)
    # indexing = lambda action: action_list.index(action)
    # state_size = mdp.num_col * mdp.num_row
    qtable = np.zeros((states_number, len(action_list)))
    for episode in range(total_episodes):
        # Reset the environment
        state = init_state
        step = 0
        done = False

        for step in range(max_steps):
            # Choose an action (a) in the current world state (s)

            # First we randomize a number
            exp_exp_tradeoff = random.uniform(0, 1)
            curr_q_index = state_index_map[state]

            # If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
            if exp_exp_tradeoff > epsilon:
                action = action_list[np.argmax(qtable[curr_q_index, :])]

            # Else doing a random choice --> exploration
            else:
                action = random.choice(action_list)


            # Take the action (a) and observe the outcome state(s') and reward (r)
            # print(state, action)
            new_state, reward, done = outcome(state, action, mdp)
            # print(outcome((1, 3), "RIGHT", mdp))
            # return

            new_q_index = state_index_map[new_state]

            action_index = action_list.index(action)
            # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            qtable[curr_q_index, action_index] = qtable[curr_q_index, action_index] + learning_rate * (reward + mdp.gamma *
                                                                             np.max(qtable[new_q_index, :]) - qtable[
                                                                                 curr_q_index, action_index])

            if state in mdp.terminal_states:
                qtable[curr_q_index, :].fill(reward)
            # Our new state is state
            state = new_state

            # If done : finish episode
            if done:
                break

        # Reduce epsilon (because we need less and less exploration)
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    return qtable
    # ========================


def q_table_policy_extraction(mdp, qtable):
    # TODO:
    # Given the mdp and the Qtable:
    # return: the policy corresponding to the Qtable
    #

    # ====== YOUR CODE: ======
    state_index_map, states_number = state_calc(mdp)

    action_list = [action for action in mdp.actions]
    policy = deepcopy(mdp.board)
    for i in range(mdp.num_row):
        for j in range(mdp.num_col):
            if mdp.board[i][j] == "WALL":
                continue
            q_index = state_index_map[(i, j)]
            policy[i][j] = action_list[np.argmax(qtable[q_index, :])]
    return policy
    # ========================


# BONUS

def p_value_calc(s, policy, mdp, states_number, state_index_map):
    if s in mdp.terminal_states:
        return np.zeros(states_number)
    p_dict = {}
    for i in range(mdp.num_row):
        for j in range(mdp.num_col):
            p_dict[(i, j)] = 0
    P = np.zeros(states_number)
    step = policy[s[0]][s[1]]
    if step == 0 or step == 'WALL':
        return P
    possible_states = [mdp.step(s, action) for action in mdp.actions]
    for k in range(len(mdp.actions)):
        p_dict[possible_states[k]] += float(mdp.transition_function[step][k])
    for i in range(mdp.num_row):
        for j in range(mdp.num_col):
            if mdp.board[i][j] == 'WALL':
                continue
            P[state_index_map[(i, j)]] = p_dict[(i, j)]
    return P


def policy_evaluation(mdp, policy):
    # TODO:
    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #

    # ====== YOUR CODE: ======
    state_index_map, states_number = state_calc(mdp)
    P = np.zeros((states_number, states_number))
    R = np.zeros(states_number)
    for i in range(mdp.num_row):
        for j in range(mdp.num_col):
            if mdp.board[i][j] == "WALL":
                continue
            else:
                state = (i, j)
                P[state_index_map[state]] = p_value_calc((i, j), policy, mdp, states_number, state_index_map)
                R[state_index_map[state]] = float(mdp.board[i][j])

    I = np.eye(11)
    U = np.linalg.inv(I - mdp.gamma * P) @ R
    U_reshape = deepcopy(mdp.board)
    for i in range(mdp.num_row):
        for j in range(mdp.num_col):
            if mdp.board[i][j] == "WALL":
                continue
            U_reshape[i][j] = U[state_index_map[(i, j)]]


    # print(U.reshape((mdp.num_row, mdp.num_col)))
    # print("AA")
    return U_reshape
    # ========================


def policy_iteration(mdp, policy_init):
    # TODO:
    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #

    # ====== YOUR CODE: ======
    policy = deepcopy(policy_init)
    while True:
        U = policy_evaluation(mdp, policy)
        unchanged = True
        for i in range(mdp.num_row):
            for j in range(mdp.num_col):
                if mdp.board[i][j] == "WALL":
                    continue
                value_action_lst = [(expectation_calc(mdp.transition_function[action], i, j, U,
                                                      mdp), action) for action in mdp.actions]
                best_value, best_action = max(value_action_lst, key=lambda tup: tup[0])
                policy_action = policy[i][j]
                if policy_action == 0:
                    continue
                if best_value > expectation_calc(mdp.transition_function[policy_action], i, j, U, mdp):
                    policy[i][j] = best_action
                    unchanged = False
        if unchanged:
            return policy

    # ========================
