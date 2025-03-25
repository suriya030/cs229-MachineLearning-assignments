"""
CS 229 Machine Learning
Question: Reinforcement Learning - The Inverted Pendulum
"""
from __future__ import division, print_function
from env import CartPole, Physics
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import lfilter

def initialize_mdp_data(num_states):
    transition_counts = np.zeros((num_states, num_states, 2))
    transition_probs = np.ones((num_states, num_states, 2)) / num_states
    reward_counts = np.zeros((num_states, 2)) 
    reward = np.zeros(num_states)
    value = np.random.rand(num_states) * 0.1

    return {
        'transition_counts': transition_counts,
        'transition_probs': transition_probs,
        'reward_counts': reward_counts,
        'reward': reward,
        'value': value,
        'num_states': num_states,
    }

def sample_random_action():
    return 0 if np.random.uniform() < 0.5 else 1

def choose_action(state, mdp_data):
    gamma = 0.995
    transition_probs = mdp_data['transition_probs']
    reward = mdp_data['reward']
    value = mdp_data['value']
    
    q0 = np.dot(transition_probs[state, :, 0], reward + gamma * value)
    q1 = np.dot(transition_probs[state, :, 1], reward + gamma * value)
    
    if q0 > q1:
        return 0
    elif q1 > q0:
        return 1
    else:
        return sample_random_action()

def update_mdp_transition_counts_reward_counts(mdp_data, state, action, new_state, reward):
    mdp_data['transition_counts'][state, new_state, action] += 1
    
    if reward == -1:
        mdp_data['reward_counts'][new_state, 0] += 1
    mdp_data['reward_counts'][new_state, 1] += 1

def update_mdp_transition_probs_reward(mdp_data):
    num_states = mdp_data['num_states']
    transition_counts = mdp_data['transition_counts']
    transition_probs = mdp_data['transition_probs']
    reward_counts = mdp_data['reward_counts']
    reward = mdp_data['reward']
    
    for s in range(num_states):
        for a in [0, 1]:
            total = transition_counts[s, :, a].sum()
            if total > 0:
                transition_probs[s, :, a] = transition_counts[s, :, a] / total
    
    for s in range(num_states):
        total_visits = reward_counts[s, 1]
        if total_visits > 0:
            reward[s] = (reward_counts[s, 0] / total_visits) * (-1)

def update_mdp_value(mdp_data, tolerance, gamma):
    num_states = mdp_data['num_states']
    value = mdp_data['value']
    transition_probs = mdp_data['transition_probs']
    reward = mdp_data['reward']
    
    converged_in_one = False
    iteration = 0
    max_diff = float('inf')
    
    while max_diff >= tolerance:
        new_value = np.zeros(num_states)
        for s in range(num_states):
            q0 = np.dot(transition_probs[s, :, 0], reward + gamma * value)
            q1 = np.dot(transition_probs[s, :, 1], reward + gamma * value)
            new_value[s] = max(q0, q1)
        
        max_diff = np.max(np.abs(new_value - value))
        iteration += 1
        
        if max_diff < tolerance:
            if iteration == 1:
                converged_in_one = True
            break
        value[:] = new_value.copy()
    
    mdp_data['value'][:] = new_value
    return converged_in_one

def main(plot=True):
    np.random.seed(0)

    pause_time = 0.0001
    min_trial_length_to_start_display = 100
    display_started = min_trial_length_to_start_display == 0

    NUM_STATES = 163
    GAMMA = 0.995
    TOLERANCE = 0.01
    NO_LEARNING_THRESHOLD = 20

    time = 0
    time_steps_to_failure = []
    num_failures = 0
    time_at_start_of_current_trial = 0
    max_failures = 500

    cart_pole = CartPole(Physics())
    state_tuple = (0.0, 0.0, 0.0, 0.0)
    state = cart_pole.get_state(state_tuple)
    mdp_data = initialize_mdp_data(NUM_STATES)

    consecutive_no_learning_trials = 0
    while consecutive_no_learning_trials < NO_LEARNING_THRESHOLD:
        action = choose_action(state, mdp_data)
        state_tuple = cart_pole.simulate(action, state_tuple)
        time += 1
        new_state = cart_pole.get_state(state_tuple)

        R = -1 if new_state == NUM_STATES - 1 else 0
        update_mdp_transition_counts_reward_counts(mdp_data, state, action, new_state, R)

        if new_state == NUM_STATES - 1:
            update_mdp_transition_probs_reward(mdp_data)
            converged_in_one_iteration = update_mdp_value(mdp_data, TOLERANCE, GAMMA)

            if converged_in_one_iteration:
                consecutive_no_learning_trials += 1
            else:
                consecutive_no_learning_trials = 0

        if new_state == NUM_STATES - 1:
            num_failures += 1
            if num_failures >= max_failures:
                break
            print(f'[INFO] Failure number {num_failures}')
            time_steps_to_failure.append(time - time_at_start_of_current_trial)
            time_at_start_of_current_trial = time

            if time_steps_to_failure[num_failures - 1] > min_trial_length_to_start_display:
                display_started = 1

            x = -1.1 + np.random.uniform() * 2.2
            state_tuple = (x, 0.0, 0.0, 0.0)
            state = cart_pole.get_state(state_tuple)
        else:
            state = new_state

    if plot:
        log_tstf = np.log(np.array(time_steps_to_failure))
        plt.plot(np.arange(len(time_steps_to_failure)), log_tstf, 'k')
        window = 30
        w = np.array([1/window for _ in range(window)])
        weights = lfilter(w, 1, log_tstf)
        x = np.arange(window//2, len(log_tstf) - window//2)
        plt.plot(x, weights[window:len(log_tstf)], 'r--')
        plt.xlabel('Num failures')
        plt.ylabel('Log of num steps to failure')
        plt.savefig('./control.pdf')

    return np.array(time_steps_to_failure)
    
if __name__ == '__main__':
    main()