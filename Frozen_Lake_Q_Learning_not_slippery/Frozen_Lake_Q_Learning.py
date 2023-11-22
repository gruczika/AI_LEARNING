import gymnasium as gym
import numpy as np
env = gym.make('FrozenLake-v1', render_mode='human', is_slippery = False)
obs_shape = env.observation_space.n
action_shape = env.action_space.n
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability
epsilon_decay_rate = 0.99         # Exponential decay rate for exploration prob
epsilon = max_epsilon

discount_factor = 0.9
learning_rate = 0.7
print(obs_shape)
print(action_shape)

q_table = np.zeros((obs_shape, action_shape))

def bellman_equation(reward, next_state):
    return reward + discount_factor * np.max(q_table[next_state][:])

def temporal_difference(state, action, reward, next_state):
    return q_table[state, action] + learning_rate * (bellman_equation(reward, next_state) - q_table[state, action])

def distance(x1, y1, x2, y2):
    return np.sqrt(np.square(x2-x1) + np.square(y2-y1))

def get_action(state):
    random = np.random.random()
    e_greedy = epsilon < random 
    if e_greedy:
        return np.argmax(q_table[state])
    else: 
        return env.action_space.sample()
    
def get_coordinates(state):
    state_x = state % 4 
    state_y = int((state - state_x) / 4)
    return state_x, state_y

for episode in range(1, 1000):
    done = False
    state, _ = env.reset()
    epsilon = max(min_epsilon, epsilon_decay_rate * epsilon)
    goal_x = 3
    goal_y = 3
    
    while not done: 
        state_x, state_y = get_coordinates(state)
        action = get_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state_x, next_state_y = get_coordinates(next_state)
        cur_distance = distance(state_x, state_y, goal_x, goal_y)
        next_distance = distance(next_state_x, next_state_y, goal_x, goal_y)
        if terminated or next_distance >= cur_distance:
            reward -= 0.1
        if next_distance < cur_distance:
            reward += 0.1
        q_table[state, action] = temporal_difference(state, action, reward, next_state)
        done = terminated or truncated
        state = next_state
        epsilon = max(epsilon_decay_rate*epsilon, epsilon) 
