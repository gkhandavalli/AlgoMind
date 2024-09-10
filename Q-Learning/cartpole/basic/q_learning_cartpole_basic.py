import numpy as np
import gym
import time
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

# Create the cartpole environment
env = gym.make('CartPole-v1')

# Set the number of discrete bins for each continuous state variable
num_bins = 20
bins = [
    np.linspace(-4.8,4.8, num_bins), # cart position 
    np.linspace(-4,4, num_bins), # cart velocity
    np.linspace(-0.418,0.418, num_bins), # pole angle
    np.linspace(-4,4, num_bins) # pole angular velocity
]

# Initialize the Q table with zeros
q_table = np.zeros((num_bins, num_bins, num_bins, num_bins, env.action_space.n))

# Hyperparameters
alpha = 0.22670341782967035
gamma = 0.9961604467998204
epsilon = 1.0
epsilon_decay = 0.9986520484157592
min_epsilon = 0.01
num_episodes = 5000
max_steps_per_episode = 200

# Training plot data
episode_rewards = []

def discretize_state(state):
    """Convert cont. state to discrete state by finding the bin index"""
    state_bin = []
    for i in range(len(state)):
        state_bin.append(np.digitize(state[i], bins[i]) - 1)
    return tuple(state_bin)


def moving_average(data, window_size):
    """Calculate the moving average for given data and its window size"""
    return np.convolve(data, np.ones(window_size) / window_size, mode = 'valid')

# Q-Learning algorithm
for episode in tqdm(range(num_episodes)):
    state = env.reset()
    state = np.array(state[0], dtype=np.float32)
    state = discretize_state(state)
    done = False
    total_reward = 0 # initialize total reward for the episode

    for step in range(max_steps_per_episode):
        # Epsilon greedy action selection
        if np.random.uniform(0,1) < epsilon:
            action = env.action_space.sample() # Explore: chose a random action
        else:
            action = np.argmax(q_table[state]) # Exploit: choose the action with highest Q-Value

        # Take action and observe the reward and next state
        next_state, reward, done, _, _ = env.step(action)
        next_state = np.array(next_state, dtype=np.float32)
        next_state = discretize_state(next_state)

        # Update the Q-value using the Bellman equation
        best_next_action = np.argmax(q_table[next_state])
        td_target = reward + gamma * q_table[next_state][best_next_action]
        td_error = td_target - q_table[state][action]
        q_table[state][action]+=alpha * td_error

        # Update the state
        state = next_state
        total_reward += reward
        if done:
            break

    # Decay epsilon
    epsilon = max(min_epsilon, epsilon*epsilon_decay)

    # store total reward
    episode_rewards.append(total_reward) # accumulate the total reward for each episode


print("Training finished.")

with open(r".\Q-Learning\cartpole\basic\q_table_cartpole_basic.pkl", "wb") as file:
    pickle.dump(q_table, file)

# Plotting the rewards
plt.figure(figsize=(10, 5))
plt.plot(episode_rewards)
window_size = 50
average_rewards = moving_average(episode_rewards, window_size)
plt.plot(range(window_size - 1, num_episodes), average_rewards, color='red', label='Moving Average Reward')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Reward per Episode and Moving Average during Training')
plt.legend()

# Save the plot to a file
plt.savefig(r".\Q-Learning\cartpole\basic\training_cartpole_basic.png")  # Save as a PNG file
plt.close()  # Close the plot to free up memory

# Visualization
with open(r".\Q-Learning\cartpole\basic\q_table_cartpole_basic.pkl", "rb") as file:
    q_table = pickle.load(file)

    
env = gym.make('CartPole-v1', render_mode='human')
state = env.reset()
state = np.array(state[0], dtype=np.float32)

for _ in tqdm(range(200)):
    env.render()
    state = discretize_state(state)
    action = np.argmax(q_table[state])
    state, reward, done, _, _ = env.step(action)
    time.sleep(0.05)
    if done:
        break

env.close()