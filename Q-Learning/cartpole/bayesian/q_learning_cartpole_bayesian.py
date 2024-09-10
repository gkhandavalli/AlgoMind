import numpy as np
import gym
import optuna
import pickle
from tqdm import tqdm

def train_q_learning(alpha, gamma, epsilon_decay, num_episodes=5000, max_steps_per_episode=200):
    # Create the cart pole environment
    env = gym.make("CartPole-v1")

    # set the number of discrete bins for each continuous state variable
    num_bins = 20

    bins = [
        np.linspace(-4.8, 4.8, num_bins), # cart position
        np.linspace(-4, 4, num_bins), # cart velocity
        np.linspace(-0.418, 0.418, num_bins), # pole angle
        np.linspace(-4, 4, num_bins), # pole angular velocity
    ]

    # Initialize the Q table with zeros
    q_table = np.zeros((num_bins, num_bins, num_bins, num_bins, env.action_space.n))

    # Initialize the exploration rate
    epsilon = 1.0
    min_epsilon = 0.01

    # To store episode rewards for plotting
    episode_rewards = []

    def discretize_state(state):
        """Convert continuous state to discrete state by finding the bin index"""
        state_bin = []
        for i in range(len(state)):
            state_bin.append(np.digitize(state[i],bins[i])-1)
        return tuple(state_bin)
    

    # Q-Learning algorithm
    for episode in range(num_episodes):
        state = env.reset()
        state = np.array(state[0], dtype=np.float32)
        state = discretize_state(state)
        done = False
        total_reward = 0

        # steps in each episode
        for step in range(max_steps_per_episode):
            # Epsilon greedy action selection
            if np.random.uniform(0,1) < epsilon:
                # Explore: choose a random action
                action = env.action_space.sample()

            else:
                # Exploit: choose the action with max q-value
                action = np.argmax(q_table[state])

            # Take action and observe the next state
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.array(next_state, dtype=np.float32)
            next_state = discretize_state(next_state)

            # Update the Q-value using the Bellman equation
            best_next_action = np.argmax(q_table[next_state])
            td_target = reward + gamma * q_table[next_state][best_next_action]
            td_error = td_target - q_table[state][action]
            q_table[state][action] += alpha * td_error

            # Update the state
            state = next_state
            total_reward += reward

            if done:
                break

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        # Store total reward
        episode_rewards.append(total_reward)

    # Calcualte average reward over the last 100 episodes
    average_reward = np.mean(episode_rewards[-100:])

    return average_reward

def objective(trial):
    # Suggest hyperparameters to optimize
    alpha = trial.suggest_float('alpha', 0.01, 0.3)
    gamma = trial.suggest_float('gamma', 0.8, 0.999)
    epsilon_decay = trial.suggest_float('epsilon_decay', 0.99, 0.999)

    # train q-learning
    average_reward = train_q_learning(alpha, gamma, epsilon_decay)
    return average_reward

# Create a study and optimize the objective function
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Best hyperparameters found
print("Best hyperparameters: ", study.best_params)


# Save the best hyperparameters
with open(r".\Q-Learning\cartpole\bayesian\q_learning_cartpole_bayesian_best_hyperparams.pkl", "wb") as f:
    pickle.dump(study.best_params, f)