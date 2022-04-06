from collections import deque
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from environment import Environment
from typing import List, Tuple

class DqnAgent():
    def __init__(self, action_space: int, observation_space: int) -> None:
        self.action_space = action_space
        self.observation_space = observation_space 
        self.model = self.Construct_Neural_Network()
        self.target_model = self.Construct_Neural_Network()
        self.model_location = r'.\Model'
        # Checkpoints to store the models
        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0), net=self.model)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, 'checkpoints', max_to_keep=20000)
        self.load_checkpoint()
         
    def Construct_Neural_Network(self):
        model = Sequential()
        model.add(Dense(200, input_dim = self.observation_space, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(150, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_space, activation='linear', kernel_initializer='he_uniform'))
        model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.0002), loss='mse')
        return model
    
    def policy(self, state: List[float], epsilon: float) -> int:
        # Epsilon greed 
        if np.random.random() < epsilon: return np.random.randint(0, self.action_space)
        input_states = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
        Qval = self.model(input_states)
        action = np.argmax(Qval.numpy()[0], axis=0)
        return action

    def update_target_network(self) -> None:
        self.target_model.set_weights(self.model.get_weights())
        
    def train(self, batch):
        """ Execute Deep Q-learning Algorithm """
        state_batch, next_state_batch, action_batch, reward_batch, done_batch = batch
        current_state_Qval_batch = self.model(state_batch)
        target_Qval_batch = np.copy(current_state_Qval_batch)
        next_state_Qval_batch = self.target_model(next_state_batch)
        max_next_state_Qval_batch = np.amax(next_state_Qval_batch, axis=1)
        for i in range(state_batch.shape[0]):
            target_Qval_batch[i][action_batch[i]] = reward_batch[i] if done_batch[i] \
                else (reward_batch[i] + 0.99 * max_next_state_Qval_batch[i])
         
        """ Train the Neural Network """
        result = self.model.fit(x=state_batch, y=target_Qval_batch, use_multiprocessing=True)
        self.save_checkpoint()
        return result.history['loss'] 
       
    def save_checkpoint(self) -> None:
        self.checkpoint_manager.save()
        
    def load_checkpoint(self) -> None:
        self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)

class ReplayBuffer:
    def __init__(self) -> None:
        """ Determine the memory length """
        self.Replay_memory = deque(maxlen=100000)

    def Append_experience(self, current_state1, current_state2, next_state1, next_state2, reward, action, done):
        self.Replay_memory.append((current_state1, next_state1, reward, action[0], done))
        self.Replay_memory.append((current_state2, next_state2, reward, action[1], done))
        
    def Sample_minibatch(self):
        """ Fetch random batch of experiences for neural network training """
        batch_size = min(Batch_size, len(self.Replay_memory))
        minibatch = random.sample(self.Replay_memory, batch_size)
        current_state_batch, next_state_batch, action_batch, reward_batch, done_batch = [], [], [], [], []
        for experience in minibatch:
            current_state_batch.append(experience[0])
            next_state_batch.append(experience[1])
            reward_batch.append(experience[2])
            action_batch.append(experience[3])
            done_batch.append(experience[4])
        return np.array(current_state_batch), np.array(next_state_batch), action_batch, reward_batch, done_batch

def AgentTraining(episode_count: int, epsilon: float, buffer, agent, env) -> Tuple[float, float]:
    """ Define starting and goal positions"""
    start_position = np.array([[4.],[2.],[0.]]) 
    orientation = -1.2
    goal = [np.array([[7.13],[5.51],[0.]]), np.array([[6.866],[6.481],[0.]])]
    goal_threshold = 0.1 
    state1, state2 = env.reset(start_position, orientation, goal, goal_threshold)
    done = False
    total_reward_per_episode=0.
    for i in range(Steps_per_episode):
        # env.render()
        action1 = agent.policy(state1, epsilon)
        action2 = agent.policy(state2, epsilon)
        next_state1, next_state2, reward, done = env.step([action1,action2])
        buffer.Append_experience(state1, state2, next_state1, next_state2, reward, [action1, action2], done)
        total_reward_per_episode += reward
        state1, state2 = next_state1, next_state2
        if done:
            break
        
    Minibatch = buffer.Sample_minibatch()          
    """ Train the DQNetwork """
    loss = agent.train(Minibatch)
    
    """ Store the rewards to plot """ 
    rewards.append(total_reward_per_episode)
    Average_rewards.append(sum(rewards[-Average_rewards_over:])/Average_rewards_over)

    # Plot interval
    if episode_count % Plot_every == 0: 
        Plot(rewards, Average_rewards, episode_count)
        
    """ Epsilon decay - exploration vs exploitation """
    if episode_count > Warm_up_episodes: epsilon = max(epsilon * epsilon_decay, epsilon_min) 
    if episode_count % Update_target_every == 0: agent.update_target_network()
    return epsilon, total_reward_per_episode
    
def Plot(rewards, Average_rewards, episode_count) -> None: 
    plt.figure(figsize=(20,14))
    plt.plot(rewards, '0.8')
    plt.plot(Average_rewards, 'r', label='Average reward')    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.title('DQN')
    fig1 = plt.gcf()
    plt.show() 
    fig1.savefig(r".\convergence-graph.png")

def RL_Training(agent, epsilon, buffer, env) -> None:
    for episode_count in range(Episodes_to_train): 
        epsilon, total_reward_per_episode = AgentTraining(episode_count, epsilon, buffer, agent, env)
        print(episode_count, epsilon, total_reward_per_episode)

env = Environment()
action_space, observation_space =  env.action_space, env.observation_space
agent = DqnAgent(action_space, observation_space)
buffer = ReplayBuffer()

# Random seed
RANDOM_SEED = 7
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

''' Training parameters '''
epsilon=1
epsilon_decay=0.9998
epsilon_min=0.05

Start_evaluation_episode = 10_00_000
Plot_every = 100
Warm_up_episodes = 500
Episodes_to_train = 10000
Average_rewards_over = 300
Steps_per_episode = 500
Update_target_every = 4
Batch_size = 12000
    
rewards = []
Average_rewards = []

def main() -> None:
    try:
        RL_Training(agent, epsilon, buffer, env)
    except AssertionError as e:
        print(f'ERROR: {e}, Aborting.')
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
