from environment import Environment
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from typing import List


class DqnAgent():
    def __init__(self, action_space: int, observation_space: int) -> None:
        self.action_space = action_space
        self.observation_space = observation_space 
        self.model = self.Construct_Neural_Network()
        self.target_model = self.Construct_Neural_Network()
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

    def load_checkpoint(self) -> None:
        self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
        
def simulation(env, steps_per_episode, epsilon):
    '''
    Simulation execution 

    Parameters
    ----------
    env : TYPE: Class object
        DESCRIPTION: simulation environment
    steps_per_episode : TYPE: int
       
    epsilon : TYPE: float
        DESCRIPTION: epsilon value

    Returns
    -------
    None.

    '''
    start_position = np.array([[4.],[2.],[0.]]) 
    orientation = -1.2
    goal = [np.array([[7.13],[5.51],[0.]]), np.array([[6.866],[6.481],[0.]])]
    goal_threshold = 0.1 
    state1, state2 = env.reset(start_position, orientation, goal, goal_threshold)
    done = False
    total_reward_per_episode=0.
    for i in range(steps_per_episode):
        env.render()
        action1 = agent.policy(state1, epsilon)
        action2 = agent.policy(state2, epsilon)
        next_state1, next_state2, reward, done = env.step([action1,action2])
        total_reward_per_episode += reward
        state1, state2 = next_state1, next_state2
        if done:
            break

env = Environment()
action_space, observation_space =  env.action_space, env.observation_space
agent = DqnAgent(action_space, observation_space)
agent.load_checkpoint()
epsilon = 0
steps_per_episode = 500

def main() -> None:
    try:
        simulation(env, steps_per_episode, epsilon)
    except AssertionError as e:
        print(f'ERROR: {e}, Aborting.')

if __name__ == '__main__':
    main()

