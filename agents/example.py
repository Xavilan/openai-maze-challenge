"""

base_agent.py: It is an example agaent

"""

__copyright__ = "Copyright 2018, MoM"
__license__ = ""
__author__ = "Mostafa Rafaie"
__maintainer__ = "Mostafa Rafaie"

from agents.base_agent import BaseAgent
import numpy as np
import math
import random

class Example(BaseAgent):
    

    def __init__(self, maze, num_episodes, observation_space_low, observation_space_high, 
                 observation_space_shape, space_action_n, max_t, debug_mode):
        super().__init__(maze, num_episodes, observation_space_low, observation_space_high, 
                        observation_space_shape, space_action_n, max_t, debug_mode)
        self.init_example_agent()


    def init_example_agent(self):
        '''
        Creating a Q-Table for each state-action pair
        '''
        self.maze_size = tuple((self.observation_space_high + np.ones(self.observation_space_shape)).astype(int))
        self.num_buckets = self.maze_size
        self.q_table = np.zeros(self.maze_size + (self.space_action_n,), dtype=float)

        self.MIN_EXPLORE_RATE = 0.001
        self.MIN_LEARNING_RATE = 0.2
        self.DECAY_FACTOR = np.prod(self.maze_size, dtype=float) / 10.0
        self.state_bounds = list(zip(self.observation_space_low, self.observation_space_high))

        # Instantiating the learning related parameters
        self.learning_rate = self.get_learning_rate(0)
        self.explore_rate = self.get_explore_rate(0)
        self.discount_factor = 0.99

        self.tries = 0
        self.num_streaks = 0
        self.total_reward = 0
        self.state_0 = None
        self.SOLVED_T = np.prod(self.maze_size, dtype=int)
        self.STREAK_TO_END = 100



    def state_to_bucket(self, state):
        bucket_indice = []
        for i in range(len(state)):
            if state[i] <= self.state_bounds[i][0]:
                bucket_index = 0
            elif state[i] >= self.state_bounds[i][1]:
                bucket_index = self.num_buckets[i] - 1
            else:
                # Mapping the state bounds to the bucket array
                bound_width = self.state_bounds[i][1] - self.state_bounds[i][0]
                offset = (self.num_buckets[i]-1)*self.state_bounds[i][0]/bound_width
                scaling = (self.num_buckets[i]-1)/bound_width
                bucket_index = int(round(scaling*state[i] - offset))
            bucket_indice.append(bucket_index)
        return tuple(bucket_indice)


    # get the agent information
    def get_info(self):
        return { 
            'team_name': 'TEAM1_EXAMPLE',
            'team member': 'member1',
            'version': '1.0.0',
            'slogan': 'Nothing :-D',
            'email': ''
        }
    
    # Inform the agent that a new episode is started
    def reset(self, obv):
        self.state_0 = self.state_to_bucket(obv)
        self.done = False
        self.explore_rate = self.get_explore_rate(self.tries)
        self.learning_rate = self.get_learning_rate(self.tries)
        self.tries += 1
        self.total_reward = 0

    # request the agent to do an action
    def select_action(self):
        # Select a random action
        if random.random() < self.explore_rate:
            action = int(np.random.uniform(0,4)) #np.random.randint(self.space_action_n)
        # Select the action with the highest q
        else:
            action = int(np.argmax(self.q_table[self.state_0]))
        return action

    def observe(self, obv, reward, done, action):
        state = self.state_to_bucket(obv)
        self.total_reward += reward

        # Update the Q based on the result
        best_q = np.amax(self.q_table[state])
        self.q_table[self.state_0 + (action,)] += self.learning_rate * (reward + self.discount_factor * (best_q) - self.q_table[self.state_0 + (action,)])

        # Setting up for the next iteration
        self.state_0 = self.state_to_bucket(obv)
        self.done = done
    
    def need_to_stop_episode(self):
        return False

    def need_to_stop_game(self):
        if self.done is True:
            if self.tries <= self.SOLVED_T:
                self.num_streaks += 1
            else:
                self.num_streaks = 0

        if self.num_streaks > self.STREAK_TO_END:
            return True

        return False

    def get_explore_rate(self, t):
        return max(self.MIN_EXPLORE_RATE, min(0.8, 1.0 - math.log10((t+1)/self.DECAY_FACTOR)))

    def get_learning_rate(self, t):
        return max(self.MIN_LEARNING_RATE, min(0.8, 1.0 - math.log10((t+1)/self.DECAY_FACTOR)))
