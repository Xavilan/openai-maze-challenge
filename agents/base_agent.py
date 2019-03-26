"""

base_agent.py: Base agent class. All the new class needs to inherit from 
               this class.

"""

__copyright__ = "Copyright 2018, MoO"
__license__ = ""
__author__ = "Mostafa Rafaie"
__maintainer__ = "Mostafa Rafaie"


class BaseAgent:

    def __init__(self, maze, num_episodes, observation_space_low, observation_space_high, 
                 observation_space_shape, space_action_n, max_t, debug_mode):
        self.maze = maze
        self.num_episodes = num_episodes
        self.observation_space_low = observation_space_low
        self.observation_space_high = observation_space_high
        self.observation_space_shape = observation_space_shape
        self.space_action_n = space_action_n
        self.max_t = max_t
        self.debug_mode = debug_mode
    
    # get the agent information
    def get_info(self):
        print("Subclasses should implement get_info function!")
        raise NotImplementedError
    
    # Inform the agent that a new episode is started
    def reset(self, obv):
        pass

    # request the agent to do an action
    def select_action(self):
        print("Subclasses should implement select_action function!")
        raise NotImplementedError

    def observe(self, obv, reward, done, action):
        print("Subclasses should implement observe function!")
        raise NotImplementedError
    
    def need_to_stop_episode(self):
        return False

    def need_to_stop_game(self):
        return False
