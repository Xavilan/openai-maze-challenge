"""

base_agent.py: Base agent class. All the new class needs to inherit from 
               this class.

"""

__copyright__ = "Copyright 2018, MoM"
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
        return { 
            'team_name': 'base_team',
            'team member': 'member1, member2',
            'version': '1.0.0',
            'slogan': 'Nothing :-D',
            'email': 'mostafa.rafaiejokandan@mutualofomaha.com'
        }
    
    # Inform the agent that a new episode is started
    def reset(self, obv):
        pass

    # request the agent to do an action
    def select_action(self):
        pass

    def observe(self, obv, reward, done, action):
        pass
    
    def need_to_stop_episode(self):
        return False

    def need_to_stop_game(self):
        return False
