"""

tournament_run.py: run the tournament 

"""

__copyright__ = "Copyright 2019, MoO"
__license__ = ""
__author__ = "Mostafa Rafaie"
__maintainer__ = "Mostafa Rafaie"

import sys
from single_run import Simulator


class TournamentSimulator(Simulator):

    # Initialize the class 
    def __init__(self, maze, num_episodes, min_episodes, max_t, 
                 render_maze, enable_recording, debug_mode=False):
        super().__init__(agent_name=None, maze=maze, num_episodes=num_episodes,
                         min_episodes=min_episodes, max_t=max_t, render_maze=render_maze,
                         enable_recording=enable_recording, debug_mode=debug_mode,
                         need_agent=False)

    def run(self, agents, enable_print=False):
        results = []
        for a in agents:
            self.agent = a
            self.agent_name = a.get_info()['team_name']
            r = super().run(enable_print)
            results.append(r)

        return results