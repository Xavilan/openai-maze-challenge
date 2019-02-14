"""

generate_leaderboard.py: manage running the tournament and submit the resul to target services.

"""

__copyright__ = "Copyright 2018, MoO"
__license__ = ""
__author__ = "Mostafa Rafaie"
__maintainer__ = "Mostafa Rafaie"


# import pandas as pd
# from tqdm import tqdm
# from multiprocessing import Pool, Lock
# import numpy as np
# import sys
# from score_model import Score, session
# from datetime import datetime
import os
import glob
import argparse
from collect_agents import prepare_tournament_code, reset_base_repo

DEFAULT_AGENTS_PATH = 'agents'
DEFAULT_CUNCURRENCY = 1
RUN_TIME_IGONRED_AGENTS=['base_agent.py', '__init__.py']


class Leaderboard:
    def __init__(self, agents_path="agents"):
        self.agents_path = agents_path

    def get_agents_list(self):
        lst = []
        for i in glob.glob(os.path.join(self.agents_path, "*.py")):
            if i not in RUN_TIME_IGONRED_AGENTS:
                lst.append(i)
        return lst

    def create_agents(self, lst):
        pass
    

    def get_agents_class(self):
        lst = self.get_agents_list()
        agents = self.create_agents(lst)


    def run_a_round(self, agents):
        pass

    def generate(self, cuncurrency):
        # extract all the agents from branches and prepare the base code for the competition
        prepare_tournament_code(agents_path=self.agents_path)

        agents = self.get_agents_class()




        # rest project
        reset_base_repo (agents_path=self.agents_path)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--agents_path', default=DEFAULT_AGENTS_PATH,
                        help="Indicates the default path of agents folder")

    parser.add_argument('-c', '--cuncurrency', default=DEFAULT_AGENTS_PATH,
                        help="Indicates the default cuncurrency to generate the leaderboard")

    args = parser.parse_args()
    l = Leaderboard(args.agents_path)
    l.generate(args.cuncurrency)

