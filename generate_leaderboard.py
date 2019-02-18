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
# from score_model import Score, session
# from datetime import datetime
import os
import glob
import sys
import argparse
import importlib
import gym
import gym_maze
from multiprocessing import Pool, Lock
from tqdm import tqdm
from collect_agents import prepare_tournament_code, reset_base_repo

# Leaderboard base params
DEFAULT_AGENTS_PATH = 'agents'
DEFAULT_CUNCURRENCY = 1
RUN_TIME_IGONRED_AGENTS=['base_agent.py', '__init__.py']

# game params
DEFAULT_MAZE = "maze-random-10x10-plus-v0"
NUM_EPISODES = 100
MIN_EPISODES = 100
MAX_T = 10000
ROUNDS_COUNT=100
DEBUG_MODE = 0
RENDER_MAZE = False
ENABLE_RECORDING = False
RECORDING_FOLDER = "game_video_logs"
BASE_AGENT_CLASS_NAME = "BaseAgent"


class Leaderboard:
    def __init__(self, agents_path="agents"):
        self.agents_path = agents_path

    def get_agents_list(self):
        lst = []
        for i in glob.glob(os.path.join(self.agents_path, "*.py")):
            if os.path.basename(i) not in RUN_TIME_IGONRED_AGENTS:
                lst.append(i)
        return lst

    def test_agents(self, agents, env):
        agents2 = []

        for a in agents:
            print('--------------------------------------')
            print('Testing the agent:', type(a).__name__)

            try:
                # test get_info function
                a.get_info()

                # test reset function
                obv = env.reset()
                a.reset(obv)

                # test select action function
                action = a.select_action()

                # execute the action
                obv, reward, done, info = env.step(action)

                # test the observe
                a.observe(obv, reward, done, action)

                # test need_to_stop_episode
                a.need_to_stop_episode()

                # test need_to_stop_game
                a.need_to_stop_game()

                print(a.get_info())
                agents2.append(a)
            
            except NotImplementedError:
                print('Removed from list because of NotImplemented Error')
    
            except Exception as e:
                print('Removed from list because of', e)

        return agents2

    def create_agents(self, lst):
        
        env = gym.make(DEFAULT_MAZE, enable_render=RENDER_MAZE)

        agents = []
        for agent_p in lst:
            print('--------------------------------------')
            print('processing to load agent:', agent_p)
            try:
                # Find class name
                with open(agent_p, 'r') as fi:
                    class_name = None
                    for l in fi.readlines():
                        n = l.find('class')
                        if n >= 0:
                            n = l.find(BASE_AGENT_CLASS_NAME)
                            if n > 0:
                                name = l[:n-1].replace('class', '').replace(' ', '')

                                if class_name is None:
                                    class_name = name
                                else:
                                    print("There ate more than one agent class in {}, please store every agent in a new file".format(agent_p))

                # Create agent
                module_name = agent_p.replace(os.sep, '.').replace('.py', '')
                class_ = getattr(importlib.import_module(module_name), class_name)
                agent = class_(DEFAULT_MAZE, NUM_EPISODES, 
                               env.observation_space.low, env.observation_space.high, 
                               env.observation_space.shape, env.action_space.n, MAX_T, DEBUG_MODE)
                
                agents.append(agent)
            except Exception as e:
                print(e)

        agents2 = self.test_agents(agents, env)
        env.close()
        return agents2

    def get_agents_class(self):
        lst = self.get_agents_list()
        print(lst)
        agents = self.create_agents(lst)

    @classmethod
    def run_a_round(self, agents):
        pass

    def generate(self, cuncurrency):
        # extract all the agents from branches and prepare the base code for the competition
        # prepare_tournament_code(agents_path=self.agents_path)

        self.agents = self.get_agents_class()

        # Run the tournoment
        Lock()
        round_results = []
        pool = Pool(cuncurrency)
        for d in tqdm(pool.imap_unordered(self.run_a_round, range(ROUNDS_COUNT)),
                  total=ROUNDS_COUNT):
            round_results.append(d)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--agents_path', default=DEFAULT_AGENTS_PATH,
                        help="Indicates the default path of agents folder")

    parser.add_argument('-c', '--cuncurrency', default=DEFAULT_CUNCURRENCY,
                        help="Indicates the default cuncurrency to generate the leaderboard")

    args = parser.parse_args()
    l = Leaderboard(args.agents_path)
    l.generate(args.cuncurrency)

