"""

generate_leaderboard.py: manage running the tournament and submit the resul to target services.

"""

__copyright__ = "Copyright 2018, MoO"
__license__ = ""
__author__ = "Mostafa Rafaie"
__maintainer__ = "Mostafa Rafaie"


import os
import glob
import sys
import argparse
import importlib
import gym
import gym_maze
from multiprocessing import Pool, Lock
from tqdm import tqdm
import numpy as np
import pandas as pd
from datetime import datetime
from functools import partial
from collections import defaultdict
from sqlalchemy.inspection import inspect
import matplotlib.pyplot as plt
from matplotlib import pylab
import seaborn as sn
from pylab import savefig
from collect_agents import prepare_tournament_code, reset_base_repo
from score_model import Score, session
from tournament_run import TournamentSimulator


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
        return self.create_agents(lst)


    @classmethod
    def run_a_round(self, agents, round):
        t = TournamentSimulator(maze=DEFAULT_MAZE, num_episodes=NUM_EPISODES, 
                                min_episodes=MIN_EPISODES, max_t=MAX_T, 
                                render_maze=RENDER_MAZE, enable_recording=ENABLE_RECORDING, 
                                debug_mode=DEBUG_MODE)
        r = t.run(agents, enable_print=False)
        t.close()
        r = [ [round] + s for s in r]
        return r

    def rset_to_dataframe(self, rset):
        result = defaultdict(list)
        for obj in rset:
            instance = inspect(obj)
            for key, x in instance.attrs.items():
                result[key].append(x.value)
        return pd.DataFrame(result)

    def export_result(self, runtime):
        rset = session.query(Score) \
                       .filter(Score.runtime == runtime)\
                       .order_by(Score.id.desc()).all()

        df = self.rset_to_dataframe(rset)
        df.rename(columns={'team_name':'Team Name'}, inplace=True)

        plt.close("all")
        plt.rcParams['figure.figsize']=(10,5)
        sn.set(style="whitegrid")
        p = sn.lineplot(x='round_no', y='overall_score', hue="Team Name", data=df)

        # Update titles
        plt.title('OpenAI Maze Challenge - Data Science COP (' + runtime + ')')
        plt.xlabel('Round')
        plt.ylabel('Overall Score')


        # Put a legend to the right side
        box = p.get_position()
        p.set_position([box.x0 -0.025, box.y0, box.width * 0.85, box.height]) # resize position
        p.legend(loc='center right', bbox_to_anchor=(1.32, 0.5), ncol=1)

        if os.path.exists("fig") is False:
            os.mkdir("fig")
        figure = p.get_figure()
        figure.savefig(os.path.join("fig", 'result_' + runtime + '.png'), dpi=400)


    def generate(self, cuncurrency):
        # extract all the agents from branches and prepare the base code for the competition
        prepare_tournament_code(agents_path=self.agents_path)

        runtime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.agents = self.get_agents_class()

        # Run the tournoment
        Lock()
        round_no = 0
        round_results = []
        last_score = [0] * len(self.agents)
        pool = Pool(cuncurrency)
        for d in tqdm(pool.imap_unordered(partial(self.run_a_round, self.agents), range(ROUNDS_COUNT)),
                  total=ROUNDS_COUNT):
            round_results.append(d)
            round_no += 1
            last_score = [ last_score[i] + d[i][2] for i in range(len(self.agents))] 

            for j in range(len(self.agents)):
                s = Score(runtime=runtime, round_no=round_no,
                          team_name=d[j][1], score=d[j][2],
                          overall_score=last_score[j],
                          param_int1=d[j][3])
                session.add(s)
            session.commit()

        self.export_result(runtime)
        reset_base_repo()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--agents_path', default=DEFAULT_AGENTS_PATH,
                        help="Indicates the default path of agents folder")

    parser.add_argument('-c', '--cuncurrency', default=DEFAULT_CUNCURRENCY,
                        help="Indicates the default cuncurrency to generate the leaderboard")

    args = parser.parse_args()
    l = Leaderboard(args.agents_path)
    l.generate(args.cuncurrency)

