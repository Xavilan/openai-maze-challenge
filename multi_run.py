"""

single_run.py: Create an enviroment to create a new maze player

"""

__copyright__ = "Copyright 2019, MoO"
__license__ = ""
__author__ = "Mostafa Rafaie"
__maintainer__ = "Mostafa Rafaie"

import sys
import numpy as np
import random
import math
import argparse
import glob
import os
import importlib
import traceback
from datetime import datetime

import gym
import gym_maze
from gym import wrappers

'''
Defining the simulation related constants
'''
DEFAULT_MAZE = "maze-random-20x20-plus-v0" #"maze-random-30x30-plus-v0" #"maze-random-10x10-plus-v0" #"maze-random-100x100-v0" #
NUM_EPISODES = 100
MIN_EPISODES = 100
MAX_T1 = 10000
DEBUG_MODE1 = 0
RENDER_MAZE = True
ENABLE_RECORDING = True
RECORDING_FOLDER = "game_video_logs"
BASE_AGENT_CLASS_NAME = "BaseAgent"


class Simulator:

    # Initialize the class 
    def __init__(self, agent_name, maze, num_episodes, min_episodes, max_t, 
                 render_maze, enable_recording, debug_mode=False, need_agent=True):
        self.agent_name = ['Xavilan','Xavilan_X2'] #agent_name
        self.maze = maze
        self.num_episodes = num_episodes
        self.min_episodes = min_episodes
        self.max_t = max_t
        self.render_maze = render_maze
        self.enable_recording = enable_recording
        self.debug_mode = debug_mode

        self.create_enviroment(render_maze)
        if need_agent is True:
            self.create_agent(self.agent_name)
#            self.hello_msg()

    def create_enviroment(self, render_maze):
        self.env = gym.make(self.maze, enable_render=render_maze)
        if self.enable_recording:
            if os.path.isdir(RECORDING_FOLDER) is False:
                os.mkdir(RECORDING_FOLDER)

            self.env = wrappers.Monitor(self.env, RECORDING_FOLDER, force=True)

    def create_agent(self, agent_name):
        base_path = 'agents'
        agent_path = agent_name
        
        # Attach base folder
        self.agent=[]
        for i in range(len(agent_path)):
            if agent_path[i].find(os.sep) == -1:
                agent_path[i] = os.path.join(base_path, agent_path[i])
        
            # Attach python extention (py)
            if agent_path[i][-3:] != '.py':
                agent_path[i] = agent_path[i] + '.py'

            # Find class name
            with open(agent_path[i], 'r') as fi:
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
                                print("There ate more than one agent class in {}, please store every agent in a new file".format(agent_path))
                                sys.exit(1)

            # Create agent
            module_name = agent_path[i].replace(os.sep, '.').replace('.py', '')
            class_ = getattr(importlib.import_module(module_name), class_name)
            self.agent.append(class_(self.maze, self.num_episodes, 
                                self.env.observation_space.low, self.env.observation_space.high, 
                                self.env.observation_space.shape, self.env.action_space.n, 
                                self.max_t, self.debug_mode))


    def hello_msg(self):
        for i in range(len(self.agent)):
            info = self.agent[i].get_info()
            print ('------------------------------------------')
            print('Agent Info:')
            for k in info:
                print('{} : {}'.format(k, info[k]))


    def run(self, enable_print=True):
        results=[]
        for i in range(len(self.agent)):
            num_streaks = 0
            total_reward = 0
    
            try:
                for episode in range(self.num_episodes):
    
                    # Reset the environment
                    obv = self.env.reset()
                    self.agent[i].reset(obv)
    
                    if self.render_maze is True:
                        self.env.render()
    
                    for t in range(self.max_t):
                        # Select an action
                        action = self.agent[i].select_action()
    
                        # execute the action
                        obv, reward, done, info = self.env.step(action)
                        total_reward += reward
    
                        # Observe the result
                        self.agent[i].observe(obv, reward, done, action)
    
                        if self.render_maze is True:
                            self.env.render()
                        
                        if done:
                            num_streaks += 1
    
                        if done is True or self.agent[i].need_to_stop_episode() is True:
                            break
                    if enable_print is True:
                        print("%s - Episode %d finished after %f time steps with total reward = %f (streak %d)."
                                % (self.agent_name[i], episode, t, total_reward, num_streaks))
    
                    if episode > self.min_episodes and self.agent[i].need_to_stop_game() is True:
                        if enable_print is True:
                            print("Finish the game in episode ", episode)
                        break
                print (self.agent_name[i], total_reward, t+1)
                results.append((total_reward, t))


            except Exception as e:
                if enable_print is True:
                    traceback.print_exc()
                else:
                    if not os.path.isdir('logs'):
                        os.mkdir("logs")
                    runtime = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                    fname = self.agent[i].get_info()['team_name'] + '_' + runtime + '_err.log'
                    with open(os.path.join("logs", fname ), 'w') as fi:
                        fi.write(traceback.format_exc())

#        return [self.agent_name, total_reward, t, episode + 1, num_streaks]
        return results


    def close(self):
        self.env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--agent_name', required=False,
                        help="It is the relative path to the agent or the name of the agent"+
                        "Example: agents/agent1.py, agent1.py, agent1")
    parser.add_argument('-m', '--maze', default=DEFAULT_MAZE,
                        help="Indicates the maze type. Default = maze-random-10x10-plus-v0")
    parser.add_argument('-n', '--num_episodes', default=NUM_EPISODES, type=int,
                        help="Indicates the number of episodes. Default = 250")
    parser.add_argument('-t', '--max_t', default=MAX_T1, type=int,
                        help="Maximum action in an episode. Default = 10000")
    parser.add_argument('-p', '--min_episodes', default=MIN_EPISODES, type=int,
                        help="Minimum episodes to end the game. Default = 100")
    parser.add_argument('-d', '--debug_mode', action="store_true", default=False,
                        help="Activate debug mode. Default = False")
    parser.add_argument('-r', '--no_render_maze', action="store_false", default=False,
                        help="Render the maze. Default = True")
    parser.add_argument('-e', '--enable_recording',  action="store_true", default=False,
                        help="Indicates if it needs to record video of the game. Default = False")

    args = parser.parse_args()
    results=np.zeros((2,2), dtype=float)
    run=0
    for i in range(100):
        run+=1
        print ('run',run)
        s = Simulator(agent_name=args.agent_name,
                      maze=args.maze, num_episodes=args.num_episodes, 
                      min_episodes=args.min_episodes, max_t=args.max_t, 
                      render_maze=args.no_render_maze,
                      enable_recording=args.enable_recording,
                      debug_mode=args.debug_mode)
    
        r = s.run(enable_print=False)
        results[0]+=r[0]
        results[1]+=r[1]
    
    results=results/run
    print(results)
    s.close()

    
