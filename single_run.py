"""

single_run.py: Create an enviroment to create a new maze player

"""

__copyright__ = "Copyright 2018, MoM"
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

import gym
import gym_maze
from gym import wrappers

'''
Defining the simulation related constants
'''
DEFAULT_MAZE = "maze-random-10x10-plus-v0"
NUM_EPISODES = 250
MIN_EPISODES = 100
MAX_T = 10000
DEBUG_MODE = 0
RENDER_MAZE = True
ENABLE_RECORDING = True
RECORDING_FOLDER = "game_video_logs"
BASE_AGENT_CLASS_NAME = "BaseAgent"


class Simulator:

    # Initialize the class 
    def __init__(self, agent_name, render_maze=True):
        self.create_enviroment(render_maze)
        self.create_agent(agent_name)

        self.hello_msg()

    def create_enviroment(self, render_maze):
        self.env = gym.make(DEFAULT_MAZE, enable_render=render_maze)
        if ENABLE_RECORDING:
            if os.path.isdir(RECORDING_FOLDER) is False:
                os.mkdir(RECORDING_FOLDER)

            self.env = wrappers.Monitor(self.env, RECORDING_FOLDER, force=True)

    def create_agent(self, agent_name):
        base_path = 'agents'
        agent_path = agent_name
        
        # Attach base folder
        if agent_path.find(os.sep) == -1:
            agent_path = os.path.join(base_path, agent_path)
        
        # Attach python extention (py)
        if agent_path[-3:] != '.py':
            agent_path = agent_path + '.py'

        # Find class name
        with open(agent_path, 'r') as fi:
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
        module_name = agent_path.replace(os.sep, '.').replace('.py', '')
        class_ = getattr(importlib.import_module(module_name), class_name)
        self.agent = class_(DEFAULT_MAZE, NUM_EPISODES, 
                            self.env.observation_space.low, self.env.observation_space.high, 
                            self.env.observation_space.shape, self.env.action_space.n, MAX_T, DEBUG_MODE)


    def hello_msg(self):
        info = self.agent.get_info()
        print ('------------------------------------------')
        print('Agent Info:')
        for k in info:
            print('{} : {}'.format(k, info[k]))


    def run(self):
        num_streaks = 0
        total_reward = 0

        for episode in range(NUM_EPISODES):

            # Reset the environment
            obv = self.env.reset()
            self.agent.reset(obv)

            if RENDER_MAZE is True:
                self.env.render()

            for t in range(MAX_T):
                # Select an action
                action = self.agent.select_action()

                # execute the action
                obv, reward, done, info = self.env.step(action)
                total_reward += reward

                # Observe the result
                self.agent.observe(obv, reward, done, action)

                if RENDER_MAZE is True:
                    self.env.render()
                
                if done:
                    num_streaks += 1

                if done is True or self.agent.need_to_stop_episode() is True:
                    break

            print("Episode %d finished after %f time steps with total reward = %f (streak %d)."
                      % (episode, t, total_reward, num_streaks))

            if episode > MIN_EPISODES and self.agent.need_to_stop_game() is True:
                print("Finish the game in episode ", episode)
                break


    def close(self):
        self.env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--agent_name', required=True,
                        help="It is the relative path to the agent or the name of the agent"+
                        "Example: agents/agent1.py, agent1.py, agent1")
    parser.add_argument('-m', '--maze', default=DEFAULT_MAZE,
                        help="Indicates the maze type. Default = maze-random-10x10-plus-v0")
    parser.add_argument('-n', '--num_episodes', default=NUM_EPISODES,
                        help="Indicates the number of episodes. Default = 250")
    parser.add_argument('-t', '--max_t', default=MAX_T,
                        help="Maximum action in an episode. Default = 10000")
    parser.add_argument('-p', '--min_episodes', default=MIN_EPISODES,
                        help="Minimum episodes to end the game. Default = 100")
    parser.add_argument('-d', '--debug_mode', action="store_true", default=False,
                        help="Activate debug mode. Default = False")
    parser.add_argument('-r', '--no_render_maze', action="store_false", default=True,
                        help="Render the maze. Default = True")
    parser.add_argument('-e', '--enable_recording',  action="store_true", default=False,
                        help="Indicates if it needs to record video of the game. Default = False")

    args = parser.parse_args()

    # Update global variables
    agent_name = args.agent_name
    DEFAULT_MAZE = args.maze
    NUM_EPISODES = args.num_episodes
    MAX_T = args.max_t
    MIN_EPISODES = args.min_episodes
    SOLVED_T = MAX_T / 100
    DEBUG_MODE = args.debug_mode
    RENDER_MAZE = args.no_render_maze
    ENABLE_RECORDING = args.enable_recording
    
    print("++++++++", RENDER_MAZE, args.no_render_maze)
    s = Simulator(agent_name, RENDER_MAZE)

    s.run()
    s.close()

    
