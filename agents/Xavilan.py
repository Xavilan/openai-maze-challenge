"""

Xavilan.py: Dan Freimund's agent started from an example agent
               It's a new version driven from below example:
               https://github.com/MattChanTK/ai-gym/blob/master/maze_2d/maze_2d_q_learning.py

"""

__copyright__ = "Copyright 2019, MoO"
__license__ = ""
__author__ = "Dan Freimund"
__maintainer__ = "Dan Freimund"

from agents.base_agent import BaseAgent
import numpy as np
import math
import random

class Xavilan(BaseAgent):
    

    def __init__(self, maze, num_episodes, observation_space_low, observation_space_high, 
                 observation_space_shape, space_action_n, max_t, debug_mode):
        super().__init__(maze, num_episodes, observation_space_low, observation_space_high, 
                        observation_space_shape, space_action_n, max_t, debug_mode)
        self.init_example_agent()


    def init_example_agent(self):
        '''
        Creating a Q-Table for each state-action pair
        '''

        # Create a tuple with the maze size. As an example for the Random 10x10 random maze, it works in this way:
        # self.observation_space_high is a list [9, 9] ( I guess it's not [10, 10] as the first value of the array is started from zero)
        # self.observation_space_shape is a tuple showing the dimension of the maze. for this example, it's (2,)
        # and finally, self.maze_size is converted to actual size [10, 10]!
        self.maze_size = tuple((self.observation_space_high + np.ones(self.observation_space_shape)).astype(int))
        # Refers to the possible states with the actual dimension. For this example, it's [10, 10] 
        self.num_buckets = self.maze_size
        # Create q-table which is used in the Q-Learning algorithm.
        # It builds a multi-dimensional array derived from maze dimension + possible actions.
        # For this example, it establishes a 3-dimensional array with this size [10, 10, 4].
        self.q_table = np.zeros(self.maze_size + (self.space_action_n,), dtype=float)
        
        #Initialize the q_table as if there were no walls or teleporters
        for i in range(len(self.q_table)):  #Columns
        	  for j in range(len(self.q_table[i])):  #Rows
        	  	  #North
        	  	  if j==0:
        	  	  		self.q_table[i][j][0]=-1  #Wall
        	  	  else:
        	  	  		self.q_table[i][j][0]=1-(len(self.q_table)-1-i+len(self.q_table)-1-j+1)*.1/len(self.q_table)**2
        	  	 	#South
        	  	  if j==len(self.q_table)-1:
        	  	  		self.q_table[i][j][1]=-1  #Wall
        	  	  else:
        	  	  		self.q_table[i][j][1]=1-(len(self.q_table)-1-i+len(self.q_table)-1-j-1)*.1/len(self.q_table)**2
        	  	  #East
        	  	  if i==len(self.q_table)-1:
        	  	  		self.q_table[i][j][2]=-1  #Wall
        	  	  else:
        	  	  		self.q_table[i][j][2]=1-(len(self.q_table)-1-i+len(self.q_table)-1-j-1)*.1/len(self.q_table)**2
        	  	  #West
        	  	  if i==0:
        	  	  		self.q_table[i][j][3]=-1  #Wall
        	  	  else:
        	  	  		self.q_table[i][j][3]=1-(len(self.q_table)-1-i+len(self.q_table)-1-j+1)*.1/len(self.q_table)**2

        # A variable for keeping the minimum exploration rate and learning rate
        self.MIN_EXPLORE_RATE = 0.001
        self.MIN_LEARNING_RATE = 1.0 # Maze is deterministic, so set learning rate to 1.0

        # It's a decay factor to change the exploration and exploitation rate in every step.
        # for this example, it generates 10.0
        self.DECAY_FACTOR = np.prod(self.maze_size, dtype=float) / 10.0
        
        # It just define the bound of the maze in every dimension
        # for this example, it generates [(0, 9), (0, 9)]
        self.state_bounds = list(zip(self.observation_space_low, self.observation_space_high))

        # Instantiating the learning related parameters
        self.learning_rate = self.get_learning_rate(0)
        self.explore_rate = self.get_explore_rate(0)

        # A Q-Learning variable 
        # is a number between 0 and 1 and has the effect of valuing rewards received earlier higher 
        # than those received later (reflecting the value of a "good start"). It may also be interpreted 
        # as the probability to succeed (or survive) at every step.
        # https://en.wikipedia.org/wiki/Q-learning#Discount_factor
        self.discount_factor = 1.00 # No discount factor. Step punishment and final reward all the same.

        # Keep the number of tries (episodes)
        self.tries = 0

        # Keep the number of successful tries (episodes)
        self.num_streaks = 0
        
        # Keep the total reward collected
        self.total_reward = 0
        
        # Determine the current state.
        self.state_0 = None
        self.SOLVED_T = np.prod(self.maze_size, dtype=int)
        self.STREAK_TO_END = 100


    # It's the function to map the observed state received from the environment
    # to bucket as the internal dataset keeping the state info. For this example,
    # there is no need for scaling, but I prefer to use this function as I wasn't
    # sure if it works for other maze environment or not.
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


    # Get the agent information
    # It's required to be updated by every team.
    def get_info(self):
        return { 
            'team_name': 'Xavilan',
            'team member': 'Dan Freimund',
            'version': '1.1.0',
            'slogan': 'Propter quod vincere!',
            'email': 'daniel.freimund@mutualofomaha.com'
        }
    
    # Inform the agent that a new episode is started
    # It actually called by simulator informing the agent
    # the game reset 
    def reset(self, obv):
        # Reset the current state and other variables.
        self.state_0 = self.state_to_bucket(obv)
        self.done = False
        self.explore_rate = self.get_explore_rate(self.tries)
        self.learning_rate = self.get_learning_rate(self.tries)
        self.tries += 1
        self.total_reward = 0

    # Request the agent to decide about the action
    # It's called by the simulator
    def select_action(self):
        # Select a random action
        randx=random.random()
        if randx < self.explore_rate:
            
            action = int(np.random.uniform(0,4))
            if self.q_table[self.state_0+(action,)]==-1: #if a wall move to next action
            	action=(action+1)%4  #stay within 0 to 3
            	if self.q_table[self.state_0+(action,)]==-1:
            		action=(action+1)%4
            		if self.q_table[self.state_0+(action,)]==-1:
            		    action=(action+1)%4
            		    if self.q_table[self.state_0+(action,)]==-1:
            		        action=(action)+1%4
            		
        # Select the action with the highest q from Q-table
        else:
            action = int(np.argmax(self.q_table[self.state_0]))

        return action

    # It's called by the simulator and share the 
    # information after applying the action by simulator on the environment.
    def observe(self, obv, reward, done, action):
        # Convert state observation received from environment to the internal data structure.
        state = self.state_to_bucket(obv)
        self.total_reward += reward

        # Update the Q-Table based on the result. 
        # Actually, it just updates a Q(the previous state + decided action)
        # It's based on Q-Learning algorithm.
        # https://en.wikipedia.org/wiki/Q-learning
        # self.q_table[self.state_0 + (action,)] : It refers to the specific action on a state. 
        #                                          Don't forget that the action is the last dimension of the q-table
        # best_q also returns the q value of the best possible action in the new state.
        
        if action==0: #North
        		oppAction=1
        		oppState=(self.state_0[0],self.state_0[1]-1)
        if action==1: #South
        		oppAction=0
        		oppState=(self.state_0[0],self.state_0[1]+1)
        if action==2: #East
        		oppAction=3
        		oppState=(self.state_0[0]+1,self.state_0[1])
        if action==3: #West
        		oppAction=2
        		oppState=(self.state_0[0]-1,self.state_0[1])

        if state==self.state_0: #Hit a wall
        		self.q_table[self.state_0 + (action,)]=-1 #It's a wall.
        		self.q_table[oppState + (oppAction,)]=-1 #It's a wall.
        else:  #Didn't hit a wall
        		best_q = np.amax(self.q_table[state])
        		self.q_table[self.state_0 + (action,)] += self.learning_rate * (reward + self.discount_factor * (best_q) - self.q_table[self.state_0 + (action,)])
        		oppBest_q = np.amax(self.q_table[self.state_0])
        		self.q_table[oppState + (oppAction,)] += self.learning_rate * (reward + self.discount_factor * (oppBest_q) - self.q_table[oppState + (oppAction,)])

        # Setting up for the next iteration and update the current state
        self.state_0 = self.state_to_bucket(obv)
        self.done = done
    
    # Give control to stop the episodes if the agent needs!
    def need_to_stop_episode(self):
        return False

    # Give control to stop the game if the agent needs!
    # In the leaderboard, it is controlled by the simulator, and it will not work.
    def need_to_stop_game(self):
        if self.done is True:
            if self.tries <= self.SOLVED_T:
                self.num_streaks += 1
            else:
                self.num_streaks = 0

        if self.num_streaks > self.STREAK_TO_END:
            return True

        return False

    # Generate an exploration rate dependent on the steps of an episode by using a logarithmic equation
    def get_explore_rate(self, t):
        return max(self.MIN_EXPLORE_RATE, min(0.8, 1.0 - math.log10((t+1)/self.DECAY_FACTOR)))

    # Generate a learning rate dependent on the steps of an episode by using a logarithmic equation
    def get_learning_rate(self, t):
        return max(self.MIN_LEARNING_RATE, min(0.8, 1.0 - math.log10((t+1)/self.DECAY_FACTOR)))

#04/11/2019 04:14pm: Changed learning rate to 1.0 because the maze is deterministic
#04/11/2019 06:41pm: No discount factor. Step punishment and final reward all the same.
#04/11/2019 09:04pm: #Initialize the q_table as if there were no walls or teleporters
#04/12/2019 08:37am: Added the outer walls as a -1.
#04/12/2019 10:28am: Added avoiding walls.
#04/12/2019 10:46am: Added wall memorizing, but not from the other side yet.
#04/12/2019 12:31pm: Fixed initial q bug. Fixed random wall avoidance.
#04/12/2019 01:21pm: Update the q of the attempted state and opposite action.
