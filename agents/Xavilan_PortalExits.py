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


    # Number of portals based on map
    def get_portalCount(self,maze):
        count={
            "maze-random-10x10-plus-v0": 3
            ,"maze-random-20x20-plus-v0": 7
            ,"maze-random-30x30-plus-v0": 10
            ,"maze-random-100x100-v0": 0
            }
        return count.get(maze,0)

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

        # Maze Length
        self.mazeLength=self.observation_space_high[0]+1

        # Goal state
        self.goalState=tuple(self.observation_space_high)
        
        # Create q-table which is used in the Q-Learning algorithm.
        # It builds a multi-dimensional array derived from maze dimension + possible actions.
        # For this example, it establishes a 3-dimensional array with this size [10, 10, 6, 4].
        self.q_table = np.zeros(self.maze_size + (6,self.space_action_n), dtype=float) # StateCol, StateRow, Values (Rewards, Visits, outActionCol, outActionRow, inActionCol, inActionCol), Action

        # Initial exploration map
        self.explored = np.zeros(self.maze_size,dtype=int)
        self.explored[0,0]=1 # start is known
        self.explored[self.goalState]=1 # goal is known
        
        #Punish
        self.punish=-.1/self.mazeLength**2
        
        #Average portal q
        self.portalAvgQ=0.9955  #1+self.punish*self.mazeLength #full scan of initial conditions
        
        # initialized the portal counter
        self.portalCount = 0
        self.portalTotal = self.get_portalCount(self.maze)
        self.portalProb=self.get_portalProb(self.get_unexplored())

        # action inverse
        self.invAction=np.array([1,0,3,2])
        
        # Action does this
        self.colRolAction=np.array([[0,-1],[0,1],[1,0],[-1,0]]) #((0,0,1,-1),(-1,1,0,0)) # (N,S,E,W)
        self.linearPunish=np.array([1,-1,-1,1]) #N,S,E,W

        # Initialize updateBucket
        self.updateBucket=[]

        # Keep the number of tries (episodes)
        self.tries = 0
        
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

        # Keep the number of successful tries (episodes)
        self.num_streaks = 0
        
        # Keep the total reward collected
        self.total_reward = 0
        
        # Determine the current state.
        self.state_0 = None
        self.SOLVED_T = np.prod(self.maze_size, dtype=int)
        self.STREAK_TO_END = 100

        self.tolerance = self.punish/self.num_episodes/100
        self.scanMax = 10000

        # Initialize the q_table as if there were no walls or teleporters
        for i in range(len(self.q_table)):  #Columns
            for j in range(len(self.q_table[i])):  #Rows
                for k in range(self.space_action_n): #Actions
                    if k==0 and j==0 or k==1 and j==self.mazeLength-1 or k==2 and i==self.mazeLength-1 or k==3 and i==0: #if on edge
                        self.q_table[i,j,0,k]=-1  # wall
                        self.q_table[i,j,1,k]=1   # visited
                        self.q_table[i,j,2:4,k]=[i,j]   # same column
                        self.q_table[i,j,4:6,k]=[-1,-1]  # nowhere
                    else:
#                        self.q_table[i,j,0,k]=max(self.portalAvgQ-1/(self.num_episodes+1)*self.punish/self.portalProb, \
#                                                  1-(self.mazeLength-1-i+self.mazeLength-1-j+self.linearPunish[k])*self.punish) # linear away from goal
                        self.q_table[i,j,0,k]=1+(self.mazeLength-1-i+self.mazeLength-1-j+self.linearPunish[k])*self.punish # linear away from goal
                        self.q_table[i,j,1,k]=0   # unvisited
                        self.q_table[i,j,2:4,k]=[i,j]+self.colRolAction[k]   # same column
                        self.q_table[i,j,4:6,k]=[i,j]+self.colRolAction[k]  # same column
        
        # initialize goalState action 
        self.q_table[self.goalState][0:2]=0 # No rewards for leaving the goal and unvisited
        self.q_table[self.goalState][2:4]=[[self.goalState[0]],[self.goalState[1]]] # same column
        
        self.make_Matrix()
        
#        self.q_table[0,1,0,0]=0.9
#        self.q_table[1,0,0,3]=0.9
        test=self.q_table[:,:,0,:]
        self.q_table_update_full()

    def get_unexplored(self):
        return self.mazeLength**2-np.sum(self.explored)

    # probability of finding a portal given the number of squares in the maze and the amount that has been explored
    def get_portalProb(self,unexplored):
        if unexplored!=0:
            portalProb=2*(self.portalTotal-self.portalCount)/unexplored
        else:
            portalProb=0
        return portalProb

    def get_portalAvgQ(self,unexplored):            
        if unexplored!=0: 
            # sum all the unexplored q values in the max divided by the number of unexplored with 1 step of punishment
            portalAvgQ=np.sum(np.amax(self.q_table[:,:,0,:],axis=2)*(1-self.explored))/unexplored+self.punish
        return portalAvgQ
    
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
            'team_name': 'Xavilan_X2',
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
        self.step=0

    # Request the agent to decide about the action
    # It's called by the simulator
    def select_actionx(self):
        
        # Store important variables locally so I can see them  in the variable explorer in Spyder
        # And it makes the formula legible
        state =self.state_0
        visited=(self.q_table[self.state_0][1]>0) # Has this direction been passed thru?
        
        q=self.q_table[self.state_0][0] # Unadjusted expected reward in this direction
        otherQMax=np.zeros(self.space_action_n) # Initialized Maximum expected reward of the other directions
        explored=np.zeros(self.space_action_n) # Initialized explored cell directions
        for i in range(self.space_action_n): # Loop thru the directions
                ones=np.ones(self.space_action_n)  # Initial mask to get the other max q
                ones[i]=0.  # Set current direction to zero so only the rest pass thru
                otherQMax[i]=np.max(q*ones)  # Set the max
                if i==0 and state[1]==0 or i==1 and state[1]==self.mazeLength-1 or i==2 and state[0]==self.mazeLength-1 or i==3 and state[0]==0: #if along the outer edge
                    explored[i]=1  # count edge as explored so no portal probability
                else:
                    explored[i]=self.explored[state[0]+self.colRolAction[0][i],state[1]+self.colRolAction[1][i]]  # set explored cell directions
                    
        if self.mazeLength**2-np.sum(self.explored)!=0:
            portalProb=2*(self.portalTotal-self.portalCount)/(self.mazeLength**2-np.sum(self.explored))*(1-explored)
        else:
            portalProb=0*(1-explored)

        tiebreaker=(self.q_table[self.state_0][0]>0)*self.punish/100*np.random.rand(self.space_action_n)
        runsRemaining=self.num_episodes-self.tries
        portalAvgQ=self.portalAvgQ*np.ones(self.space_action_n)
        walls=(self.q_table[self.state_0][0]==-1)
        wallProb=.5*(1-visited)+walls
        stepPunish=self.punish

        # explored are normal q values
        # unexplored = (1-WallProb)*(
        #                   PortalProb*(PortalAvgQ+max(PortalAvg-OtherQ,0)*RemainingEpisodes)
        #                   +(1-PortalProb)*q)
        #               +WallProb*(-Punish+OtherQ)
        # + random scaled to 1% of punishment size for tie breaking
        q_mod=(visited*q              \
               +(1-visited)                                            \
                   *((1-wallProb)*(
                           portalProb   *(portalAvgQ+(portalAvgQ-otherQMax)*(portalAvgQ>otherQMax)*runsRemaining) \
                         +(1-portalProb)*(q         +(q         -otherQMax)*(q         >otherQMax)*runsRemaining)) \
                   +wallProb*(-stepPunish+otherQMax))                         \
               +tiebreaker \
               )


        action=int(np.argmax(q_mod))
        return action

    def select_action(self):
        
        tiebreaker=0#(self.q_table[self.state_0][0]>0)*self.punish/100*np.random.rand(self.space_action_n)
        q=self.q_table[self.state_0][0]
        
        # q values + 10 for unexplored action + random scaled to half punishment size for tie breaking
        action=int(np.argmax(q+tiebreaker))
        return action

    # It's called by the simulator and share the 
    # information after applying the action by simulator on the environment.
    def observe(self, obv, reward, done, action):
        # Convert state observation received from environment to the internal data structure.
        actState = (obv[0],obv[1])
        self.step+=1
        self.total_reward += reward
        
        if self.total_reward<-.1/self.mazeLength**2*2000:
            test=1

        # Update the Q-Table based on the result. 
        # Actually, it just updates a Q(the previous state + decided action)
        # It's based on Q-Learning algorithm.
        # https://en.wikipedia.org/wiki/Q-learning
        # self.q_table[self.state_0 + (action,)] : It refers to the specific action on a state. 
        #                                          Don't forget that the action is the last dimension of the q-table
        # best_q also returns the q value of the best possible action in the new state.
        
        expState=tuple(self.q_table[self.state_0][2:4,action].astype(int)) # expected state after action out of state_0
        oppState=tuple([self.state_0[0],self.state_0[1]]+self.colRolAction[action]) # state on the opposite side of the wall
        oppAction=self.invAction[action]  #N->S, S->N, E->W, W->E
        scanInd=0 #initial scan indicator
        if self.q_table[self.state_0+(1,action)]==0: #if unvisited action, add to update bucket
            if not(self.state_0 in [i[0:2] for i in self.updateBucket]): # if expState is not in the update bucket
                self.updateBucket.append(self.state_0+(np.amax(self.q_table[self.state_0+(0,)]),)) # append to update bucket
            
        if self.explored[actState]==0: #if the action state was unexplored
            scanInd=1 #Indicate a scan
            if not(actState in [i[0:2] for i in self.updateBucket]): # if expState is not in the update bucket
                self.updateBucket.append(actState+(np.amax(self.q_table[actState+(0,)]),)) # append to update bucket
        self.explored[actState]=1

        self.q_table[self.state_0 + (1,action)]+=1 # add a visit to action starting state
        self.q_table[oppState + (1,oppAction)]+=1 # add a visit to oppAction of opp state
        if actState!=expState: # if actual state is the expected state
            scanInd=1  #if unexpected happens, indicate a scan
            if actState==self.state_0: # found a wall
                self.q_table[self.state_0 + (0,action)]=-1 # set forward action as wall
                self.q_table[self.state_0][2:4,action]=actState # update expected state_0
                if tuple(self.q_table[self.state_0][4:6,action])==oppState: #if the in looking action is from oppState, then state_0 is not on a portal
                    self.q_table[self.state_0][4:6,action]=[-1,-1] # nowhere
                self.q_table[oppState + (0,oppAction)]=-1 # set oppAction of oppState as wall
                self.q_table[oppState][2:4,action]==oppState # set expected column of oppAction of oppState to same column
                self.q_table[expState][4:6,action]=[-1,-1] # doesn't lead to expected state

                if not(self.state_0 in [i[0:2] for i in self.updateBucket]): # if state_0 is not in the update bucket
                    self.updateBucket.append(self.state_0+(np.amax(self.q_table[self.state_0+(0,)]),))  # append to update bucket
                if not(expState in [i[0:2] for i in self.updateBucket]): # if expState is not in the update bucket
                    self.updateBucket.append(expState+(np.amax(self.q_table[expState+(0,)]),)) # append to update bucket
                if expState!=oppState and not(oppState in [i[0:2] for i in self.updateBucket]): #if expected state was a portal and oppState is not in update bucket
                    self.updateBucket.append(oppState+(np.amax(self.q_table[oppState+(0,)]),)) # append the oppState behind the new wall
            else: # found a portal
                self.portalCount+=1 #increment the portal count
                self.explored[expState]=1  #Both ends of portal count as discovered
                self.explored[oppState]=1

                # Swap inward looking state between portal entrance and exit
                temp=self.q_table[expState][4:6,:]
                self.q_table[expState][4:6,:] = self.q_table[actState][4:6,:]
                self.q_table[actState][4:6,:]=temp

                if not(expState in [i[0:2] for i in self.updateBucket]): # if expState is not in the update bucket
                    self.updateBucket.append(expState+(np.amax(self.q_table[expState][0]),)) # append to update bucket
                # Update cells surrounding new portal entrance
                for k in range(self.space_action_n):
                    kstate=(expState[0]+self.colRolAction[0][k],expState[1]+self.colRolAction[1][k])
                    invK=self.invAction[k]
                    if (k==0 and expState[1]!=0 or k==1 and expState[1]!=self.mazeLength-1 or \
                    k==2 and expState[0]!=self.mazeLength-1 or k==3 and expState[0]!=0)\
                    and self.q_table[kstate][0][invK]>0: # north of portal exit, not on northern edge, not a wall, not in goalState
                        self.q_table[kstate][2:4,invK] = actState # set to portal entrance
                        self.q_action_update(kstate,invK)# update the q value of that state looking back in. If it changes, put that state in the list so states looking into it can be checked.
                if not(actState in [i[0:2] for i in self.updateBucket]): # if expState is not in the update bucket
                    self.updateBucket.append(actState+(np.amax(self.q_table[actState+(0,)]),)) # append to update bucket
                # Update cells surrounding new portal exit
                for k in range(self.space_action_n):
                    kstate=tuple([actState[0],actState[1]]+self.colRolAction[k])
                    invK=self.invAction[k]
                    if (k==0 and actState[1]!=0 or k==1 and actState[1]!=self.mazeLength-1 or \
                    k==2 and actState[0]!=self.mazeLength-1 or k==3 and actState[0]!=0)\
                    and self.q_table[kstate+(0,invK)]>0: # north of portal exit, not on northern edge, not a wall, not in goalState
                        self.q_table[kstate][2:4,invK] = expState # set to portal entrance
                        self.q_action_update(kstate,invK)# update the q value of that state looking back in. If it changes, put that state in the list so states looking into it can be checked.
              
                test=1 #debug break in portal catcher

        self.q_table_update()
#        if scanInd==1:
#            self.q_table_update_full()

        # Setting up for the next iteration and update the current state
        self.state_0 = (obv[0],obv[1])
        self.done = done
        if self.step==400:
            test=1 #debug caught in loop
        if self.tries==20 and self.state_0==self.goalState:
            test=1 # debug for examine end of run
        
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

    # Used to sort the update bucket    
    def qkey(self,elem):
        return(elem[2])

    # Update a q value of one action of one state. If that value changes, add the state to the update bucket to check the states looking in
    def q_action_update(self,updateState,action):
        if updateState!=(-1,-1):
            portalProb=self.portalProb
            runsRemaining=self.num_episodes #constant for testing
            portalAvg=self.portalAvgQ
            qCurrent=self.q_table[updateState][0] # current expected reward by action
            walls=(qCurrent==-1)   # Walls by action indicator
            visited=(self.q_table[updateState][1]>0) # Has this direction been passed thru?
            wallProb=.5*(1-visited)+walls  # No wall gets a 0, walls get a 1, unvisited gets a 1/2
            otherQMax=np.zeros(self.space_action_n) # initialize otherQMax by action
            explored=np.zeros(self.space_action_n) # initial explored by action indicator
            for k in range(self.space_action_n): # Loop thru the directions
                ones=np.ones(self.space_action_n)  # Initial mask to get the other max q
                ones[k]=0.  # Set current direction to zero so only the rest pass thru
                otherQMax[k]=np.max(qCurrent*ones)  # Set the max
                if k==0 and updateState[1]==0 or k==1 and updateState[1]==self.mazeLength-1 or k==2 and updateState[0]==self.mazeLength-1 or k==3 and updateState[0]==0: #if along the outer edge
                    explored[k]=1  # count edge as explored so no portal probability
                else:
                    explored[k]=self.explored[tuple([updateState[0],updateState[1]]+self.colRolAction[k])]  # set explored cell directions
            portalProbAction=(1-explored)*portalProb

            if updateState==(1,0) and action==3: 
                test=1 #debug
            stateAction=tuple(self.q_table[updateState][2:4,action].astype(int))
            if self.q_table[updateState+(0,action)]>0: # if not a wall (-1) and not in goalState (0)
                if stateAction==self.goalState:  #if looking at the goal
                    reward=1
                else:
                    reward=self.punish
                q=reward+np.amax(self.q_table[stateAction][0])
                update_q= runsRemaining/(runsRemaining+1)*                                  \
                               (wallProb[k]                           *otherQMax[k]                 \
                              +(1-wallProb[k])*portalProbAction[k]    *max(otherQMax[k],portalAvg)  \
                              +(1-wallProb[k])*(1-portalProbAction[k])*max(otherQMax[k],q))         \
                         +1/(runsRemaining+1)*                                              \
                               (wallProb[k]                           *(self.punish+otherQMax[k])        \
                              +(1-wallProb[k])*portalProbAction[k]    *portalAvg         \
                              +(1-wallProb[k])*(1-portalProbAction[k])*q)
                if abs(update_q-qCurrent[action])>self.tolerance:
                    self.q_table[updateState][0,action]=update_q
                    if not(updateState in [i[0:2] for i in self.updateBucket]):
                        self.updateBucket.append(updateState+(np.amax(self.q_table[updateState+(0,)]),)) # add state looking out to updateBucket to check states looking back in

    # update the q_table if any walls or portals are discovered
    def q_table_update(self):
        scan=0
        while len(self.updateBucket)>0: # while the bucket is not empty
            scan+=1
            self.updateBucket.sort(key=self.qkey, reverse=True) #Sort the Bucket in reverse q expected order
            
            updateState=self.updateBucket.pop(0)[0:2] #pull out the first in bucket
            #Looking back
            for k in range(self.space_action_n): #look back in toward the updateState
                state=tuple(self.q_table[updateState][4:6,k].astype(int)) # this is the state looking back in from that action
                self.q_action_update(state,self.invAction[k]) # update the q value of that state looking back in. If it changes, put that state in the list so states looking into it can be checked.

            if scan>100000:
                test=1 # debug too many scans of the q_table
                
    # update the full q_table if any walls or portals are discovered. rescan until no updates happen.
    def q_table_update_full(self):
        scan=0

        unexplored=self.get_unexplored() #unexplored count doesn't change during a scan

        #Calculated overall portal
        portalProb=self.get_portalProb(unexplored) #test a fixed portal probability to reduce number of scans
#        portalAvg=self.get_portalAvg()
            
        runsRemaining=self.num_episodes-self.tries    #N
    
        while True:
            scan+=1
            updates=0
            initQ=np.zeros(self.maze_size)  #best q per state in q_table for visualization

            for i in range(len(self.q_table)):  #Columns
                for j in range(len(self.q_table[i])):  #Rows
                    portalAvg=self.get_portalAvgQ(unexplored)
                    stateCurrent=(i,j)
                    qCurrent=self.q_table[stateCurrent+(0,)] # current expected reward by action
                    walls=(qCurrent==-1)   # Walls by action indicator
                    visited=(self.q_table[stateCurrent+(1,)]>0) # Has this direction been passed thru?
                    wallProb=.5*(1-visited)+walls  # No wall gets a 0, walls get a 1, unvisited gets a 1/2
                    otherQMax=np.zeros(self.space_action_n) # initialize otherQMax by action
                    explored=np.zeros(self.space_action_n) # initial explored by action indicator
                    for k in range(self.space_action_n): # Loop thru the directions
                        ones=np.ones(self.space_action_n)  # Initial mask to get the other max q
                        ones[k]=0.  # Set current direction to zero so only the rest pass thru
                        otherQMax[k]=np.max(qCurrent*ones)  # Set the max
                        if k==0 and j==0 or k==1 and j==self.mazeLength-1 or k==2 and i==self.mazeLength-1 or k==3 and i==0: #if along the outer edge
                            explored[k]=1  # count edge as explored so no portal probability
                        else:
                            explored[k]=self.explored[tuple([i,j]+self.colRolAction[k])]  # set explored cell directions
                    portalProbAction=(1-explored)*portalProb

                    for k in range(self.space_action_n): #Actions
                        if stateCurrent==(1,0) and k==3: 
                            test=1 #debug
                        stateAction=tuple(self.q_table[stateCurrent][2:4,k].astype(int))
                        if self.q_table[stateCurrent+(0,k)]>0: # if not a wall (-1) and not in goalState (0)
                            if stateAction==self.goalState:  #if looking at the goal
                                reward=1
                            else:
                                reward=self.punish
                            q=reward+np.amax(self.q_table[stateAction+(0,)])
                            
                            update_q= runsRemaining/(runsRemaining+1)*                                  \
                                           (wallProb[k]                           *otherQMax[k]                 \
                                          +(1-wallProb[k])*portalProbAction[k]    *max(otherQMax[k],portalAvg)  \
                                          +(1-wallProb[k])*(1-portalProbAction[k])*max(otherQMax[k],q))         \
                                     +1/(runsRemaining+1)*                                              \
                                           (wallProb[k]                           *(self.punish+otherQMax[k])        \
                                          +(1-wallProb[k])*portalProbAction[k]    *portalAvg         \
                                          +(1-wallProb[k])*(1-portalProbAction[k])*q)
                            if abs(update_q-qCurrent[k])>self.tolerance:
                                self.q_table[stateCurrent+(0,k)]=update_q
                                updates+=1
                        test=1
                    initQ[j,i]=np.amax(self.q_table[i,j,0])
            print(scan,sep="",end="\r",flush=True)
                    #print("Scan", scan, i,j,"q=",update_q, updates)
            if scan%1000==0: #self.state_0==(19,19): 
                test=1 # debug too many scans of the q_table
            if updates==0: # or scan>=self.scanMax: # Redo the update until there are no more updates  
                self.updateBucket=[]
                test=self.q_table[:,:,0,:]
                break
            
    def make_Matrix(self):
        scan=0
        unexplored=self.get_unexplored() #unexplored count doesn't change during a scan
        runsRemaining=self.num_episodes-max(1,self.tries)    #N . tries is never zero. THe initial use would be the first try.
        longProb=runsRemaining/(runsRemaining+1)
        shortProb=1/(runsRemaining+1)
        actions=np.arange(self.space_action_n) #used for argMax

        while True:
            scan+=1
            print(scan,sep="",end="\r",flush=True)
            #Calculated overall portal
            portalProb=self.get_portalProb(unexplored) #test a fixed portal probability to reduce number of scans
            portalAvg=self.get_portalAvgQ(unexplored)
    
            qMatrix=np.zeros(self.maze_size+(self.space_action_n,)+self.maze_size+(self.space_action_n,),dtype=float) # initialize the transition matrix from one state and action to another
            qSolution=np.zeros(self.maze_size+(self.space_action_n,),dtype=float) # initialize the solution "vector"
            qY=np.zeros(self.maze_size+(self.space_action_n,),dtype=float) # initialize the Y "vector"

            for i in range(len(self.q_table)):  #Columns solving for
                for j in range(len(self.q_table[i])):  #Rows solving for
                    stateCurrent=(i,j)
                    qCurrent=self.q_table[stateCurrent+(0,)] # current expected reward by action
                    walls=(qCurrent==-1)   # Walls by action indicator
                    visited=(self.q_table[stateCurrent+(1,)]>0) # Has this direction been passed thru?
                    wallProb=.5*(1-visited)+walls  # No wall gets a 0, walls get a 1, unvisited gets a 1/2
    
                    for k in range(self.space_action_n): # Loop thru the directions
                        if stateCurrent==(19,18) and k==1:
                            test=1
                        otherQMax=np.max(qCurrent-1e6*(actions==k)) #subtract 1,000,000 from current action to remove from check
                        otherQArgMax=np.delete(actions,np.trim_zeros(np.sort((actions+1)*((qCurrent!=otherQMax)*(actions!=k)+(actions==k))))-1) #drop indices where q<>maxQ or same index
    #                    otherQArgMax=np.insert([],actions*((qCurrent!=otherQMax)*(actions!=k)+(actions==k))) #drop indices where q<>maxQ or same index
                        stateAction=tuple(self.q_table[stateCurrent][2:4,k].astype(int))
                        portalProbAction=(1-self.explored[stateAction])*portalProb
    
                        stateActionQMax=np.max(self.q_table[stateAction+(0,)])               
                        stateActionQArgMax=np.delete(actions,np.trim_zeros(np.sort((actions+1)*(self.q_table[stateAction+(0,)]!=stateActionQMax)))-1) #drop indices where q<>maxQ 
                        if stateAction==self.goalState:  #if looking at the goal
                            stateActionQMax+=1
                        else:
                            stateActionQMax+=self.punish
    
                        #state current action coeffficent
                        qMatrix[stateCurrent+(k,)+stateCurrent+(k,)]+=1
                        if stateCurrent!=self.goalState:
                            #long wall: next try will be in another direction tries with punish in the q
                            qMatrix[stateCurrent+(k,)+stateCurrent+(otherQArgMax,)]+=-longProb*wallProb[k]/len(otherQArgMax)
                            #long portal: if otherQ is higher, it will go another direction next tries with punish in the q, 
                            # if portal higher, need to add a punish, divide by unexplored less one for each portal exit and by the number of exit states that have the highest
                            if otherQMax>portalAvg: 
                                qMatrix[stateCurrent+(k,)+stateCurrent+(otherQArgMax,)]+=-longProb*(1-wallProb[k])*portalProbAction/len(otherQArgMax)
                            elif portalProbAction>0:
                                for portalCol in range(len(self.q_table)):
                                    for portalRow in range(len(self.q_table)):
                                        portalExit=(portalCol,portalRow)
                                        if self.explored[portalExit]==0 and portalExit!=stateAction:
                                            portalQMax=np.max(self.q_table[portalExit+(0,)])               
                                            portalQArgMax=np.delete(actions,np.trim_zeros(np.sort((actions+1)*(self.q_table[portalExit+(0,)]!=portalQMax)))-1) #drop indices where q<>maxQ 
                                            qMatrix[stateCurrent+(k,)+portalExit+(portalQArgMax,)]+=-longProb*(1-wallProb[k])*portalProbAction/(unexplored-1)/len(portalQArgMax)
                                qY[stateCurrent+(k,)]+=longProb*(1-wallProb[k])*portalProbAction*self.punish
                            #long normal: the goal gives 1
                            # if otherQ is higher,  it will go another direction next tries with punish in the q,
                            # if stateactionQ higher, need to add a punish
                            if stateAction==self.goalState:
                                qY[stateCurrent+(k,)]+=longProb*(1-wallProb[k])
                            elif otherQMax>stateActionQMax:
                                qMatrix[stateCurrent+(k,)+stateCurrent+(otherQArgMax,)]+=-longProb*(1-wallProb[k])*(1-portalProbAction)/len(otherQArgMax)
                            else:
                                qMatrix[stateCurrent+(k,)+stateAction+(stateActionQArgMax,)]+=-longProb*(1-wallProb[k])*(1-portalProbAction)/len(stateActionQArgMax)
                                qY[stateCurrent+(k,)]+=longProb*(1-wallProb[k])*(1-portalProbAction)*self.punish
                            #short wall: Hitting a wall adds another punish before going another direction
                            qMatrix[stateCurrent+(k,)+stateCurrent+(otherQArgMax,)]+=-shortProb*wallProb[k]/len(otherQArgMax)
                            qY[stateCurrent+(k,)]+=shortProb*wallProb[k]*self.punish
                            #short portal: Add a punish then divide by unexplored less one for each portal exit and by the number of exit states that have the highest
                            if portalProbAction>0:
                                for portalCol in range(len(self.q_table)):
                                    for portalRow in range(len(self.q_table)):
                                        portalExit=(portalCol,portalRow)
                                        if self.explored[portalExit]==0 and portalExit!=stateCurrent:
                                            portalQMax=np.max(self.q_table[portalExit+(0,)])               
                                            portalQArgMax=np.delete(actions,np.trim_zeros(np.sort((actions+1)*(self.q_table[portalExit+(0,)]!=portalQMax)))-1) #drop indices where q<>maxQ 
                                            qMatrix[stateCurrent+(k,)+portalExit+(portalQArgMax,)]+=-shortProb*(1-wallProb[k])*portalProbAction/(unexplored-1)/len(portalQArgMax)
                                qY[stateCurrent+(k,)]+=shortProb*(1-wallProb[k])*portalProbAction*self.punish
                            #short normal
                            if stateAction==self.goalState:
                                qY[stateCurrent+(k,)]+=shortProb*(1-wallProb[k])
                            else:
                                qMatrix[stateCurrent+(k,)+stateAction+(stateActionQArgMax,)]+=-shortProb*(1-wallProb[k])*(1-portalProbAction)/len(stateActionQArgMax)
                                qY[stateCurrent+(k,)]+=shortProb*(1-wallProb[k])*(1-portalProbAction)*self.punish
            
            qMatrixUnr=np.reshape(qMatrix,(self.maze_size[0]*self.maze_size[1]*self.space_action_n,self.maze_size[0]*self.maze_size[1]*self.space_action_n))
            qYUnr=np.ravel(qY)
            qSolutionUnr=np.linalg.solve(qMatrixUnr, qYUnr)
            qSolution=np.reshape(qSolutionUnr,self.maze_size+(self.space_action_n,))
            test=np.max(qSolution,axis=2)
            updates=np.sum(np.argmax(np.round(self.q_table[:,:,0,:],8),axis=2)==np.argmax(np.round(qSolution,8),axis=2))
            if updates>0:
                self.q_table[:,:,0,:]=qSolution
            else:
                test=1
                break
                    
#04/11/2019 04:14pm: Changed learning rate to 1.0 because the maze is deterministic
#04/11/2019 06:41pm: No discount factor. Step punishment and final reward all the same.
#04/11/2019 09:04pm: #Initialize the q_table as if there were no walls or teleporters
#04/12/2019 08:37am: Added the outer walls as a -1.
#04/12/2019 10:28am: Added avoiding walls.
#04/12/2019 10:46am: Added wall memorizing, but not from the other side yet.
#04/12/2019 12:31pm: Fixed initial q bug. Fixed random wall avoidance.
#04/12/2019 01:21pm: Update the q of the attempted state and opposite action.
#04/12/2019 02:33pm: Remembers where it's been.
#04/12/2019 04:12pm: Update the q of oppostion action for portals.
#04/12/2019 08:37pm: Zeroed rewards for leaving goal.
#04/13/2019 09:04am: Added goalState tuple.
#04/13/2019 11:31am: Added action state coordinates to q_table
#04/13/2019 12:45pm: Added update of action and oppAction state coordinates
                    #Fixed bug of updating reward of portal exit if a wall was already known.
                    #Fixed bug not updating visits to portals.
#04/13/2019 03:47pm: Added random number to q value when selecting as a tie breaker.
#04/13/2019 04:21pm: Action is now q values + 10 for unexplored action + random for tie breaking
#04/14/2019 02:11pm: Added update of action coordinates into a new portal entrance.
                    #Moved all updates of q into q_table_update method.
#04/14/2019 03:00pm: Reversed the scan order of the q_table_update as updates tend to propagate up and left.
#04/14/2019 04:34pm: Added in looking state coordinates to q_table
#04/14/2019 05:14pm: Fixed news walls next to known portals.
#04/14/2019 07:32pm: self.updateBucket added along with appends to bucket during wall and portal detections.
#04/15/2019 01:51pm: Fixed bug on the state_0 look back when sitting on a portal.
#04/15/2019 02:36pm: Changed to force reexplore an already visited action.
#04/16/2019 08:54am: Changed named of team to separate it from main team.
#04/16/2019 09:33am: Commented out the exploring the unsearched.
#04/16/2019 12:25pm: Put back exploring the unsearched but made it go down the worse unsearched path.
                    #Added a portal counter.
                    #Changed select_action to stop exploring the unsearched once all portals are found.
#04/16/2019 02:22pm: Stop preferring unexplored directs once expected steps is less than 25.
#04/17/2019 02:55pm: Add exploration map of just the cells.
#04/18/2019 08:40am: Moved visit of oppState oppAction to always, removed wall version of visit. Adjusted if statement accordingly.
                    #Modified select_action to include probability of hit wall, finding a portal. and effect of multiple episodes.
#04/19/2019 03:45pm: Added scan indicator in observe. Won't scan if no changes happen.
