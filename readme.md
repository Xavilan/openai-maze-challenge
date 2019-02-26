# gym-maze

A simple 2D maze environment where an agent (blue dot) finds its way from the top left corner (blue square) to the goal at the bottom right corner (red square). 
The objective is to find the shortest path from the start to the goal.

<kbd>![Simple 2D maze environment](http://i.giphy.com/Ar3aKxkAAh3y0.gif)</kbd>

### Action space
The agent may only choose to go up, down, left, or right ("N", "S", "W", "E"). If the way is blocked, it will remain at the same the location. 

### Observation space
The observation space is the (x, y) coordinate of the agent. The top left cell is (0, 0).

### Reward
A reward of 1 is given when the agent reaches the goal. For every step in the maze, the agent recieves a reward of -0.1/(number of cells).

### End condition
The maze is reset when the agent reaches the goal. 

## Maze Versions
* 10 cells x 10 cells: _MazeEnvRandom10x10_

## Installation
It should work on both Python 3.4+. It requires pygame and numpy.

# How to run it
