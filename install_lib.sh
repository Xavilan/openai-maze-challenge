#!/bin/bash

# It eases the installation of the libs on Mac and Linux 

# install main libs
conda install numpy pandas scikit-learn
conda install gym

# install maze lib
pip install git+https://github.com/rafaie/gym-maze
