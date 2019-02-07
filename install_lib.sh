#!/bin/bash

# It eases the installation of the libs on Mac and Linux 

# install main libs
pip install numpy pandas scikit-learn
pip install gym

# install maze lib
pip install git+https://github.com/rafaie/gym-maze

pip freeze
