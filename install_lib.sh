#!/bin/bash

# It eases the installation of the libs on Mac and Linux 

# install main libs
pip install numpy pandas scikit-learn
pip install gym

# install maze lib
mkdir -p tmp
cd tmp
git clone https://github.com/rafaie/gym-maze
cd gym-maze
python setup.py install
cd ../..
rm -rf tmp/gym-maze  && rm -rf tmp

pip freeze