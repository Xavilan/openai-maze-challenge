
rem It eases the installation of the libs on Mac and Linux 

rem install main libs
pip install numpy pandas scikit-learn
pip install gym

rem install maze lib
md -p tmp
cd tmp
git clone https://github.com/rafaie/gym-maze
cd gym-maze
python setup.py install
cd ../..
rem rm -rf tmp/gym-maze  && rm -rf tmp

pip freeze