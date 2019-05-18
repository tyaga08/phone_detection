#!/bin/bash

echo "installing the required python packages"

# install pip3 for python
sudo apt-get install python3-pip

# install python packages
sudo pip3 install numpy scikit-learn joblib imutils