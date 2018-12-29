#!/usr/bin/env bash

sudo apt-get install -y git
sudo apt-get install -y python-pip
sudo apt-get install -y unzip
pip install numpy matplotlib scipy dtcwt
python run.py cnn test
pip install opencv-python
python run.py cnn test
sudo apt-get install libsm6 libxext6 libxrender-dev
sudo apt-get install python-tk
