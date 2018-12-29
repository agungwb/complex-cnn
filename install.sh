#!/usr/bin/env bash

sudo apt-get install -y git
sudo apt-get install -y python-pip
sudo apt-get install -y unzip
sudo apt-get install -y libsm6 libxext6 libxrender-dev
sudo apt-get install -y python-tk

pip install --upgrade pip
pip install numpy matplotlib scipy dtcwt opencv-python cupy-cuda80