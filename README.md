# ViewSynthesisMPI-Pytorch

## Overview

This repository contains the implementation of Novel View Synthesis Model from Google in 2020. The training dataset is sourced from RealEstate10k and use the annotations for data labels. This repository does use a depth-prediction model for depth-prediction and ORB for spare 2D points rather than ORB-SLAM3 directly for spare 3D point clouds. Data can be downloaded and prepared using ```data_download.py``` and ```data_preprocess.py```. This repository is intended for educational purposes to illustrate the method of the original paper as well as its performance.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/CodeKnight314/ViewSynthesisMPI-Pytorch.git
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```bash
    python -m venv ViewSynth-env
    source ViewSynth-env/bin/activate
    ```

3. cd to project directory: 
    ```bash 
    cd ViewSynthesisMPI-Pytorch/
    ```

4. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```