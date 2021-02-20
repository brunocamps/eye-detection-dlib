# Execution of the script
# python3 train_shape_predictor.py  --training ibug_300W_large_face_landmark_dataset/labels_ibug_300W_train_eyes.xml --model eye_predictor.dat

# Importing necessary packages

import multiprocessing
import argparse
import dlib

# Construct the argument parser and parse the arguments

ap = argparse.ArgumentParser()

ap.add_argument("-t", "--training", required=True,
    help="path to input training XML file")

ap.add_argument("-m", "--model", required=True,
    help="path serialized dlib shape predictor model")

