# EXECUTION:

# python3 evaluate_shape_predictor.py --predictor eye_predictor.dat --xml ibug_300W_large_face_landmark_dataset/labels_ibug_300W_train_eyes.xml
# python3 evaluate_shape_predictor.py --predictor eye_predictor.dat --xml ibug_300W_large_face_landmark_dataset/labels_ibug_300W_test_eyes.xml

# Import packages

import argparse
import dlib

# Construct the argument parser and parse the arguments

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--predictor", required=True,
    help="path to dlib shape predictor model")

ap.add_argument("-x", "--xml", required=True, 
    help="path to input training/testing XML file")

args = vars(ap.parse_args())

# Compute the error est. over the supplied data split
print("[INFO] evaluating shape predictor...")
error = dlib.test_shape_predictor(args["xml"], args["predictor"])
print("[INFO] error: {}".format(error))