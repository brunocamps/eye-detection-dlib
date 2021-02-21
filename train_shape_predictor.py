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

args = vars(ap.parse_args())

# Default options for dlib's shape predictor

print("[INFO] setting shape predictor options...")
options = dlib.shape_predictor_training_options()

# Setup the tree depth options:
# Depth of each regression tree in the Ensemble of Regression
# Trees (ERTs). Try to balance depth with speed.
options.tree_depth = 4

# Nu is the floating point value used as a regularization
# parameter to help our model generalize
options.nu = 0.1

# Cascade depth impacts accuracy and output size.
# More cascades, the bigger the model will become
options.cascade_depth = 15

# Feature of pool size is the number of pixels to generate
# features for the random trees at each cascade
options.feature_pool_size = 400

# Num test splits selects best features at each cascade when training
options.num_test_splits = 50

# Oversampling amount controls the "jitter" when training
# the shape predictor 
options.oversampling_amount = 5

# Oversampling translation jitter constrols the amount
# of translation augmentation applied to our training set
options.oversampling_translation_jitter = 0.1

# Be verbose indicates to print or not the status
options.be_verbose = True

# Define the number of CPUs cores to be used when training
options.num_threads = multiprocessing.cpu_count()

# log our training options to terminal
print("[INFO] shape predictor options: ")
print(options)

# Train the shape predictor
print("[INFO] training shape predictor...")
dlib.train_shape_predictor(args["training"], 
args["model"], options)


