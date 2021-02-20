# Turning a custom dlib shape predictor

# Project Objective

Train DLIB to verify is a pair of eyes is framed and where it's framed

# Project structure

1. Extracted the IBUG 300-W dataset to main folder
2. parse_xml.py is responsible for parsing the train/test XML dataset files for eyes-only landmark coordinates
3. train_shape_predictor.py Accepts the parsed XML files to train our shape predictor with dlib
4. evaluate_shape_predictor.py: Calculates the Mean Average Error (MAE) of our custom shape predictor
5. predict_eyes.py: Performs shape prediction using our custom dlib shape predictor, trained only to recognize eye landmarks.

The iBUG-300W is a model used to train our shape predictor. It provides (x,y)-coordinate pairs for all facial structures. However, we want to train our shape predictor on just the eyes.

To train our shape predictor on just the eyes, we will parse the XML annotations and create a new training file that includes just the eye coordinates. 

## Warning: Experimental code with multiple comments.