# EXEC
# python3 predict_eyes.py --shape-predictor eye_predictor.dat

# Import packages
# on branch image -> refactored to read image instead of video
from imutils import face_utils
from imutils.video import VideoStream
import argparse
import imutils
import time
import dlib
import cv2

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
    help="path to facial landmark predictor")
args = vars(ap.parse_args())

# Initialize dlib's face detector (HOG-based) and then load
# trained shape predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# read image 

frame = cv2.imread('image.jpg')
frame = imutils.resize(frame, width=400)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

rects = detector(gray, 0)

for rect in rects:
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    print(rect)
    cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
    
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    for (sX, sY) in shape:
        cv2.circle(frame, (sX, sY), 1, (0, 0, 255), -1)
    
    # save the frame
    cv2.imwrite('eyes_detected.jpg', frame )
    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key was pressed, break frmo loop
    if key == ord("q"):
        break

# Clean up
cv2.destroyAllWindows()
