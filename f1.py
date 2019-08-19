from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import math

def distance(p1,p2):
        return math.sqrt(math.pow(p1[0]-p2[0],2)+math.pow(p1[1]-p2[1],2))


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# load the input image, resize it, and convert it to grayscale
image = cv2.imread("images/example_10.jpg")
#image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale image
rects = detector(gray, 1)
# loop over the face detections
for (i, rect) in enumerate(rects):
	# determine the facial landmarks for the face region, then
	# convert the facial landmark (x, y)-coordinates to a NumPy
	# array
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)
	
       
	# loop over the (x, y)-coordinates for the facial landmarks
	# and draw them on the image
	for (x, y) in shape:
		cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
# show the output image with the face detections + facial landmarks
cv2.line(image,tuple(shape[36].reshape(1, -1)[0]),tuple(shape[45].reshape(1, -1)[0]),(255,0,0),2)
cv2.imshow("Output", image)
cv2.waitKey(0)

#print(type(shape[37]),tuple(shape[40].reshape(1, -1)[0]),shape[43],shape[46])
print(2*distance(tuple(shape[8].reshape(1, -1)[0]),tuple(shape[27].reshape(1, -1)[0])))



