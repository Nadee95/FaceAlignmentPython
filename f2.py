from imutils import face_utils
import dlib
import cv2
import math
 
def distance(p1,p2):
        return math.sqrt(math.pow(p1[0]-p2[0],2)+math.pow(p1[1]-p2[1],2))

p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

cap = cv2.VideoCapture(0)
_,imgcap = cap.read()
capsize=0

while True:
    # Getting out image by webcam 
    _, image = cap.read()
    # Converting the image to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    # Get faces into webcam's image
    rects = detector(gray, 0)
    
    # For each detected face, find the landmark.
    for (i, rect) in enumerate(rects):
        # Make the prediction and transfom it to numpy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        cv2.line(image,tuple(shape[36].reshape(1, -1)[0]),tuple(shape[45].reshape(1, -1)[0]),(255,0,0),2)
        # Draw on our image, all the finded cordinate points (x,y) 
        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

        capsize=(2*distance(tuple(shape[8].reshape(1, -1)[0]),tuple(shape[27].reshape(1, -1)[0])))
        print("face size(pixels) %s ",capsize)
        print(cap.get(14))
    
    
    if(capsize>200):
            imgcap=image
            if (cv2.waitKey(1) & 0xFF) == ord('w'):
                break
        
    # Show the image
    cv2.imshow("Output",imgcap )

    

while True:
        cv2.imshow("CapOutput",imgcap )
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

cv2.destroyAllWindows()
cap.release()
