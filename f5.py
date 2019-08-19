import tkinter as tk
from imutils import face_utils
import dlib
import cv2
import math
from PIL import Image, ImageTk
import threading



#camera to object distance = 23 inch
headToBodyRatio = 7.194
ratio = 9.2/208 #pixel to inch
calibrationValue=207


mainWindow = tk.Tk()
mainWindow.title("Hight from Face")
mainWindow.configure(bg="#adc5ed")
mainWindow.geometry('%dx%d+%d+%d' % (800,600,0,0))
mainWindow.resizable(1000,800)

imgLabel = tk.Label(mainWindow)
imgLabel.pack(side="top",padx=10,pady=10)



def distance(p1,p2):
        return math.sqrt(math.pow(p1[0]-p2[0],2)+math.pow(p1[1]-p2[1],2))

def getHight(pix):
    return (pix*headToBodyRatio*ratio)/12

def setDisplay(image):
    imgcap=image
    im=cv2.cvtColor(imgcap, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(im)
    imgtk = ImageTk.PhotoImage(image=im)
    imgLabel.image=imgtk
    imgLabel.configure(image=imgtk)


p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

cap = cv2.VideoCapture(0)

capsize = None
def click():
    
    print("clicked....")
    
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
            #size (hight) the face
            capsize=(2*distance(tuple(shape[8].reshape(1, -1)[0]),tuple(shape[27].reshape(1, -1)[0])))
            print(capsize)
            print("Hight :",getHight(capsize))
           

        if(capsize>calibrationValue):
                setDisplay(image)
                break
                
            


startButton = tk.Button(mainWindow,text="Start",width=25,command=click)
endButton = tk.Button(mainWindow,text="End",width=25,command=mainWindow.destroy)

endButton.pack(side="bottom",  expand="yes", padx="10", pady="10")
startButton.pack(side="bottom",  expand="yes", padx="10", pady="10")


mainWindow.mainloop()
