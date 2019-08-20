import tkinter as tk
from imutils import face_utils
import dlib
import cv2
import math
from PIL import Image, ImageTk
import numpy as np



#camera to object distance = 23 inch
headToBodyRatio = 7.194
ratio = 9.2/208 #pixel to inch
calibrationValue=207


mainWindow = tk.Tk()
mainWindow.title("Hight from Face")
mainWindow.configure(bg="gray34")
mainWindow.geometry('%dx%d+%d+%d' % (900,600,0,0))
mainWindow.resizable(1000,800)

imgLabel = tk.Label(mainWindow)
imgLabel.place(x=20,y=10)
shrpLabel = tk.Label(mainWindow)
shrpLabel.place(x=460,y=10)



def distance(p1,p2):
        return math.sqrt(math.pow(p1[0]-p2[0],2)+math.pow(p1[1]-p2[1],2))

def getHight(pix):
    h=(pix*headToBodyRatio*ratio)/12
    heightLabel = tk.Label(mainWindow,text="Height : {:06.4f} inch".format(h),bg="green")
    heightLabel.place(x=756,y=400)
    return h


def setDisplay(image,imglbl):
    imgcap=image
    im=cv2.cvtColor(imgcap, cv2.COLOR_BGR2RGB)
    im=cv2.resize(im,(400,300),interpolation=cv2.INTER_AREA)
    im = Image.fromarray(im)
    imgtk = ImageTk.PhotoImage(image=im)
    imglbl.image=imgtk
    imglbl.configure(image=imgtk)

def rotate(image,angle,center):
    rows, cols, ch= image.shape
    print(image.shape)
    M = cv2.getRotationMatrix2D(center, angle*-1, 1.0)
    rotated = cv2.warpAffine(image, M,(rows, cols))
    cv2.imshow("Rotated (Correct)", rotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

cap = cv2.VideoCapture(0)


def click():

    capsize = 0
  
    print("clicked....")
    
    while True:
        # Getting out image by webcam 
        _, image = cap.read()
        #rotate(image,30)
        # Converting the image to gray scale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
          
##        gray_image_mb= cv2.medianBlur(gray,3,0)
##
##        kernel = np.array([[1,1,1],[1,-8,1],[1,1,1]])
##
##        sharpened = cv2.filter2D(gray_image_mb,-1,kernel)
          
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
            
            x1=shape[8][0]
            y1=shape[8][1]
            x2=shape[27][0]
            y2=shape[27][1]
            
            angle = math.atan2(x1-x2,y1-y2)
            angle = angle * 180 /3.14
            print(angle)
            rotate(image,angle,tuple(shape[33].reshape(1, -1)[0]))
            
        if(capsize>calibrationValue):
                setDisplay(image,imgLabel)
                setDisplay(image,shrpLabel)
                break
                
            


startButton = tk.Button(mainWindow,text="Start",width=25,command=click)
endButton = tk.Button(mainWindow,text="End",width=25,command=mainWindow.destroy)

endButton.place(x=500,y=400)
startButton.place(x=60,y=400)
#endButton.pack(side="bottom",  expand="yes", padx="10", pady="10")
#startButton.pack(side="bottom",  expand="yes", padx="10", pady="10")


mainWindow.mainloop()
