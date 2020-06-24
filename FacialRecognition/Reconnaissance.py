import cv2
import numpy as np
import os 
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "formation_visage.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX

id = 0

print("[INFO] Facial recognition : Enter the ID Names, Empty if don't exist.")
name1 = input("Enter the name of the ID 1 >>")
name2 = input("Enter the name of the ID 2 >>")
name3 = input("Enter the name of the ID 3 >>")

names = ['None'] 

names.append(name1)
if name2 != "":
    names.append(name2)

if name3 != "":
    names.append(name3)

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480) 
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)
while True:
    ret, img =cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "Unknown person!"
            confidence = "  {0}%".format(round(100 - confidence))
        
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
    
    cv2.imshow('Recognition',img) 
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break
print("\n [INFO] End of the program")
cam.release()
cv2.destroyAllWindows()