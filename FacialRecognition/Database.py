import cv2
import os
import numpy as np
from PIL import Image

cascade = cv2.CascadeClassifier("formation_visage.xml")

cam = cv2.VideoCapture(0)

id = input("Enter an ID for this person >> ")
print("[INFO] Scan of the face")

count = 0

while True:
    _, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in face:
        count += 1
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.imwrite("dataset/User." + str(id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
    cv2.imshow("Reconnaissance", img)
    k = cv2.waitKey(100) & 0xFF
    if k == 27:
        break
    elif count == 30:
        break

path = 'dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("formation_visage.xml");

def entrainement(path):

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []

    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L') 
        img_numpy = np.array(PIL_img,'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples,ids

print ("\n [INFO] Training of the person face")
faces,ids = entrainement(path)
recognizer.train(faces, np.array(ids))

recognizer.write('trainer/trainer.yml') 


print("\n [INFO] {0} Training of the data base \n\n This person is now detectable".format(len(np.unique(ids))))
