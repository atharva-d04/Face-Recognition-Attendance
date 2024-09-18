import cv2
import numpy as np
import face_recognition

imgAth = face_recognition.load_image_file("D:\VIT SEMESTER 7\IOT\Project\Dataset\Atharva Training\IMG-20240908-WA0001.jpg")
imgAth = cv2.cvtColor(imgAth,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file("D:\VIT SEMESTER 7\IOT\Project\Dataset\Atharva Training\IMG-20240908-WA0001.jpg")
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)
 
# FOR TRAINING IMAGE
faceLoc = face_recognition.face_locations(imgAth)[0]
encodeAth = face_recognition.face_encodings(imgAth)[0]
cv2.rectangle(imgAth,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

# FOR TESTING IMAGE
faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

# COMPARING THE TRAINING AND TESTING IMAGES
results = face_recognition.compare_faces([encodeAth],encodeTest)
faceDis = face_recognition.face_distance([encodeAth],encodeTest)
print(results,faceDis)
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
 
cv2.imshow('Atharva',imgAth)
cv2.imshow('Atharva Test',imgTest)
cv2.waitKey(0)