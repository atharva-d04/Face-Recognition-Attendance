import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime, timedelta

path = 'ImageAttendance'
images = []
classNames = []
myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(img)
        if encodes:
            encodeList.append(encodes[0])
    return encodeList

encodeListKnown = findEncodings(images)

# Dictionary to track last exit time for each student
last_exit_time = {}

# Remove the student's name from Attendance.csv and record exit in Exit_sheet.csv
def removeAttendance(name):
    now = datetime.now()

    # Check if the last exit for this student was more than 15 seconds ago
    if name in last_exit_time:
        last_time = last_exit_time[name]
        if (now - last_time).total_seconds() < 15:
#            print(f"Skipping {name}'s exit. Less than 15 seconds since last exit.")
            return

    dtString = now.strftime('%H:%M:%S')
    last_exit_time[name] = now  # Update the last exit time

    # Remove the student from Attendance.csv
    with open('Attendance.csv', 'r+') as f:
        lines = f.readlines()
        f.seek(0)
        for line in lines:
            if line.split(',')[0] != name:
                f.write(line)
        f.truncate()

    # Record the exit in Exit_sheet.csv
    with open('Exit_sheet.csv', 'a') as exit_file:
        exit_file.write(f'{name},{dtString}\n')
    
    print(f'{name} exited at {dtString}.')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex] and faceDis[matchIndex] < 0.50:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            removeAttendance(name)

    cv2.imshow('Webcam', img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
