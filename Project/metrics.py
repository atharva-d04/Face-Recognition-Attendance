import face_recognition
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Path to the images folder
path = 'ImageAttendance'
images = []
classNames = []
myList = os.listdir(path)

for cl in myList:
    curImg = face_recognition.load_image_file(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

# Function to find encodings for the images
def findEncodings(images):
    encodeList = []
    for img in images:
        encodes = face_recognition.face_encodings(img)
        if encodes:
            encodeList.append(encodes[0])
    return encodeList

# Get the encoded faces from the images
encodeListKnown = findEncodings(images)

testImagesPath = "D:\VIT SEMESTER 7\IOT\Project\TestImages"

# Create ground truth and predicted lists
def evaluateFaceRecognition(testImagesPath, encodeListKnown, classNames, threshold=0.60):
    ground_truth = []
    predictions = []

    test_images_list = os.listdir(testImagesPath)

    for image_file in test_images_list:
        img = face_recognition.load_image_file(f'{testImagesPath}/{image_file}')
        ground_truth_label = os.path.splitext(image_file)[0]  # Extract the ground truth label

        # Find face encodings for the test image
        encodesCurFrame = face_recognition.face_encodings(img)

        if len(encodesCurFrame) > 0:
            encodeFace = encodesCurFrame[0]
            # Compare with known encodings
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

            # Get the best match
            matchIndex = np.argmin(faceDis)

            # Check the match and threshold
            if matches[matchIndex] and faceDis[matchIndex] < threshold:
                predicted_label = classNames[matchIndex]
            else:
                predicted_label = "Unknown"
        else:
            predicted_label = "No Face Detected"

        ground_truth.append(ground_truth_label)
        predictions.append(predicted_label)

    return ground_truth, predictions

# Path to test images for evaluation
testImagesPath = 'TestImages'  # Replace with your path to test images

# Get ground truth and predicted labels
ground_truth, predictions = evaluateFaceRecognition(testImagesPath, encodeListKnown, classNames, threshold=0.60)

# Calculate accuracy, precision, recall, and F1-score
accuracy = accuracy_score(ground_truth, predictions)
precision = precision_score(ground_truth, predictions, average='weighted', zero_division=1)
recall = recall_score(ground_truth, predictions, average='weighted', zero_division=1)
f1 = f1_score(ground_truth, predictions, average='weighted', zero_division=1)

# Print results
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)

# Confusion matrix (optional, for more detailed analysis)
conf_matrix = confusion_matrix(ground_truth, predictions)
print("Confusion Matrix:\n", conf_matrix)
