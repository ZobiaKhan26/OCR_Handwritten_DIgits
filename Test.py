import cv2
import numpy as np
from tensorflow import keras

#####################################
width = 640
height = 480
threshold = 0.65
#####################################

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

cap.set(3, width)
cap.set(4, height)

model = keras.models.load_model("my_model")


def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img


while True:
    ret, imgOriginal = cap.read()
    img = np.asarray(imgOriginal)
    frame = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    img = cv2.resize(img, (32, 32))
    img = preProcessing(img)
    cv2.imshow("Process image", img)
    img = img.reshape(1, 32, 32, 1)
    # Predict
    classIndex = int(model.predict_classes(img))
    # print(classIndex)

    predictions = model.predict(img)
    # print(predictions)
    probVal = np.amax(predictions)
    print(classIndex, probVal)

    if probVal > threshold:
        cv2.putText(imgOriginal, "Class: " + str(classIndex) + " " + "Accuracy: " + str(probVal),
                    (50, 50), cv2.FONT_HERSHEY_COMPLEX,
                    1, (0, 0, 255), 1)
    cv2.imshow("Scan", imgOriginal)

    c = cv2.waitKey(1)
    if c == 27:
        break
