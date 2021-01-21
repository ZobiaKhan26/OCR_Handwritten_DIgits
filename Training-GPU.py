import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.layers.convolutional import Conv2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.python.keras.layers import MaxPooling2D, Dropout, Flatten


import time

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.test.is_gpu_available()


start_time = time.time()


#########################################
path = 'myData'
pathLabels = 'labels.csv'
testRatio = 0.2
valRatio = 0.2
imageDimensions = (32, 32, 3)
#########################################
count = 0
images = []
classNo = []
myList = os.listdir(path)
print("Total number of classes detected: ", len(myList))
noOfClasses = len(myList)
print("Importing Classes...")
for x in range(0, noOfClasses):
    myPicList = os.listdir(path + "/" + str(count))
    # print(myPicList)
    for y in myPicList:
        curImg = cv2.imread(path + "/" + str(count) + "/" + y)
        curImg = cv2.resize(curImg, (imageDimensions[0], imageDimensions[1]))
        images.append(curImg)
        classNo.append(count)
    print(count, end="  ")
    count = count + 1
print("  ")
print("Total images in the image list: ", len(images))
print("Total IDs in classNo list: ", len(classNo))
images = np.array(images)
classNo = np.array(classNo)

print("Image Shape Before Splitting", images.shape)
# print(classNo.shape)

# Splitting the Data

X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=valRatio)
print("Image TrainSet Shape After Splitting", X_train.shape)
print("Image TestSet Shape After Splitting", X_test.shape)
print("Image ValidationSet Shape After Splitting", X_validation.shape)

noOfSamples = []
for x in range(0, noOfClasses):
    # print(len(np.where(y_train == x)[0]))
    noOfSamples.append(len(np.where(y_train == x)[0]))
print("Number of samples in each class: ", noOfSamples)

plt.figure(figsize=(10, 5))
plt.bar(range(0, noOfClasses), noOfSamples)
plt.title("No of images in each class")
plt.xlabel("Class ID")
plt.ylabel("No of Images")
# plt.show()


def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img


# img = preProcessing(X_train[30])
# img = cv2.resize(img, (300, 300))
# cv2.imshow("PreProcessed ", img)
# cv2.waitKey(0)

# Pre Processing images to grayscale

X_train = np.array(list(map(preProcessing, X_train)))
X_test = np.array(list(map(preProcessing, X_test)))
X_validation = np.array(list(map(preProcessing, X_validation)))

# Reshaping images to give a depth of 1
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))
X_validation = X_validation.reshape((X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1))

# Augmenting Data
dataGen = ImageDataGenerator(width_shift_range=0.1,  # 10%
                             height_shift_range=0.1,  # 10%
                             zoom_range=0.2,  # 20% zoom in and out
                             shear_range=0.1,  # 10%
                             rotation_range=10)  # 10 degrees
dataGen.fit(X_train)  # generate the augmented image at the time of training

# One Hot encoding step

y_train = to_categorical(y_train, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)


# Creating Model

def myModel():
    noOfFilters = 60
    sizeOfFilter1 = (5, 5)
    sizeOfFilter2 = (3, 3)
    sizeOfPool = (2, 2)
    noOfNode = 500

    model = tf.keras.Sequential()
    model.add((Conv2D(noOfFilters, sizeOfFilter1, input_shape=(imageDimensions[0],
                                                               imageDimensions[1],
                                                               1), activation='relu')))

    model.add((Conv2D(noOfFilters, sizeOfFilter1, activation='relu')))  # first convolution layer
    model.add(MaxPooling2D(pool_size=sizeOfPool))  # pooling layer 1

    model.add((Conv2D(noOfFilters // 2, sizeOfFilter2,
                      activation='relu')))  # second convolution layer, decreased to half, floor division
    model.add(
        (Conv2D(noOfFilters // 2, sizeOfFilter2, activation='relu')))  # third convolution layer, decreased to half

    # the floor division // rounds the result down to the nearest whole number

    model.add(MaxPooling2D(pool_size=sizeOfPool))  # pooling layer 2

    # Adding our first dropout layer
    model.add(Dropout(0.5))  # 50% to reduce over fitting
    model.add(Flatten())

    model.add(tf.keras.layers.Dense(noOfNode, activation="relu"))

    model.add(Dropout(0.5))  # 50% to reduce over fitting
    model.add(tf.keras.layers.Dense(noOfClasses, activation='softmax'))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model


model = myModel()
print(model.summary())

batchSizeVal = 50
epochsVal = 10  # Change epochs val here
stepsPerEpochVal = len(X_train) // batchSizeVal

history = model.fit(dataGen.flow(X_train, y_train,
                                 batch_size=batchSizeVal),
                    steps_per_epoch=stepsPerEpochVal,
                    epochs=epochsVal,
                    validation_data=(X_validation, y_validation),
                    shuffle=1)

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('loss')
plt.xlabel('epoch')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('epoch')

# plt.show()

score = model.evaluate(X_test, y_test, verbose=0)
print("Test Score = ", score[0])
print("Test Accuracy = ", score[1])


model.save("my_model")

