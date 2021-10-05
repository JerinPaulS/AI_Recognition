import numpy as np
import pandas as pd
from sklearn import svm, metrics, datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import glob
import os
import skimage
from sklearn.utils import Bunch
from skimage.io import imread
from skimage.transform import resize
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from pathlib import Path
import joblib

train_dir = Path("/home/jerinpaul/Documents/Git/AI Recognition/fruits-360_dataset/fruits-360/Training")
test_dir = Path("/home/jerinpaul/Documents/Git/AI Recognition/fruits-360_dataset/fruits-360/Test")

Train_Img_Count = len(list(train_dir.glob("*/*.jpg")))
Test_Img_Count = len(list(test_dir.glob("*/*.jpg")))

print(Train_Img_Count)
print(Test_Img_Count)

labels = []
for fruit_dir in glob.glob("/home/jerinpaul/Documents/Git/AI Recognition/fruits-360_dataset/fruits-360/Training/*"):
    labels.append(fruit_dir.split("/")[-1])

try:
    clf = joblib.load("fruitspredict.pkl")
    print("Using trained model")
except:
    fruits = []
    labels = []

    target_fruits = []
    target_labels = []

    for fruit_dir in glob.glob("/home/jerinpaul/Documents/Git/AI Recognition/fruits-360_dataset/fruits-360/Training/*"):
        fruit_label = fruit_dir.split("/")[-1]
        print("Loading " + fruit_label + " into the dataset!")
        for image_dir in glob.glob(os.path.join(fruit_dir, "*.jpg")):
            image = skimage.io.imread(image_dir)
            image_resize = resize(image, (45, 45, 4), mode = 'reflect')
            fruits.append(image_resize.flatten())
            labels.append(fruit_label)

    fruits = np.array(fruits)
    labels = np.array(labels)

    x = fruits
    y = labels

    print('Training data-set loaded!')
    clf = svm.SVC(kernel = 'rbf', C = 1000, gamma = 0.001)
    print('Fitting data-set loaded!')
    print("Building new model")
    clf.fit(x, y)
    joblib.dump(clf, "fruitspredict.pkl")

    print('Model trained!')

'''
for fruit_dir in glob.glob("/home/jerinpaul/Documents/Git/AI Recognition/fruits-360_dataset/fruits-360/Test/*"):
    fruit_label = fruit_dir.split("/")[-1]
    print("Loading " + fruit_label + " into the dataset!")
    for image_dir in glob.glob(os.path.join(fruit_dir, ("*.jpg"))):
        image = skimage.io.imread(image_dir)
        image_resize = resize(image, (45, 45, 4), mode = 'reflect')
        target_fruits.append(image_resize.flatten())
        target_labels.append(fruit_label)

target_fruits = np.array(target_fruits)
target_labels = np.array(target_labels)

target_x = target_fruits
target_y = target_labels

print('Testing data-set loaded!')
'''
image = skimage.io.imread("/home/jerinpaul/Documents/Git/AI Recognition/fruits-360_dataset/fruits-360/Training/Walnut/0_100.jpg")
image_resize = resize(image, (45, 45, 4), mode = 'reflect')
image = np.array(image_resize.flatten())
print(clf.predict([image]))
'''
print('Prediction on testing data over!')

accuracy = accuracy_score(target_y, prediction) * 100
print('Accuracy = ', accuracy)
'''
