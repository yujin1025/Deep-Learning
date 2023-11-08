# Importing necessary libraries
import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from PIL import Image
from mnist_data import *
from KNNclass import KNN

#load data
x_train, y_train, x_test, y_test = loadData()

# Normalize the data (성능 향상을 위해)
x_train = x_train / 255.0
x_test = x_test / 255.0

# Data Visualization
image = x_train[0]
label = y_train[0]

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img)) 
    pil_img.show() 
    # image를 unsigned int로

image = image.reshape(28,28) 

'''
1. KNN 784개 input 그대로 사용
'''
clf1 = KNN(x_train, y_train, x_test, y_test)
clf1.set_k(5)
clf1.run()

'''
2. 최적의 K값 도출
'''
k_values = range(1, 11)  # K값 범위 설정 (1부터 10까지 테스트)

best_accuracy = 0.0
best_k = None

x_test_subset = x_test[:100]  # 처음 100개의 테스트 데이터
y_test_subset = y_test[:100]  # 위에서 선택한 데이터에 해당하는 레이블

for k in k_values:
    clf = KNN(x_train, y_train, x_test, y_test)
    clf.set_k(k)  # K값 설정
    y_pred = clf.predict(x_test_subset)
    correct = sum(1 for target, pred in zip(y_test_subset, y_pred) if target == pred)
    accuracy = correct / len(y_test_subset)  # 정확도 측정

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

print(f"Best K: {best_k}, Best Accuracy: {best_accuracy:.2f}")

