import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from mnist_data import loadData
from LogisticRegressionClass import LogisticRegression
from LogisticRegressionClass import MultiClassLogisticRegression
import pandas as pd

# Load data
x_train, y_train, x_test, y_test = loadData()

# Normalize the data
x_train = x_train / 255.0
x_test = x_test / 255.0

'''
single class 구현 내용
single class랑 multi class를 구별하기 위해 single class 주석처리함 
주석 없애면 single class 구현됨
'''

'''
# target 설정 (현재는 target class 0)
target_class = 0

# logistic regression model
target0 = LogisticRegression(x_train, y_train, target_class)

epoch = 100 # epoch 지정
cost_0 = []
for i in range(epoch):
    cost = target0.cost()
    cost_0.append(cost)
    target0.learn(lr=0.01)
    print(f'epoch: {i}, cost: {cost}') # epoch, cost 출력

# predictions 하고 accuracy 계산
preds = target0.predict(x_test)

# 1D이면 2D로 전환
if len(y_test.shape) == 1:
    y_test = y_test.reshape(-1, 1)

accuracy = np.sum(preds == (y_test[:, target_class] == 1)) / len(preds)
print(f'Accuracy: {accuracy}')

# 비용 그래프
plt.plot(np.arange(0, epoch), cost_0)
plt.xlabel('number of iterations')
plt.ylabel('cost')
plt.show()

'''


'''
MultiClass 
'''
# Initialize MultiClassLogisticRegression model
num_classes = 10  # MNIST 데이터 세트의 경우 10개의 클래스
lr_model = MultiClassLogisticRegression(x_train, y_train, num_classes)

cost_history = lr_model.learn(lr=0.1, epoch=100)

accuracy = lr_model.calculate_accuracy(x_test, y_test)
print(f'Accuracy: {accuracy}')


# 각 클래스에 대한 비용 그래프 생성
plt.figure(figsize=(12, 6))
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:blue', 'tab:orange', 'tab:purple']  # 클래스마다 다른 색상 지정
max_class_costs = []
for class_num in range(num_classes):
    class_costs = np.array(lr_model.class_costs)[:, class_num]
    max_class_costs.append(max(class_costs))
    plt.plot(range(len(class_costs)), class_costs, label=f'Class {class_num} Cost', color=colors[class_num])
plt.xlabel('number of iterations')
plt.ylabel('cost')
plt.legend()
plt.tight_layout()
plt.show()

