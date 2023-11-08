import numpy as np
from sklearn.datasets import load_iris
from KNNClass import KNNClass

# Iris 데이터 불러오기
iris_data = load_iris()
X = iris_data.data
y = iris_data.target

# 데이터 분할
X_train = []
y_train = []
X_test = []
y_test = []

for i in range(150):
    # test data 
    if i % 15 == 14:
        X_test.append(X[i])
        y_test.append(y[i])
    # train data
    else:
        X_train.append(X[i])
        y_train.append(y[i])

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# K-NN 분류기 생성 및 학습하기
knn = KNNClass(k=5)
knn.fit(X_train, y_train)

# 테스트 데이터에 대한 예측하기
y_pred = knn.predict(X_test, weighted=True)

# calculated output과 true output 출력하기
for i, (computed_class, true_class) in enumerate(zip(y_pred, y_test)):
    computed_class_name = iris_data.target_names[computed_class]
    true_class_name = iris_data.target_names[true_class]
    print(f"TestData Index: {i}, Computed Class: {computed_class_name}, True Class: {true_class_name}")
