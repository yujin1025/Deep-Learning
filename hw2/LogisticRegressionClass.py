import numpy as np

def sigmoid(value):
    value = np.clip(value, -500, 500)
    return 1 / (1 + np.exp(-value))

'''
LogisticRegression 클래스
'''
class LogisticRegression:
    def __init__(self, X, y, target):
        self.X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        self.X = self.X[:, :-1]  # 크기를 (60000, 784)로 수정
        self.y = (y == target).astype(int).reshape(-1, 1)
        self.W = np.random.normal(0.0, 1.0, self.X.shape[1]).reshape(-1, 1)  # 크기를 (785, 1)로 수정

    # cost function
    def cost(self):
        probs = self.predict_proba(self.X)
        batch_size = probs.shape[0]
        return -np.sum(self.y * np.log(probs + 1e-7) + (1 - self.y) * np.log(1 - probs + 1e-7)) / batch_size

    # gradient descent
    def learn(self, lr=0.01):
        probs = self.predict_proba(self.X)
        self.W -= lr * np.sum((probs - self.y) * self.X, axis=0).reshape(-1, 1)
    
    # predict 확률 계산
    def predict_proba(self, inputs):
        hx = sigmoid(np.dot(inputs, self.W))
        return hx

    # binary classification 예측
    def predict(self, inputs):
        probs = self.predict_proba(inputs)
        preds = np.array([1 if x > 0.5 else 0 for x in probs])# 0.5랑 비교해 0 또는 1 예측
        return preds


'''
MultiClassLogisticRegression 클래스
'''
class MultiClassLogisticRegression:
    def __init__(self, input_data, target_output, num_classes):
        self.input_data = self.add_bias(input_data.reshape(-1, 28 * 28).astype('float32'))
        self.target_output = target_output  # 클래스 레이블 (0부터 num_classes - 1까지의 정수)
        self.num_classes = num_classes
        self.weights = np.random.randn(self.input_data.shape[1], num_classes)
        self.class_costs = []  # 각 클래스의 비용을 저장할 리스트

    # 입력 데이터에 bias항 add
    def add_bias(self, X):
        return np.c_[np.ones(X.shape[0]), X]

    # softmax 함수 계산
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    # cost function
    def cost(self, predictions, labels):
        m = len(labels)
        return -1 / m * np.sum(np.log(predictions + 1e-7) * labels)

    # 예측 계산, 오류 계산해 가중치 업데이트
    def learn(self, lr, epoch):
        cost_history = []

        for e in range(epoch):
            predictions = self.softmax(np.dot(self.input_data, self.weights))
            errors = predictions - (self.target_output[:, None] == np.arange(self.num_classes)).astype(int)
            self.weights -= lr * np.dot(self.input_data.T, errors)
            cost = self.cost(predictions, (self.target_output[:, None] == np.arange(self.num_classes)).astype(int))
            cost_history.append(cost) # cost 계산하고 cost_history에 추가
            class_costs = []
            for class_index in range(self.num_classes):
                class_labels = (self.target_output == class_index).astype(int)
                class_cost = self.cost(predictions[:, class_index], class_labels)
                class_costs.append(class_cost)
            print(f'epoch: {e} cost: {class_costs}')

            # 각 클래스별 비용을 저장
            class_costs = []
            for class_index in range(self.num_classes):
                class_labels = (self.target_output == class_index).astype(int)
                class_cost = self.cost(predictions[:, class_index], class_labels)
                class_costs.append(class_cost)
            self.class_costs.append(class_costs)

        return cost_history

    # 주어진 입력에 대한 예측을 수행
    def predict(self, X):
        X_bias = self.add_bias(X.reshape(-1, 28 * 28).astype('float32'))
        predictions = self.softmax(np.dot(X_bias, self.weights))
        return np.argmax(predictions, axis=1)

    #정확도 계산
    def calculate_accuracy(self, X, y):
        predictions = self.predict(X)
        correct = np.sum(predictions == y)
        accuracy = correct / len(y)
        return accuracy

