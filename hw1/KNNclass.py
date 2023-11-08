import numpy as np

# KNN 알고리즘을 구현하는 클래스 정의 
class KNNClass:
    def __init__(self, k=3):
        self.k = k

    # scikit-learn의 fit 메서드는 학습데이터를 받아 모델을 훈련시킨다
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    # 두 데이터간의 Euclidean을 계산한다
    def calculate_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    # 데이터들에 대한 k개의 가장 가까운 것을 찾는다.
    def nearest(self, x):
        distances = [self.calculate_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest = [self.y_train[i] for i in k_indices]
        return k_nearest

    # 가장 가까운 것들 중 가장 많이 나오는 클래스 반환
    def majority_vote(self, k_nearest):
        return np.bincount(k_nearest).argmax()

    # 가중 다수결 투표
    def weighted_majority_vote(self, k_nearest):
        label_weights = np.bincount(k_nearest)
        total_weights = np.sum(label_weights)
        weighted_votes = label_weights / total_weights
        return weighted_votes.argmax()

    # 테스트 데이터에 대한 예측
    def predict(self, X_test, weighted=False):
        y_pred = [self._predict(x, weighted) for x in X_test]
        return np.array(y_pred)

    # 데이터 포인트에 대한 예측 
    def _predict(self, x, weighted=False):
        k_nearest = self.nearest(x)
        if weighted:
            predicted_label = self.weighted_majority_vote(k_nearest)
        else:
            predicted_label = self.majority_vote(k_nearest)
        return predicted_label