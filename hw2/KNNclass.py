import numpy as np
import time
'''
KNN 클래스
저번 hw1 과제와 init, run함수빼고 전부 동일
'''
class KNN:
    def __init__(self, x_train, y_train, x_test, y_test):
        self._x_train = x_train
        self._y_train = y_train
        self._x_test = x_test
        self._y_test = y_test
        self.k = None

    def set_k(self, k):
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
        distances = [self.calculate_distance(x, x_train) for x_train in self._x_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest = [self._y_train[i] for i in k_indices]
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

    def run(self):
        if self._x_test is not None:
            x_test_subset = self._x_test[:100]  # 처음 100개의 테스트 데이터
            y_test_subset = self._y_test[:100]  # 실제 레이블

            y_pred = self.predict(x_test_subset, weighted=False)
            correct_unweighted = sum(1 for target, pred in zip(y_test_subset, y_pred) if target == pred)
            accuracy_unweighted = correct_unweighted / len(y_test_subset)

            y_pred_weighted = self.predict(x_test_subset, weighted=True)
            correct_weighted = sum(1 for target, pred in zip(y_test_subset, y_pred_weighted) if target == pred)
            accuracy_weighted = correct_weighted / len(y_test_subset)

            print("Accuracy (unweighted):")
            for target, pred in zip(y_test_subset, y_pred):
                print(f"actual: {target}, predicted: {pred}")

            print("Accuracy (weighted):")
            for target, pred in zip(y_test_subset, y_pred_weighted):
                print(f"actual: {target}, predicted: {pred}")

            print(f"accuracy (unweighted): {accuracy_unweighted:.2f}")
            print(f"accuracy (weighted): {accuracy_weighted:.2f}")

            start_time = time.time()
            y_pred = self.predict(x_test_subset, weighted=False)  # 실행 시간 측정을 위해 다시 예측
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"time: {elapsed_time:.2f} seconds")

        else:
            print("No Test data")

