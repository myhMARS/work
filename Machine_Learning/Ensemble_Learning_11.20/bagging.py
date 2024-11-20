import numpy as np
from sklearn.datasets import make_classification
from sklearn.tree  import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class BaggingForest:
    def __init__(self, n_tree=10):
        self.n_tree = n_tree
        self.trees = [
            DecisionTreeClassifier(
            criterion='gini', 
            max_depth=3, 
            min_samples_split=5) 
            for _ in range(n_tree)
        ]
        self.n_samples = 0
        self.n_features = 0
        self.n_classes = 0


    def bootstrap(self):
        sample_idx = np.random.randint(0, self.n_samples, self.n_samples)
        return sample_idx


    def fit(self, X, y):
        self.n_samples, self.n_features = X.shape
        self.n_classes = np.unique(y).shape[0]

        for tree in self.trees:
            sample_idx =  self.bootstrap()
            tree.fit(X[sample_idx], y[sample_idx])


    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X):
        ensemble = np.mean([tree.predict_proba(X) for tree in self.trees], axis=0)
        return ensemble


    def score(self, X, y):
        return np.mean(y == self.predict(X))


if __name__ == "__main__":
    X,y = make_classification(
        n_samples=1000,  # 数据集大小
        n_features=16,  # 数据特征数
        n_informative=5,  # 有效特征数
        n_redundant=2,  # 冗余特征数
        n_classes=2,  # 数据类别·
        flip_y=0.3,  # 随机类别个数
        random_state=0
    )

    print('X shape', X.shape)
    print('example X', X[0])
    print('example y', y[0])
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)
    bf = BaggingForest(n_tree=3)
    bf.fit(X_train, y_train)
    y_pred = bf.predict(X_test)
    accuracy1 = accuracy_score(y_test, y_pred)
    accuracy2 = bf.score(X_test, y_test)
    print('accuracy1', accuracy1)
    print('accuracy2', accuracy2)
