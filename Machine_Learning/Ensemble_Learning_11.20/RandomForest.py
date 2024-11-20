from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from BaggingForest import BaggingForest
from show import show


class RandomForest(BaggingForest):
    def __init__(self, n_tree=10, max_features='sqrt'):
        super().__init__(n_tree)
        self.trees = [
            DecisionTreeClassifier(
                criterion='gini',
                max_depth=3,
                max_features=max_features,  # 随机特征量
                min_samples_split=5)
            for _ in range(n_tree)
        ]


if __name__ == "__main__":
    X, y = make_classification(
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
    bf = RandomForest(n_tree=3)
    bf.fit(X_train, y_train)
    y_pred = bf.predict(X_test)
    accuracy1 = accuracy_score(y_test, y_pred)
    accuracy2 = bf.score(X_test, y_test)
    print('accuracy1', accuracy1)
    print('accuracy2', accuracy2)

    res = show(lambda n_tree: RandomForest(n_tree=n_tree), 'Random Forest', X_train, X_test, y_train, y_test)
    res.show()
