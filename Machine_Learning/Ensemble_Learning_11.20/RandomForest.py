import numpy as np
from BaggingForest import BaggingForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

class RandomForest(BaggingForest):
    def __init__(self, n_tree=10, max_features='sqrt'):
        super().__init__(n_tree)
        self.trees = [
            DecisionTreeClassifier(
            criterion='gini',
            max_depth=3,
            max_features=max_features,
            min_samples_split=5)
            for _ in range(n_tree)
        ]
 
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
    bf = RandomForest(n_tree=3)
    bf.fit(X_train, y_train)
    y_pred = bf.predict(X_test)
    accuracy1 = accuracy_score(y_test, y_pred)
    accuracy2 = bf.score(X_test, y_test)
    print('accuracy1', accuracy1)
    print('accuracy2', accuracy2)

    num_trees = np.arange(1, 101, 5)
    np.random.seed(0)
    plt.figure
    
    test_score = []
    train_score = []

    with tqdm(num_trees) as pbar:
        for n_tree in pbar:
            bf = RandomForest(n_tree=n_tree)
            bf.fit(X_train, y_train)
            train_score.append(bf.score(X_train, y_train))
            bf.fit(X_test, y_test)
            test_score.append(bf.score(X_test, y_test))
            pbar.set_postfix({
                'n_tree': n_tree,
                'train_score': train_score[-1],
                'test_score': test_score[-1]
            })
    plt.plot(
        num_trees,
        train_score,
        color='blue',
        label = 'random_train_score'
        )
    plt.plot(
        num_trees,
        test_score,
        color='blue',
        linestyle='-.',
        label='random_test_score'
        )
    plt.ylabel = 'Score'
    plt.xlabel = 'Number of tree'
    plt.legend()
    plt.show()

