import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from BaggingForest import BaggingForest
from RandomForest import RandomForest
from show import show


def decisionTree_factory(n_tree):
    base_tree = DecisionTreeClassifier(
        criterion='gini',
        max_depth=3,
        min_samples_split=5
    )
    return base_tree


if __name__ == '__main__':
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

    fig, ax = plt.subplots()
    ax.set_ylabel('Score')
    ax.set_xlabel('Number of tree')

    show(
        lambda n_tree: BaggingForest(n_tree=n_tree),
        'Bagging Forest',
        X_train, X_test, y_train, y_test,
        ax,
        'blue',
        '-'
    )
    show(
        lambda n_tree: RandomForest(n_tree=n_tree),
        'Random Forest',
        X_train, X_test, y_train, y_test,
        ax,
        'red',
        '-.'
    )
    show(
        lambda n_tree: decisionTree_factory(n_tree=n_tree),
        'Decision Tree',
        X_train, X_test, y_train, y_test,
        ax,
        'green',
        '--'

    )

    ax.legend()
    plt.show()
