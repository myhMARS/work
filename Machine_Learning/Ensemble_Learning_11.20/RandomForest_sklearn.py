from sklearn.datasets import make_classification
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

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

    bc = BaggingClassifier(n_estimators=100, random_state=0)
    bc.fit(X_train, y_train)
    y_pred = bc.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('bagging:', accuracy)

    rfc = RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=0)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('random:', accuracy)
