import pandas as pd
from sklearn.model_selection import train_test_split

from perceptron import Perceptron

df = pd.read_csv("titanic_perceptron.csv")
df.info()

print(df.head())

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1}) # pyright: ignore
df.fillna({'Age': df.Age.mean()}, inplace=True)

titanic = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']]
X = titanic.iloc[:, 0:6]
y = titanic.iloc[:,6:7]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train, y_train = X_train.to_numpy(), y_train.to_numpy()
X_test, y_test = X_test.to_numpy(), y_test.to_numpy()

perceptron = Perceptron()
perceptron.fit(X_train, y_train)
num, currect_num = 0, 0
for x_i, y_i in zip(X_test, y_test):
    predicted_y = perceptron.predict(x_i)
    num +=1
    if y_i == predicted_y:
        currect_num += 1
print(currect_num/num)

