# import
import numpy as np
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd

# import csv
dataset = pd.read_csv(open("data/datasett.csv", "r"))
print(dataset)
# labeling
le = LabelEncoder()
#
for column in dataset:
    if dataset[column].dtypes == object:
        dataset[column] = le.fit_transform(dataset[column])
print(dataset)
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=10, random_state=0)
print(X_train, X_test, y_train, y_test)

model = tree.DecisionTreeClassifier(
    random_state=0,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0,
    max_leaf_nodes=None,
    min_impurity_decrease=0,
)

clf = model.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

Xtes = np.array([2, 2, 2])

print(Xtes)

hasil_prediksi = clf.predict([Xtes])

print(hasil_prediksi)
