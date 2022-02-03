import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# feature extraction and encoder function

dataset = pd.read_csv('csv\\2_test_emg.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# to read
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)



# print(classifier.predict(sc.transform([[30,87000]])))
y_pred = classifier.predict(X_test)
print(y_pred)
# print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print( "accuracy:", accuracy_score(y_test, y_pred))