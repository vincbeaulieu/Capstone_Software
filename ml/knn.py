import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import multiprocessing


import general_ml as gml

#from servoGestureOutput import motion

q1 = multiprocessing.Queue()
q2 = multiprocessing.Queue()
sc = StandardScaler()
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)


knn_filename = '../ml/saved_model'


# Trains KNN classifier with the data in file at file_path (csv/<dataset>.csv)
def train_classifier(file_path):
    print("Starting KNN classifier training...")
    dataset = pd.read_csv(file_path)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # to read
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

    # scaling
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
#    print(y_pred)

    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("accuracy:", accuracy_score(y_test, y_pred))
    plot_confusion_matrix(classifier, X_test, y_test)
    plt.show()
    gml.save_model(classifier, knn_filename)
    print("KNN Classifier training complete.")
    return classifier, sc


def get_predicted_movement(emg, scaler, knn_classifier):
    emg_transformed = scaler.transform(emg)
    emg_predicted = knn_classifier.predict(emg_transformed)
    return emg_predicted


if __name__ == "__main__":
    train_classifier("../csv/suyash10gpieday.csv")
