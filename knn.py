import multiprocessing
import time
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from ml_training.MyoBandData import read_myoband_data, get_myoband_data

q1 = multiprocessing.Queue()
q2 = multiprocessing.Queue()
sc = StandardScaler()
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)


# feature extraction and encoder function

def train_classifier():
    dataset = pd.read_csv('csv\gesture.csv')
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # to read
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    # scaling
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    print(y_pred)

    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("accuracy:", accuracy_score(y_test, y_pred))
    return classifier


def get_predicted_movement(emg, scaler, knn_classifier):
    emg_transformed = scaler.transform(emg)
    emg_predicted = knn_classifier.predict(emg_transformed)
    return emg_predicted


if __name__ == "__main__":
    train_classifier()
    print("Starting myoband")
    try:
        p = multiprocessing.Process(target=read_myoband_data, args=(q1, q2, ))
        p.start()
        time.sleep(5)
        while True:
            input("Press enter to start")
            emg1, emg2 = get_myoband_data(q1, q2)
            emg_data = []
            emg_data.append(emg1 + emg2)

    except KeyboardInterrupt:
        p.terminate()
        p.join()
