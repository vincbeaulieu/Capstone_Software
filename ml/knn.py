from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from ml_training.MyoBandData import read_myoband_data, get_myoband_data
import pandas as pd
import pickle as pk
import multiprocessing
import time
#from servoGestureOutput import motion

q1 = multiprocessing.Queue()
q2 = multiprocessing.Queue()
sc = StandardScaler()
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knn_filename = 'ML/saved_model'


def save_model(model, file_name):
    with open(file_name, 'wb') as knn_file:
        pk.dump(model, knn_file)


def load_model(file_name):
    return pk.load(open(file_name, 'rb'))


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
    save_model(classifier, knn_filename)
    print("KNN Classifier training complete.")
    return classifier, sc


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
            predicted = get_predicted_movement(emg_data, sc, classifier)
            print(predicted)
            #motion(predicted)

    except KeyboardInterrupt:
        p.terminate()
        p.join()