import time

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import pandas as pd


import ml.general_ml as gml

classifier = MLPClassifier(max_iter=2000)
knn_filename = 'ML/saved_model'


def train_classifier(file_path):
    print("Starting NN classifier training...")
    dataset = pd.read_csv(file_path)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # to read
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

    # # scaling
    # X_train = sc.fit_transform(X_train)
    # X_test = sc.transform(X_test)

    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
#    print(y_pred)

    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("accuracy:", accuracy_score(y_test, y_pred))
    gml.save_model(classifier, knn_filename)
    print("NN Classifier training complete.")
    return classifier


def get_predicted_movement(emg, nn_classifier):
    emg_predicted = nn_classifier.predict(emg)
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
            emg_data = [emg1 + emg2]
            predicted = get_predicted_movement(emg_data, sc, classifier)
            print(predicted)
            #motion(predicted)

    except KeyboardInterrupt:
        p.terminate()
        p.join()
