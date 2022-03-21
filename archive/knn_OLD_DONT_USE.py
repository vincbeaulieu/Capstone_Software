from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from MyoBandData import read_myoband_data, get_myoband_data
from pathlib import Path
import multiprocessing
import time
import pandas as pd
import pickle as pk

q = multiprocessing.Queue()
sc = StandardScaler()
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)

csv_filename = 'gesture'
dir_name = 'csv'
suffix = '.csv'

knn_filename = 'saved_model'

# feature extraction and encoder function

def train_classifier():
    path = Path(dir_name, csv_filename).with_suffix(suffix)
    print(path)
    dataset = pd.read_csv(path)
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

    # Save model
    save_model(classifier, knn_filename)

def save_model(model, file_name):
    with open(file_name, 'wb') as knn_file:
        pk.dump(model, knn_file)

def load_model(file_name):
    return pk.load(open(file_name, 'rb'))

if __name__ == "__main__":
    train_classifier()
    print("Starting myoband")
    try:
        p = multiprocessing.Process(target=read_myoband_data, args=(q,))
        p.start()
        time.sleep(5)
        while True:
            input("Press enter to start")
            emg = get_myoband_data(q)
            # just doubling it up for now, will change when have 2 myobands going
            emg = emg + emg
            emg_data = []
            emg_data.append(emg)
            emg_transformed = sc.transform(emg_data)
            emg_predicted = classifier.predict(emg_transformed)
            print(emg_predicted)

    except KeyboardInterrupt:
        p.terminate()
        p.join()
