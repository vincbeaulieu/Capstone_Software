import pandas as pd

from ml.ml_class import train_model, save_model
from rbpi.gestures import gestures_positions


def knn_object(model_name, dataset_name):
    # Import a ML library
    from sklearn.neighbors import KNeighborsClassifier

    # Create the ML model
    knn_model = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)

    # Train the ML model
    knn_model, knn_scaler = train_model(knn_model, dataset_name)

    # Save the ML model
    save_model(knn_model, knn_scaler, model_name)

    return knn_model, knn_scaler


# Basic Machine Learning Structure:
if __name__ == "__main__":

    dataset_name = "suyash10gpieday.csv"
    gestures = list(gestures_positions.keys())
    # print(gestures[0:3])

    # Extracting data from csv
    dataset = pd.read_csv("../csv/" + dataset_name)
    values = dataset.iloc[:, :-1].values
    keys = dataset.iloc[:, -1].values

    for index, k in enumerate(keys):
        known = []
        unknown = []

        # Todo: concatenate values and k into a single string delimited by a comma
        if k in gestures[0:5]:
            # FixME: This doesn't work
            known.append(','.join(list([values[index], k])))
            unknown.append(','.join(list([values[index], "handUnknown"])))
        else:
            known.append(','.join(list([values[index], "handUnknown"])))
            unknown.append(','.join(list([values[index], k])))

        print(known)

    # Creating two KNN model
    knn_1 = knn_object("knn", dataset_name)
    knn_2 = knn_object("knn", dataset_name)
    # print(knn_1 is knn_2)





