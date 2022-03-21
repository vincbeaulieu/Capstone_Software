import sys
import time
from copy import deepcopy
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score

from ml.ml_class import train_model, save_model, get_prediction
from rbpi.gestures import handRemoved, gestures


def ml_object(model_name, dataset_name=None, dataset_path=None):
    # Import and create a ML model
    from sklearn.ensemble import BaggingClassifier, HistGradientBoostingClassifier
    # ml_model = BaggingClassifier(HistGradientBoostingClassifier())  # 80% - Stupid slow
    ml_model = HistGradientBoostingClassifier()  # 77% - Not very fast (need to be tested on the rbpi)

    # Train the ML model
    ml_model, ml_scaler, ml_dataset = train_model(ml_model, dataset_name, dataset_path)

    # Save the ML model
    save_model(ml_model, ml_scaler, model_name)

    return ml_model, ml_scaler, ml_dataset


def data_extractor(name, path="../csv/"):
    # Extracting data from csv
    dataset = pd.read_csv(path + name)
    values = dataset.iloc[:, :-1].values
    keys = dataset.iloc[:, -1].values
    return keys, values


def data_divider(source_name="dataset.csv", destination_path="saved_model/datasets/"):
    # Creating the directory if it doesn't exist
    Path(destination_path).mkdir(parents=True, exist_ok=True)

    # gestures = list(gestures_positions.keys())

    # Extracting data from csv
    keys, values = data_extractor(source_name)

    ml_1 = []
    ml_2 = []

    # Reformat the dataset into 2 complementary sets
    part = int(len(gestures) / 2)
    for index, key in enumerate(keys):
        # ','.join(map(str, values[index])) + ',' + 'handUnknown'
        assigned = np.concatenate((values[index], key), axis=None)
        if key in handRemoved:
            pass  # Do nothing
        elif key in gestures[0:part]:
            ml_1.append(assigned)
        else:
            ml_2.append(assigned)

    # Save reformatted dataset to csv
    def dataset_to_csv(datalist, path, name):
        pathname = path + name
        df = pd.DataFrame(datalist)
        df.to_csv(pathname, index=False, header=False, mode='w')

    dataset_to_csv(ml_1, destination_path, "ml_1")
    dataset_to_csv(ml_2, destination_path, "ml_2")


# Dual ML as 80% accuracy
if __name__ == "__main__":
    data_path = "saved_model/dual_ml/datasets/"
    data_name = "suyash10gpieday.csv"

    data_divider(data_name, data_path)

    # Creating many ML models
    model_1, scaler_1, dataset_1 = ml_object("dual_ml/ml_1", dataset_name="ml_1", dataset_path=data_path)
    model_2, scaler_2, dataset_2 = ml_object("dual_ml/ml_2", dataset_name="ml_2", dataset_path=data_path)

    # Getting testing data
    data_values = np.concatenate((dataset_1[1], dataset_2[1]), axis=0)
    data_keys = np.concatenate((dataset_1[3], dataset_2[3]), axis=0)

    # data_keys, data_values = data_extractor(data_name)
    keys_temp = []
    values_temp = []
    for i, data_key in enumerate(data_keys):
        if data_key in handRemoved:
            pass
        else:
            keys_temp.append(data_keys[i])
            values_temp.append(data_values[i])
    data_keys = keys_temp
    data_values = values_temp

    # Testing the model
    data_len = len(data_values)
    y_pred = [int] * data_len

    benchmark = []

    for i in range(data_len):
        start = time.time()

        data_value = [data_values[i]]
        pred_1, conf_1 = get_prediction(data_value, model_1, scaler_1)
        pred_2, conf_2 = get_prediction(data_value, model_2, scaler_2)

        if max(conf_1[0]) > max(conf_2[0]):
            y_pred[i] = pred_1
        else:
            y_pred[i] = pred_2

        end = time.time()
        benchmark.append(end - start)
        sys.stdout.write("\r{0} / {1} \t accuracy: {2} ".format(i, data_len - 1, sum(benchmark)/len(benchmark)))
        sys.stdout.flush()

    sys.stdout.write("\n")

    y_pred = np.array(y_pred)

    ConfusionMatrixDisplay.from_predictions(data_keys, y_pred)
    plt.show()

    # Printing the model accuracy
    model_accuracy = accuracy_score(data_keys, y_pred)
    print("Accuracy: ", model_accuracy)
