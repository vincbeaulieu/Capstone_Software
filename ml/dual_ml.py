import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split

from ml.ml_class import train_model, save_model, get_prediction, data_extractor, dataset_to_csv
from MLObject import MLObject

from rbpi.gestures import gestures_positions
handRemoved = ['handPeace', 'handRock']


# Save and load location
_save_dir = "saved_model/"

def data_divider(source_dir='../csv/dataset.csv'):

    gestures = list(gestures_positions.keys())

    # Extracting data from csv
    values, keys = data_extractor(source_dir)

    # Reformat the dataset into 2 complementary sets
    ml_1, ml_2 = [], []
    part = int(len(gestures) / 2)
    for index, key in enumerate(keys):
        assigned = np.concatenate((values[index], key), axis=None)
        if key in handRemoved:
            pass  # Do nothing
        elif key in gestures[0:part]:
            ml_1.append(assigned)
        else:
            ml_2.append(assigned)

    dataset_to_csv(ml_1, "dual_ml/ml_1")
    dataset_to_csv(ml_2, "dual_ml/ml_2")


def data_remover(dataset_path):
    _data_values, _data_keys = data_extractor(dataset_path)
    values_temp, keys_temp = [], []
    for i, data_key in enumerate(_data_keys):
        if not(data_key in handRemoved):
            keys_temp.append(_data_keys[i])
            values_temp.append(_data_values[i])
    return values_temp, keys_temp


# Dual ML has 80% accuracy
if __name__ == "__main__":

    # Split the original dataset
    dataset_path = "../csv/suyash10gpieday.csv"
    data_divider(dataset_path)

    # Creating many ML models
    ml_obj_1 = MLObject("dual_ml/ml_1", _save_dir + "dual_ml/ml_1/dataset.csv").train().evaluate().save()
    ml_obj_2 = MLObject("dual_ml/ml_2", _save_dir + "dual_ml/ml_2/dataset.csv").train().evaluate().save()

    # Reformat dataset by removing exclusion
    x_test, y_test = data_remover(dataset_path)

    # Testing the model
    data_len = len(x_test)
    y_pred = [int] * data_len
    benchmark = []

    for i in range(data_len):
        start = time.time()

        x_true = [x_test[i]]
        pred_1, conf_1 = ml_obj_1.predict(x_true)
        pred_2, conf_2 = ml_obj_2.predict(x_true)

        if max(conf_1[0]) > max(conf_2[0]):
            y_pred[i] = pred_1
        else:
            y_pred[i] = pred_2

        end = time.time()
        benchmark.append(end - start)
        average = sum(benchmark) / len(benchmark)
        sys.stdout.write("\r{0} / {1} \t benchmark average = {2} (sec)".format(i, data_len - 1, average))
        sys.stdout.flush()

    sys.stdout.write("\n")

    # Display confusion matrix
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.show()

    # Printing the model accuracy
    model_accuracy = accuracy_score(y_test, y_pred)
    print("Total accuracy: ", model_accuracy)
