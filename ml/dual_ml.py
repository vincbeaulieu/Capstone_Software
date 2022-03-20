import os
import sys
import time
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split

from ml.ml_class import train_model, save_model, get_prediction, data_extractor, dataset_to_csv
from MLObject import MLObject

from rbpi.gestures import gestures_positions, gestures_list

handRemoved = ['handPeace', 'handRock', 'handOk', 'handFlip', 'handExit']


# Save and load location
_save_dir = "saved_model/"

def data_divider(_data_values, _data_keys):
    # Reformat the dataset into 2 complementary sets
    ml_1, ml_2, ml_3 = [], [], []
    gestures = gestures_list
    gestures = [g for g in gestures if g not in handRemoved]
    print(gestures)
    #random.shuffle(gestures)
    group_1 = gestures[0:4] + ['handPinky']
    group_2 = gestures[2:6]
    group_3 = gestures[0:2] + gestures[4:7]
    print(group_1)
    print(group_2)
    print(group_3)
    for index, key in enumerate(_data_keys):
        assigned = np.concatenate((_data_values[index], key), axis=None)
        if key in group_1:
            # print("group_1: ", key)
            ml_1.append(assigned)
        if key in group_2:
            # print("\tgroup_2: ", key)
            ml_2.append(assigned)
        if key in group_3:
            # print("\t\tgroup_3: ", key)
            ml_3.append(assigned)

    dataset_to_csv(ml_1, "dual_ml/ml_1")
    dataset_to_csv(ml_2, "dual_ml/ml_2")
    dataset_to_csv(ml_3, "dual_ml/ml_3")

    return pd.DataFrame(ml_1), pd.DataFrame(ml_2), pd.DataFrame(ml_3)


def data_remover(_data_values, _data_keys):
    values_temp, keys_temp = [], []
    for i, data_key in enumerate(_data_keys):
        if not(data_key in handRemoved):
            keys_temp.append(data_key)
            values_temp.append(_data_values[i])
    return values_temp, keys_temp


# Dual ML has 80% accuracy
if __name__ == "__main__":
    # Split the original dataset
    dataset_path = "../csv/suyashretry.csv"

    _data_values, _data_keys = data_extractor(dataset_path)
    _data_values, _data_keys = data_remover(_data_values, _data_keys)
    ml_1, ml_2, ml_3 = data_divider(_data_values, _data_keys)

    # Creating many ML models
    ml_obj_1 = MLObject("dual_ml/ml_1", _save_dir + "dual_ml/ml_1/dataset.csv").train().evaluate().save()
    ml_obj_2 = MLObject("dual_ml/ml_2", _save_dir + "dual_ml/ml_2/dataset.csv").train().evaluate().save()
    ml_obj_3 = MLObject("dual_ml/ml_3", _save_dir + "dual_ml/ml_3/dataset.csv").train().evaluate().save()

    # Testing the model
    data_len = len(_data_values)
    y_pred = [None]*data_len
    benchmark = []

    if True:  # for debugging purposes
        for i in range(data_len):
            start = time.time()

            x_true = [_data_values[i]]
            pred_1, conf_1 = ml_obj_1.predict(x_true)
            pred_2, conf_2 = ml_obj_2.predict(x_true)
            pred_3, conf_3 = ml_obj_3.predict(x_true)

            if max(conf_1[0]) > max(conf_2[0]):
                y_pred[i] = pred_1
            elif max(conf_2[0]) > max(conf_3[0]):
                y_pred[i] = pred_2
            else:
                y_pred[i] = pred_3

            # print(x_true, y_pred[i])
            # dataset_to_csv(pd.DataFrame(y_pred), "dual_ml/predictions")

            end = time.time()
            benchmark.append(end - start)
            average = sum(benchmark) / len(benchmark)
            sys.stdout.write("\r{0} / {1} \t benchmark average = {2} (sec)".format(i, data_len - 1, average))
            sys.stdout.flush()
        sys.stdout.write("\n")

        dataset_to_csv(pd.DataFrame(y_pred), "dual_ml/predictions")

    _data_values, y_pred = data_extractor(_save_dir + "dual_ml/predictions/dataset.csv")

    # Display confusion matrix
    ConfusionMatrixDisplay.from_predictions(_data_keys[:-1], y_pred)
    Path(_save_dir + "dual_ml").mkdir(parents=True, exist_ok=True)
    plt.savefig(_save_dir + "dual_ml" + '/confusion_matrix.png')
    plt.show()

    # Printing the model accuracy
    model_accuracy = accuracy_score(_data_keys[:-1], y_pred)
    print("Total accuracy: ", model_accuracy)
