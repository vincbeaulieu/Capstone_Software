import os
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score

import ml.ml_class
from ml.ml_class import data_extractor, dataset_to_csv
from ml.MLObject import MLObject

from rbpi.gestures import gestures_list

# Save and load location
_save_dir = "ml/saved_model/"

handRemoved = ['handPeace', 'handPinky', 'handRing', 'handFlip', 'handExit']
gestures = [g for g in gestures_list if g not in handRemoved]

v = True  # verbose
def group_gen(group_size, group_qty):
    global gestures
    gestures = gestures_list
    random.shuffle(gestures)
    print(gestures)

    # Remove unwanted gesture
    gestures = [g for g in gestures if g not in handRemoved]

    gestures_groups = []
    tmp_list = []
    group_nb, i = 0, 0
    while group_nb <= group_qty:
        _GREEN_, _END_ = '\033[92m', '\033[0m'
        v and (not(i % group_size) and print(_GREEN_, end=''))

        gesture_index = i % len(gestures)
        gesture = gestures[gesture_index]

        if group_nb * group_size / 2 < i:
            random.shuffle(gestures)

        if i % group_size == 0:
            (i != 0) and gestures_groups.append(tmp_list)
            tmp_list = [gesture]
            group_nb += 1
        else:
            tmp_list.append(gesture)

        v and print('\t' * gesture_index, gesture)
        v and (not(i % group_size) and print(_END_, end=''))

        i += 1

    print(gestures_groups)
    return gestures_groups

def ml_gen(_data_values, _data_keys, group_size, ml_qty):
    gestures_groups = group_gen(group_size, group_qty=ml_qty)

    ml_groups, ml_objects = [], []
    for k in range(ml_qty):
        ml_groups.append(list())

    for i, key in enumerate(_data_keys):
        gesture_data = np.concatenate((_data_values[i], key), axis=None)
        for ml_nb in range(ml_qty):
            if key in gestures_groups[ml_nb]:
                ml_groups[ml_nb].append(gesture_data)

    for j in range(ml_qty):
        save_path = "many_ml/ml_" + str(j+1)
        dataset_to_csv(ml_groups[j], save_path)
        ml_objects.append(MLObject(save_path, _save_dir + save_path + "/dataset.csv").train().evaluate().save())

    return ml_objects, ml_groups


def predict(ml_objects, in_data, ml_qty):
    _pred, _conf = [], []
    for j in range(ml_qty):
        tmp_pred, tmp_conf = ml_objects[j].predict(in_data)
        _pred.append(tmp_pred)
        _conf.append(tmp_conf)

    # FIXME: Will lead to false positive when one of the ml lack gesture diversity
    # TODO: Could be resolved by calculating the sum of all conf for each gesture and using the highest sum
    # However, each conf gesture must all have the same number of input, and all gestures must be represented equally
    k_winner, best_conf = 0, 0
    for k in range(ml_qty):
        max_conf = max(_conf[k][0])
        if max_conf > best_conf:
            best_conf = max_conf
            k_winner = k
    return _pred[k_winner]


def data_remover(_data_values, _data_keys):
    values_temp, keys_temp = [], []
    for i, data_key in enumerate(_data_keys):
        if not (data_key in handRemoved):
            keys_temp.append(data_key)
            values_temp.append(_data_values[i])
    return values_temp, keys_temp


def load(ml_qty):
    ml_objects = []

    for j in range(ml_qty):
        save_path = "many_ml/ml_" + str(j+1)
        ml_objects.append(MLObject(save_path, _save_dir + save_path + "/dataset.csv").load(save_path))

    return ml_objects



def cpu_limit():
    # For some reason, limiting Python to a single thread improve speed by a lot
    os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=4
    os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=4
    os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=6
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=4
    os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=6
    os.environ["BLIS_NUM_THREADS"] = "1"
    os.environ["NUMBER_OF_PROCESSORS"] = "1"


def initialize(dataset_path, model_size, model_qty):
    # Format and cleanup dataset
    _data_values, _data_keys = data_extractor(dataset_path)
    _data_values, _data_keys = data_remover(_data_values, _data_keys)

    # Creating many ML models
    ml_objects, ml_groups = ml_gen(_data_values, _data_keys, group_size=model_size, ml_qty=model_qty)

    return ml_objects


def launch():
    cpu_limit()

    dataset_path = "../csv/dataset.csv"

    # INITIALIZE START #
    # Format and cleanup dataset
    _data_values, _data_keys = data_extractor(dataset_path)
    _data_values, _data_keys = data_remover(_data_values, _data_keys)

    # Creating many ML models
    model_qty = 3  # 3
    model_size = 5  # 5
    ml_objects, ml_groups = ml_gen(_data_values, _data_keys, group_size=model_size, ml_qty=model_qty)
    # INITIALIZE END #

    # Testing the model
    data_len = len(_data_values)
    y_pred = [list()] * data_len
    benchmark = []

    for i in range(data_len):
        start = time.time()

        x_true = [_data_values[i]]
        y_pred[i] = predict(ml_objects, x_true, ml_qty=model_qty)

        end = time.time()
        benchmark.append(end - start)
        average = sum(benchmark) / len(benchmark)
        sys.stdout.write("\r{0} / {1} \t benchmark average = {2} (sec)".format(i, data_len - 1, average))
        sys.stdout.flush()
    sys.stdout.write("\n")

    dataset_to_csv(pd.DataFrame(y_pred), "many_ml/predictions")

    _data_values, y_pred = data_extractor(_save_dir + "many_ml/predictions/dataset.csv")

    # Display confusion matrix
    ConfusionMatrixDisplay.from_predictions(_data_keys[:-1], y_pred)
    Path(_save_dir + "many_ml").mkdir(parents=True, exist_ok=True)
    plt.savefig(_save_dir + "many_ml" + '/confusion_matrix.png')
    plt.show()

    # Printing the model accuracy
    model_accuracy = accuracy_score(_data_keys[:-1], y_pred)
    print("Total accuracy: ", model_accuracy)


# Many ML has very high accuracy
if __name__ == "__main__":
    launch()


