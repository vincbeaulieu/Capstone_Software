import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score

from ml.MLObject import MLObject
from ml.data_prep import data_remover, data_extractor, dataset_to_csv
from ml.group_generator import group_gen

from ml.sim_test import cpu_limit

from rbpi.gestures import Gestures, handRemoved, gestures_list

# Global variables and parameters
gestures = gestures_list
excluded_gestures = handRemoved

# Save and load location
_save_dir = "saved_model/"
dataset_path = "../csv/suyash10gpieday.csv"


def ml_gen(_data_values, _data_keys, group_size=4, ml_qty=3):
    gestures_groups = group_gen(gestures, group_size, group_qty=ml_qty)

    print(gestures_groups)

    ml_groups = [list() for _ in range(ml_qty)]
    ml_objects = []

    for i, key in enumerate(_data_keys):
        for ml_nb in range(ml_qty):
            if key in gestures_groups[ml_nb]:
                gesture_data = np.concatenate((_data_values[i], key), axis=None)
                ml_groups[ml_nb].append(gesture_data)

    for j in range(ml_qty):
        model_name = "many_ml/ml_" + str(j + 1)
        save_path = _save_dir + model_name
        dataset_to_csv(ml_groups[j], save_path)

        ml_object = MLObject(model_name, save_path + "/dataset.csv")
        ml_object.ml_gestures = gestures_groups[j]
        ml_objects.append(ml_object.train().evaluate().save())

    return ml_objects, gestures_groups


def predict(ml_objects, in_data, ml_qty=3):
    results = []
    for j in range(ml_qty):
        tmp_pred, tmp_conf = ml_objects[j].predict(in_data)
        results.append([tmp_pred[0], list(tmp_conf[0]), max(tmp_conf[0])])

    k_winner, best_conf = 0, 0
    for k in range(ml_qty):
        max_conf = results[k][2]
        print(results[k][0])
        if max_conf > best_conf:
            best_conf = max_conf
            k_winner = k
    return results[k_winner][0]

    # # gestures = [g for g in gestures_list if g not in handRemoved]
    # conf_scores = [float(0) for _ in range(len(gestures))]
    # conf_counts = [float(0)] * len(gestures)
    #
    # for k in range(ml_qty):
    #     for i, g in enumerate(ml_groups[k]):
    #         # print(ml_groups[k])
    #         # print(_conf[k])
    #         conf_scores[gestures.index(g)] += _conf[k][0][i]
    #         conf_counts[gestures.index(g)] += 1
    # # conf_average = [i / j for i, j in zip(conf_scores, conf_counts)]
    # # pred_index = conf_average.index(max(conf_average))
    # # print(_pred, max(conf_scores))
    # # print(gestures[pred_index])

#    return gestures[pred_index]

    # FIXME: Will lead to false positive when one of the ml lack gesture diversity
    # TODO: Could be resolved by calculating the sum of all conf for each gesture and using the highest sum
    # However, each conf gesture must all have the same number of input, and all gestures must be represented equally
    # k_winner, best_conf = 0, 0
    # for k in range(ml_qty):
    #     max_conf = max(_conf[k][0])
    #     if max_conf > best_conf:
    #         best_conf = max_conf
    #         k_winner = k
    # return _pred[k_winner]


def update_gestures(_data_keys):
    global gestures, excluded_gestures
    gesture_object = Gestures().update(_data_keys, handRemoved)

    gestures = gesture_object.list()
    excluded_gestures = gesture_object.excluded()

    random.shuffle(gestures)


def launch():
    # Format and cleanup dataset
    _data_values, _data_keys = data_extractor(dataset_path)
    update_gestures(_data_keys)
    _data_values, _data_keys = data_remover(_data_values, _data_keys, excluded_gestures)

    # Creating many ML models
    model_qty = 3  # 3
    model_size = 4  # 5
    ml_objects, ml_groups = ml_gen(_data_values, _data_keys, group_size=model_size, ml_qty=model_qty)

    # Testing the model
    x_test_data = ml_objects[0].dataset[1]
    y_test_data = ml_objects[0].dataset[3]

    print()

    data_len = len(x_test_data)
    y_pred = [list() for _ in range(data_len)]
    benchmark = []
    for i in range(data_len):
        start = time.time()
        print(ml_objects[0].scaler is None)
        y_pred[i] = predict(ml_objects, [x_test_data[i]], ml_qty=model_qty)
        print(y_pred[i], y_test_data[i])
        for k in range(3):
            print(y_test_data[i], y_pred[i])

        end = time.time()
        benchmark.append(end - start)
        average = sum(benchmark) / len(benchmark)
        sys.stdout.write("\r{0} / {1} \t benchmark average = {2} (sec)".format(i, data_len - 1, average))
        sys.stdout.flush()
    sys.stdout.write("\n")
    dataset_to_csv(pd.DataFrame(y_pred), _save_dir + "many_ml/predictions")

    # Display confusion matrix
    _data_values, y_pred = data_extractor(_save_dir + "many_ml/predictions/dataset.csv")
    ConfusionMatrixDisplay.from_predictions(y_test_data[:-1], y_pred)
    Path(_save_dir + "many_ml").mkdir(parents=True, exist_ok=True)
    plt.savefig(_save_dir + "many_ml" + '/confusion_matrix.png')
    plt.show()

    # Printing the model accuracy
    model_accuracy = accuracy_score(y_test_data[:-1], y_pred)
    print("Total accuracy: ", model_accuracy)


# Many ML has very high accuracy
if __name__ == "__main__":
    # cpu_limit()
    launch()
