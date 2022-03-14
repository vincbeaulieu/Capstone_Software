from sklearn.model_selection import train_test_split

import ml.knn as ml_algorithm
from myoband.MyoBandData import read_myoband_data, get_myoband_data
from sklearn.preprocessing import StandardScaler
import multiprocessing
import time


q1 = multiprocessing.Queue()
q2 = multiprocessing.Queue()

sc = StandardScaler()


def myo_predict(model):
    ml_algorithm.train_model(model)

    print("Starting myoband")
    p = multiprocessing.Process(target=read_myoband_data, args=(q1, q2,))
    try:
        p.start()
        time.sleep(5)
        while True:
            input("Press enter to start")
            emg1, emg2 = get_myoband_data(q1, q2)
            emg_data = [emg1 + emg2]
            predicted = get_prediction(emg_data, sc, model)
            print(predicted)
            # motion(predicted)

    except KeyboardInterrupt:
        p.terminate()
        p.join()
