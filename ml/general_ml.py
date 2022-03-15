from ml.ml_class import train_model, get_prediction, load_model
from myoband.MyoBandData import read_myoband_data, get_myoband_data
from sklearn.preprocessing import StandardScaler
import multiprocessing
import time

from rbpi.servoGestureOutput import motion

q1 = multiprocessing.Queue()
q2 = multiprocessing.Queue()

sc = StandardScaler()


def myo_predict(model):
    train_model(model)

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


if __name__ == "__main__":

    # Import a ML library
    from sklearn.neighbors import KNeighborsClassifier

    # Load a ML model
    knn_model = load_model('../ml/saved_model')

    myo_predict()