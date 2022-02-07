import multiprocessing
import time
from sklearn.preprocessing import StandardScaler
from ml_training.MyoBandData import read_myoband_data, get_myoband_data
from knn import train_classifier, get_predicted_movement
from servoGestureOutput import motion

q1 = multiprocessing.Queue()
q2 = multiprocessing.Queue()

def main():
    test()
    return 0

def test():
    classifier,sc = train_classifier()
    print("Starting myoband")
    try:
        p = multiprocessing.Process(target=read_myoband_data, args=(q1, q2, ))
        p.start()
        time.sleep(5)
        while True:
            input("Press enter to start")
            emg1, emg2 = get_myoband_data(q1, q2)
            emg_data = []
            emg_data.append(emg1 + emg2)
            predicted = get_predicted_movement(emg_data, sc, classifier)
            print(predicted)
            motion(predicted[0])

    except KeyboardInterrupt:
        p.terminate()
        p.join()

    pass

if __name__ == '__main__':
    main()


