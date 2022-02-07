import multiprocessing
import time
from sklearn.preprocessing import StandardScaler
from ml_training.MyoBandData import read_myoband_data, get_myoband_data
from knn import train_classifier, get_predicted_movement
from servoGestureOutput import motion

q1 = multiprocessing.Queue()
q2 = multiprocessing.Queue()

q3 = []
counter = [0,0,0]
index_dictionary = {
    0:'handClose',
    1:'handOpen',
    2:'handRelax'
}

dictionary = {
    'handClose':0,
    'handOpen':1,
    'handRelax':2
}



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
            emg1, emg2 = get_myoband_data(q1, q2)
            emg_data = []
            emg_data.append(emg1 + emg2)

            predicted = get_predicted_movement(emg_data, sc, classifier)

            if len(q3) >= 11:
                counter_index = counter.index(max(counter))
                motion(index_dictionary[counter_index])
                print(index_dictionary[counter_index])
#                time.sleep(0.5)
                q3.clear()
                counter[0] = 0
                counter[1] = 0
                counter[2] = 0
            else:
                prediction = predicted[0]
                q3.append(prediction)
                counter_index = dictionary[prediction]
                counter[counter_index] += 1

    except KeyboardInterrupt:
        motion("handExit")
        p.terminate()
        p.join()

    pass

if __name__ == '__main__':
    main()


