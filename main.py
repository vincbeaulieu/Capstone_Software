from ml_training.MyoBandData import read_myoband_data, get_myoband_data
from knn import train_classifier, get_predicted_movement
from servoGestureOutput import motion
import numpy as np
import pandas as pd
import multiprocessing
import time

q1 = multiprocessing.Queue()
q2 = multiprocessing.Queue()

q3 = []
counter = [0, 0, 0]
index_dictionary = {
    0: 'handClose',
    1: 'handOpen',
    2: 'handRelax'
}

dictionary = {
    'handClose': 0,
    'handOpen': 1,
    'handRelax': 2
}


def main():
    test()
    return 0


def test():
    print("Starting myoband connection...")
    try:
        p = multiprocessing.Process(target=read_myoband_data, args=(q1, q2,))
        p.start()
        time.sleep(5)
        file_path = create_dataset()
        classifier, sc = train_classifier(file_path)
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


# Creates the dataset in the csv folder. Returns the path of the file relative to
# the top directory of the project (returns path like "csv/<filename>.csv)
def create_dataset():
    print("Starting data collection for calibration...")
    filename = input("Enter filename for dataset: ")
    filepath = "csv/" + filename + ".csv"
    secs = 5
    gestures = list(dictionary.keys())
    arm_positions = ["arm extended to the front", "arm relaxed by your side", "arm extended out to the side"]
    for arm_position in arm_positions:
        print("Collecting data with " + arm_position)
        i = 0
        for gesture in gestures:
            input("Press enter to collect data for " + gesture)
            start_time = time.time()
            myo_data = []
            while time.time() - start_time < secs:
                m1, m2 = get_myoband_data(q1, q2)
                emg = np.concatenate((m1, m2, gesture), axis=None)
                myo_data.append(emg)
            print("Gesture collection done... writing to file")
            df = pd.DataFrame(myo_data)
            df.to_csv(filepath, index=False, header=False, mode='a')
    print("Data collection complete. Dataset file created")
    return filepath


if __name__ == '__main__':
    main()
