
from myoband.MyoBandData import read_myoband_data, get_myoband_data
from ml.ml_class import train_model, get_prediction
from rbpi.servoGestureOutput import motion
from rbpi.gestures import gestures_positions
import numpy as np
import pandas as pd
import multiprocessing
from time import sleep, time
import os.path

# NOTE: This is already declared in buttonTest.py
# import RPi.GPIO as GPIO
# GPIO.setmode(GPIO.BCM)  # Use physical pin numbering
# GPIO.setup(10, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)  # Set pin 10 to be an input pin

import buttonTest as btn

q1 = multiprocessing.Queue()
q2 = multiprocessing.Queue()

gestures = list(gestures_positions.keys())
gesture_counters = [0] * len(gestures)


def print_error(exception):
    _RED_ = '\033[91m'
    _END_ = '\033[0m'
    print(_RED_ + str(exception) + _END_)


def launch():
    # Variable declarations:
    global gesture_counters
    q3 = []

    # Defining filepath of the dataset (Will be simplified later tonight)
    filepath = "csv/"
    filename = "dataset.csv"
    file_pathname = filepath + filename
    # Import and create a ML model
    from sklearn.neighbors import KNeighborsClassifier
    ml_model = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)

    print("Starting myoband connection...")
    p = multiprocessing.Process(target=read_myoband_data, args=(q1, q2,))
    try:
        p.start()
        sleep(5)
        try:
            if os.path.isfile(file_pathname):
                num_lines = sum(1 for line in open(file_pathname))
                if num_lines < 100:
                    calibrate(file_pathname)
                else:
                    # File is already populated. Therefore, load a model
                    # TODO: Load a ML model, ml_class is not completely ready. Testing is require.
                    pass
            else:
                # if file doesn't exist
                calibrate(file_pathname)  # handle the creation of the dataset

        except Exception as e:
            print_error(e)

        classifier, scaler = train_model(ml_model, file_pathname)

        while True:
            if btn.button_status() in (1, 2):
                try:
                    if btn.button_status() == 2:  # Then erase all the content of the file
                        with open(file_pathname, 'w') as file:
                            file.writelines("")

                    calibrate(file_pathname)
                    classifier, scaler = train_model(ml_model, file_pathname)

                    btn.button_status(0)
                except Exception as e:
                    print_error(e)

            emg1, emg2 = get_myoband_data(q1, q2)
            emg_data = [emg1 + emg2]
            predicted = get_prediction(emg_data, classifier, scaler)

            if len(q3) >= 15:
                counter_index = gesture_counters.index(max(gesture_counters))

                # Perform the motion on the prosthetic
                motion(gestures[counter_index])
                print(gestures[counter_index])

                q3.clear()
                gesture_counters = [0] * len(gestures)
            else:
                prediction = predicted[0]
                q3.append(prediction)
                counter_index = gestures.index(prediction)
                gesture_counters[counter_index] += 1

    except KeyboardInterrupt:
        motion("handExit")
        p.terminate()
        p.join()


# Creates the dataset in the csv folder. Returns the path of the file relative to
# the top directory of the project (returns path like "csv/<filename>.csv)
def calibrate(filepath):
    print("Starting data collection for calibration...")
    secs = 1

    for gesture in gestures:
        print('Please perform the following gesture: ' + str(gesture))
        motion(gesture)

        # TODO: light led up

        start_time = time()
        myo_data = []
        while time() - start_time < secs:
            m1, m2 = get_myoband_data(q1, q2)
            emg = np.concatenate((m1, m2, gesture), axis=None)
            myo_data.append(emg)

        motion('handOpen')

        print("Gesture collection done... writing to file")
        df = pd.DataFrame(myo_data)
        df.to_csv(filepath, index=False, header=False, mode='a')

        # TODO: close led

    # TODO: LED blink twice
    print("Data collection complete. Dataset file created")


if __name__ == '__main__':
    launch()

    pass


