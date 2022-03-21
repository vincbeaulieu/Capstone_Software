import buttonTest
from myoband.MyoBandData import read_myoband_data, get_myoband_data
# from knn import train_classifier, get_predicted_movement
# from lda import train_classifier, get_predicted_movement
from ml.ml_class import train_model, get_prediction
from rbpi.servoGestureOutput import motion
from rbpi.gestures import gestures_positions
from rbpi.haptic_feedback import HapticFeedback
from time import sleep, time
import numpy as np
import pandas as pd
import multiprocessing
import RPi.GPIO as GPIO
import os.path

GPIO.setmode(GPIO.BCM)  # Use physical pin numbering
GPIO.setup(15, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)  # Set pin 10 to be an input pin

from buttonTest import buttonStatus

q1 = multiprocessing.Queue()
q2 = multiprocessing.Queue()

gestures = list(gestures_positions.keys())
gesture_counters = [0] * len(gestures)
hf = HapticFeedback('/dev/ttyUSB0', 9600)


def test():
    # Variable declarations:
    global gesture_counters
    buttonStatus(0)
    q3 = []

    # Defining filepath of the dataset (Will be simplified later tonight)
    filepath = "csv/"
    filename = "dataset.csv"
    filepathname = filepath + filename
    # Import and create a ML model
    from sklearn.neighbors import KNeighborsClassifier
    ml_model = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)

    # Haptic feedback initialization
    hf.start() # Disabled by default


    print("Starting myoband connection...")
    p = multiprocessing.Process(target=read_myoband_data, args=(q1, q2,))
    try:
        p.start()
        sleep(5)
        try:
            # print('>>> PATH: ' + filepathname)
            if os.path.isfile(filepathname):
                num_lines = sum(1 for line in open(filepathname))
                if num_lines < 100:
                    calibrate(filepathname)
                else:
                    # File is already populated. Therefore, load a model
                    # TODO: Load a ML model (For Vincent) ml_class is not completely ready testing is require before
                    pass
            else:
                # if file doesn't exist
                calibrate(filepathname)
                pass
        except Exception as e:
            print(e)

        classifier, scaler = train_model(ml_model, filename)

        print("train_model done")

        hf.enable()

        while True:
            if buttonStatus() in (1, 2):
                try:
                    if buttonStatus() == 2:  # Then erase all the content of the file
                        with open(filepathname, 'w') as file:
                            file.writelines("")                    
                    buttonStatus(0)
                    # TODO: To simplified (Vincent)
                    calibrate(filepathname)
                    classifier, scaler = train_model(ml_model, filename)

                except Exception as e:
                    print(e)

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
        hf.terminate()
        p.join()
        hf.join()


# Creates the dataset in the csv folder. Returns the path of the file relative to
# the top directory of the project (returns path like "csv/<filename>.csv)
def calibrate(filepath):
    hf.disable()
    print("Starting data collection for calibration...")
    secs = 1

    for gesture in gestures:
        print('Please perform the following gesture: ' + str(gesture))
        motion(gesture)

        while buttonStatus() != 1:
            pass
        buttonStatus(0)

        print('start gesture')# TODO: replace with light led up

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

        print('end gesture')# TODO: replace with close led

    # TODO: LED blink twice
    print("Data collection complete. Dataset file created")
    hf.enable()


if __name__ == '__main__':
   test()
