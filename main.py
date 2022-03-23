import ml.dual_ml
from myoband.MyoBandData import read_myoband_data, get_myoband_data
from rbpi.servoGestureOutput import motion
from rbpi.gestures import gestures_list
import numpy as np
import pandas as pd
import multiprocessing
from time import sleep, time
from rbpi.haptic_feedback import HapticFeedback
import RPi.GPIO as GPIO
import os.path
from buttonTest import buttonStatus
from ml.dual_ml import initialize, predict, load, cpu_limit, handRemoved
from led import set_light_on, set_light_off

# NOTE: This is already declared in buttonTest.py
GPIO.setmode(GPIO.BCM)  # Use physical pin numbering
GPIO.setup(15, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)  # Set pin 10 to be an input pin
GPIO.setup(8, GPIO.OUT, initial=GPIO.LOW)  # Red
GPIO.setup(7, GPIO.OUT, initial=GPIO.LOW)  # Green

q1 = multiprocessing.Queue()
q2 = multiprocessing.Queue()

# gestures = list(gestures_positions.keys())

gestures = [g for g in gestures_list if g not in handRemoved]
gesture_counters = [0] * len(gestures)
hf = HapticFeedback('/dev/ttyUSB0', 9600)


def print_error(exception):
    _RED_ = '\033[91m'
    _END_ = '\033[0m'
    print(_RED_ + str(exception) + _END_)


def launch():
    # Variable declarations:
    global gesture_counters
    buttonStatus(0)

    # Defining filepath of the dataset (Will be simplified later tonight)
    filepath = "csv/"
    filename = "dataset.csv"
    file_pathname = filepath + filename

    # Creating many ML models
    model_qty = 1
    model_size = 6
    ml_objects = None

    # Import and create a ML model
    # from sklearn.neighbors import KNeighborsClassifier
    # ml_model = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)

    # Haptic feedback initialization
    hf.start()  # Disabled by default

    print("Starting myoband connection...")

    set_light_on("both")

    p = multiprocessing.Process(target=read_myoband_data, args=(q1, q2,))
    try:
        p.start()
        sleep(5)
        try:
            if os.path.isfile(file_pathname):
                num_lines = sum(1 for line in open(file_pathname))
                if num_lines < 100:
                    raise "File doesn't exist"
                else:
                    # File is already populated. Therefore, load a model
                    ml_objects = load(model_qty)
            else:
                raise "File doesn't exist"

        except Exception as e:
            print_error(str(e))
            calibrate(file_pathname)
            ml_objects = initialize(file_pathname, model_size, model_qty)

        set_light_off("both")
        # classifier, scaler = train_model(ml_model, file_pathname)

        while True:
            if buttonStatus() == 2:
                try:
                    # Erase all the content of the file
                    with open(file_pathname, 'w') as file:
                        file.writelines("")
                    calibrate(file_pathname)
                    # classifier, scaler = train_model(ml_model, file_pathname)

                    set_light_on("both")
                    ml_objects = initialize(file_pathname, model_size, model_qty)
                    set_light_off("both")

                    buttonStatus(0)
                except Exception as e:
                    print_error(e)

            set_light_on("g")

            emg1, emg2 = get_myoband_data(q1, q2)
            emg_data = [emg2 + emg1]

            t0 = time()
            predicted = predict(ml_objects, emg_data, model_qty)
            t1 = time()

            print("prediction: ", predicted)
            print("Prediction time ", (t1-t0))
            motion(predicted[0])

            # prediction_buffer = 1
            # if len(q3) >= prediction_buffer:
            #    counter_index = gesture_counters.index(max(gesture_counters))
            #
            #     Perform the motion on the prosthetic
            #    motion(gestures[counter_index])
            #    print(gestures[counter_index])
            #
            #    q3.clear()
            #    gesture_counters = [0] * len(gestures)
            # else:
            #    prediction = predicted[0]
            #    q3.append(prediction)
            #    counter_index = gestures.index(prediction)
            #    gesture_counters[counter_index] += 1

    except KeyboardInterrupt:
        motion("handExit")
        p.terminate()
        hf.terminate()
        hf.join()
        p.join()
        set_light_off("both")
        GPIO.cleanup()


# Creates the dataset in the csv folder. Returns the path of the file relative to
# the top directory of the project (returns path like "csv/<filename>.csv)
def calibrate(filepath):
    set_light_on("r")
    buttonStatus(0)
    hf.disable()
    print("Starting data collection for calibration...")
    secs = 1
    for x in range(1):
        for gesture in gestures:

            print('Please perform the following gesture: ' + str(gesture))
            motion(gesture)

            while buttonStatus() == 0:
                pass
            buttonStatus(0)

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

            for _ in range(2):
                set_light_off("r")
                sleep(0.25)
                set_light_on("r")
                sleep(0.25)

    print("Data collection complete. Dataset file created")
    hf.enable()
    set_light_off("r")


if __name__ == '__main__':
    # cpu_limit()
    launch()
    pass

