from myoband.MyoBandData import read_myoband_data, get_myoband_data
# from knn import train_classifier, get_predicted_movement
# from lda import train_classifier, get_predicted_movement
from ml.ml_class import train_model, get_prediction
from rbpi.servoGestureOutput import motion
from rbpi.gestures import gestures_positions
import numpy as np
import pandas as pd
import multiprocessing
from time import sleep, time
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BOARD)  # Use physical pin numbering
GPIO.setup(10, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)  # Set pin 10 to be an input pin

q1 = multiprocessing.Queue()
q2 = multiprocessing.Queue()
q3 = []

gestures = list(gestures_positions.keys())
gesture_counters = [0] * len(gestures)


buttonStatus = 0
def button():
    global buttonStatus
    start = []

    if GPIO.input(10) == 1:
        start = time()

    if GPIO.input(10) == 0:  # using "else" will improve performance
        end = time()
        elapsed = end - start

        if elapsed >= 5:
            buttonStatus = 2

        elif elapsed >= .1:
            buttonStatus = 1


GPIO.add_event_detect(10, GPIO.BOTH, callback=button, bouncetime=200)


def test():
    global buttonStatus, gesture_counters

    print("Starting myoband connection...")
    p = multiprocessing.Process(target=read_myoband_data, args=(q1, q2,))

    try:
        p.start()
        sleep(5)

        filepath = "csv/dataset.csv"
        num_lines = sum(1 for line in open(filepath))

        # if there is less than 10 lines, we assume the file to be empty
        if num_lines < 10:
            # Therefore, calibration is initiated
            try:
                calibrate(filepath)
            except Exception as e:
                print(e)

        classifier = train_model(filepath)

        while True:

            if buttonStatus in (1, 2):
                try:
                    if buttonStatus == 2:  # Then erase all the content of the file
                        with open(filepath, 'w') as file:
                            file.writelines("")

                    calibrate(filepath)
                    classifier = train_model(filepath)
                    buttonStatus = 0
                except Exception as e:
                    print(e)

            emg1, emg2 = get_myoband_data(q1, q2)
            emg_data = [emg1 + emg2]
            predicted = get_prediction(emg_data, classifier)

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
    global buttonStatus

    print("Starting data collection for calibration...")
    secs = 1

    for gesture in gestures:
        print('Please perform the following gesture: ' + str(gesture))
        motion(gesture)

        while buttonStatus != 1:
            pass  # Wait button press
        buttonStatus = 0

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
    test()
