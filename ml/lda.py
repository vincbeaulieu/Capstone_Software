
from myoband.MyoBandData import read_myoband_data, get_myoband_data
from ml_class import train_model, get_prediction
from rbpi.servoGestureOutput import motion
import multiprocessing
from time import sleep


if __name__ == "__main__":

    # Import and create a ML model
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda_model = LinearDiscriminantAnalysis()

    # Select a ML model
    ml_model = lda_model

    # Train the ML model
    dataset_name = "suyash10gpieday.csv"
    ml_model, ml_scaler = train_model(ml_model, dataset_name)

    # Main Code
    q1 = multiprocessing.Queue()
    q2 = multiprocessing.Queue()
    p = multiprocessing.Process(target=read_myoband_data, args=(q1, q2,))
    try:
        print("Starting myoband...")
        p.start()
        sleep(5)

        # Wait for user input
        input("Press enter to start")

        while True:

            # Read data from EMG sensors
            emg1, emg2 = get_myoband_data(q1, q2)
            emg_data = [emg1 + emg2]

            # Use the ML model on the EMG data
            prediction = get_prediction(emg_data, ml_model, ml_scaler)

            # Display and Perform the predicted gesture
            print(prediction)
            motion(prediction)

    except KeyboardInterrupt:
        p.terminate()
        p.join()
