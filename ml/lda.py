
from myoband.MyoBandData import read_myoband_data, get_myoband_data
import multiprocessing
import time

from ml_class import train_model, get_prediction


if __name__ == "__main__":

    q1 = multiprocessing.Queue()
    q2 = multiprocessing.Queue()

    # Import a ML library
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    # Create the ML model
    lda_model = LinearDiscriminantAnalysis()

    # Train the ML model
    dataset_name = "suyash10gpieday.csv"
    lda_model, lda_scaler = train_model(lda_model, dataset_name)

    print("Starting myoband")
    p = multiprocessing.Process(target=read_myoband_data, args=(q1, q2,))
    try:
        p.start()
        time.sleep(5)
        while True:
            input("Press enter to start")
            emg1, emg2 = get_myoband_data(q1, q2)
            emg_input_data = [emg1 + emg2]
            prediction = get_prediction(emg_input_data, lda_model, lda_scaler)
            print(prediction)
            # motion(prediction)

    except KeyboardInterrupt:
        p.terminate()
        p.join()
