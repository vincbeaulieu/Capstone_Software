import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle as pk
from pathlib import Path
from Capstone_Software.myoband.MyoBandData import read_myoband_data, get_myoband_data
# from rbpi.servoGestureOutput import motion
import multiprocessing


def save_model(model, scaler, name="untitled", path="saved_model/"):
    pathname = path + name

    # Creating a directory if it doesn't exist
    Path(pathname).mkdir(parents=True, exist_ok=True)

    # Saving the model
    pk.dump(model, open(pathname + "/model.pkl", 'wb'))

    # Saving the scaler
    pk.dump(scaler, open(pathname + "/scaler.pkl", 'wb'))


def load_model(name, path="saved_model/"):
    pathname = path + name

    # Loading the model
    model = pk.load(open(pathname + "/model.pkl", 'rb'))

    # Loading the scaler
    scaler = pk.load(open(pathname + "/scaler.pkl", 'rb'))

    return model, scaler


def evaluate_model(model, input_data, output_data, name="", path="saved_model/", fold=None):
    # Testing the classifier by getting predictions
    expected_output = model.predict(input_data)

    # Displaying the confusion matrix
    print(confusion_matrix(output_data, expected_output))
    ConfusionMatrixDisplay.from_estimator(model, input_data, output_data)
    # ConfusionMatrixDisplay.from_predictions(output_data, expected_output)

    plt.title(name)
    plt.savefig(path + 'confusion_matrix.png')
    plt.show()

    # Printing the model accuracy
    model_accuracy = accuracy_score(output_data, expected_output)
    print("Accuracy: ", model_accuracy)

    # K-fold cross validation
    if fold is not None:
        scores = cross_val_score(model, input_data, output_data, cv=fold)
        print("Cross validation average = ", scores.mean())
        print(str(fold) + "-fold cross validation score: ", scores)

# Trains classifier with the data in data_filepath (csv/<dataset>.csv)
def train_model(model, data_name="dataset.csv", data_path="../csv/", fold=None):
    dataset_pathname = data_path + data_name

    print("Starting model training...")

    # Extracting data from csv
    dataset = pd.read_csv(dataset_pathname)
    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # Splitting the dataset into training_set and test_set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)

    # Scaling the data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Training the model
    model.fit(x_train, y_train)

    # Evaluating the model
    evaluate_model(model, x_test, y_test, fold=fold)

    # Returning the model
    print("Model training completed.")
    return model, scaler


def get_prediction(input_data, model, scaler):
    transformed_data = scaler.transform(input_data)
    prediction = model.predict(transformed_data)
    confidence = model.predict_proba(transformed_data)
    return prediction, confidence


def myo_predict(ml_model, ml_scaler):
    q1 = multiprocessing.Queue()
    q2 = multiprocessing.Queue()
    p = multiprocessing.Process(target=read_myoband_data, args=(q1, q2,))
    try:
        print("Starting myoband...")
        p.start()

        while True:
            # Read data from EMG sensors
            emg1, emg2 = get_myoband_data(q1, q2)
            emg_data = [emg1 + emg2]

            # Use the ML model on the EMG data
            prediction = get_prediction(emg_data, ml_model, ml_scaler)

            # Display and Perform the predicted gesture
            print(prediction)
            # motion(prediction)  # This line only work on the raspberry pi

    except KeyboardInterrupt:
        p.terminate()
        p.join()


# Basic Machine Learning Structure:
if __name__ == "__main__":
    # Some model parameters
    model_name = "knn_model"

    # Import and create a ML model
    from sklearn.neighbors import KNeighborsClassifier
    ml_model = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)

    # Train the ML model
    dataset_name = "suyash10gpieday.csv"
    ml_model, ml_scaler = train_model(ml_model, dataset_name, fold=10)

    # Save the ML model
    save_model(ml_model, ml_scaler, name=model_name)

    # Load a ML model
    ml_model, ml_scaler = load_model(model_name)

    # Use the ML model
    # prediction = get_prediction(input_data, knn_model, knn_scaler)

    # Use the ML model with the Myobands
    myo_predict(ml_model, ml_scaler)

