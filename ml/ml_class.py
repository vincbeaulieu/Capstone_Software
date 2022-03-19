import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle as pk
from pathlib import Path
from myoband.MyoBandData import read_myoband_data, get_myoband_data
# from rbpi.servoGestureOutput import motion
import multiprocessing


# Save and load location
_save_dir = "saved_model/"


# Save reformatted dataset to csv
def dataset_to_csv(datalist, path):
    Path(_save_dir + path).mkdir(parents=True, exist_ok=True)
    pathname = _save_dir + path + "/dataset.csv"
    df = pd.DataFrame(datalist)
    df.to_csv(pathname, index=False, header=False, mode='w')


def save_model(ml_object, model_dirname="untitled"):
    model, scaler = ml_object
    pathname = _save_dir + model_dirname

    # Creating a directory if it doesn't exist
    Path(pathname).mkdir(parents=True, exist_ok=True)

    # Saving the model and the scaler
    pk.dump(model, open(pathname + "/model.pkl", 'wb'))
    pk.dump(scaler, open(pathname + "/scaler.pkl", 'wb'))


def load_model(model_dirname):
    pathname = _save_dir + model_dirname

    # Loading the model and the scaler
    model = pk.load(open(pathname + "/model.pkl", 'rb'))
    scaler = pk.load(open(pathname + "/scaler.pkl", 'rb'))

    return model, scaler


def evaluate_model(ml_object, input_data, output_data, model_dirname="", fold=None):
    model, scaler = ml_object
    # Testing the classifier by getting predictions
    predicted_output = model.predict(input_data)

    # Displaying the confusion matrix
    print(confusion_matrix(output_data, predicted_output))
    ConfusionMatrixDisplay.from_predictions(output_data, predicted_output)

    plt.title(model_dirname)
    Path(_save_dir + model_dirname).mkdir(parents=True, exist_ok=True)
    plt.savefig(_save_dir + model_dirname + '/confusion_matrix.png')
    plt.show()

    accuracy = accuracy_score(output_data, predicted_output)
    score, scores = None, None
    if fold is not None:
        # K-fold cross validation
        scores = cross_val_score(model, input_data, output_data, cv=fold)
        score = scores.mean()
    return accuracy, score, scores


def data_extractor(dataset_path):
    # Extracting data from csv
    dataset = pd.read_csv(dataset_path)
    in_data = dataset.iloc[:, :-1].values
    out_data = dataset.iloc[:, -1].values
    return in_data, out_data


# Trains classifier with the data in data_filepath (csv/<dataset>.csv)
def train_model(model, data_path="../csv/dataset.csv"):
    # Extracting data from csv
    x, y = data_extractor(data_path)

    # Splitting the dataset into training_set and test_set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)

    # Scaling the data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Training the model
    model.fit(x_train, y_train)

    # Returning the model and dataset
    return [model, scaler], [x_train, x_test, y_train, y_test]


def get_prediction(input_data, ml_object):
    model, scaler = ml_object
    transformed_data = scaler.transform(input_data)
    prediction = model.predict(transformed_data)
    confidence = model.predict_proba(transformed_data)
    return prediction, confidence


def myo_predict(ml_object):
    model, scaler = ml_object

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
            prediction = get_prediction(emg_data, ml_object)

            # Display and Perform the predicted gesture
            print(prediction)
            # motion(prediction)  # This line only work on the raspberry pi

    except KeyboardInterrupt:
        p.terminate()
        p.join()


# Basic Machine Learning Structure:
def launch():

    # Import and create a ML model
    from sklearn.neighbors import KNeighborsClassifier
    ml_model = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)

    # Train the ML model
    dataset_path = "../csv/suyash10gpieday.csv"
    ml_object, dataset = train_model(ml_model, dataset_path)
    in_train, in_test, out_train, out_test = dataset

    # Evaluating the model
    results = evaluate_model(ml_object, in_test, out_test, fold=3)
    print(results)

    # Save or Load the ML model
    model_name = "knn_model"
    save_model(ml_object, model_dirname=model_name)
    ml_object = load_model(model_name)

    # Use the ML model in stand alone or with the Myobands
    # prediction = get_prediction(input_data, knn_model, knn_scaler)
    myo_predict(ml_object)


if __name__ == "__main__":
    launch()

