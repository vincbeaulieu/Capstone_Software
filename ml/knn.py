import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle as pk


def save_model(model, model_path):
    with open(model_path, 'wb') as ml_file:
        pk.dump(model, ml_file)


def load_model(model_path):
    model = pk.load(open(model_path, 'rb'))
    return model


# Trains classifier with the data in data_filepath (csv/<dataset>.csv)
def train_model(model, dataset_path):
    print("Starting model training...")

    # Extracting data from csv
    dataset = pd.read_csv(dataset_path)
    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # Splitting the dataset into training_set and test_set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)

    # Scaling the data
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    # Training the model
    model.fit(x_train, y_train)

    # Testing the classifier by getting predictions
    y_pred = model.predict(x_test)

    # Displaying the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    plot_confusion_matrix(model, x_test, y_test)
    plt.show()

    # Printing the model accuracy
    model_accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: ", model_accuracy)

    # Returning the model
    print("Model training completed.")
    return model, sc


def get_prediction(input_data, model, scaler):
    transformed_data = scaler.transform(input_data)
    prediction = model.predict(transformed_data)
    return prediction


if __name__ == "__main__":
    # Import ML model library
    from sklearn.neighbors import KNeighborsClassifier

    # Creating a ML model
    knn_model = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)

    # Loading a ML model
    # knn_model = load_model('../ml/saved_model')

    # Training the model
    knn_model, knn_scaler = train_model(knn_model, "../csv/suyash10gpieday.csv")

    # Saving the model
    save_model(knn_model, '../ml/saved_model')

    # Using the model to predict input data
    # get_prediction(emg_input, knn_model, knn_scaler)
