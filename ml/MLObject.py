
import os
from sklearn.ensemble import HistGradientBoostingClassifier

from ml.ml_class import evaluate_model, get_prediction, save_model, train_model, load_model



# Save and load location
_save_dir = "saved_model/"

class MLObject:
    def __init__(self, model_name="untitled", dataset_path='../csv/dataset.csv', ml_model=None):
        # Save and load location
        _dirname, _filename = os.path.split(model_name)
        self.model_path = _save_dir + model_name
        self.model_dirname = model_name
        self.model_dir = _dirname + '/'
        self.model_name = _filename

        # Dataset location used for training
        _dirname, _filename = os.path.split(dataset_path)
        self.dataset_dir = _dirname + '/'
        self.dataset_name = _filename
        self.dataset_path = dataset_path

        # Dataset (In Memory)
        self._dataset = None

        # ML Classifier
        if ml_model is None:
            ml_model = HistGradientBoostingClassifier(max_depth=5, )  # Best ML model found
        self.model = ml_model
        self.scaler = None
        self.ml_object = [self.model, self.scaler]

        # Some Stats
        self.accuracy = None
        self.scores = None
        self.score = None
        self.fold = None

    def save(self):
        save_model(self.ml_object, self.model_dirname)
        return self

    def load(self, model_dirname):
        self.model, self.scaler = load_model(model_dirname)
        return self

    def print_stats(self):
        fold = self.fold
        accuracy = self.accuracy
        if accuracy is not None:
            print("Model accuracy: ", accuracy)
            if fold is not None:
                print("{0}-fold cross validation scores: \n{1}", fold, self.scores)
                print("Mean score: ", self.score)

    def train(self):
        self.ml_object, self._dataset = train_model(self.model, self.dataset_path)
        return self

    def evaluate(self, fold=None):
        self.fold = fold
        x_train, x_test, y_train, y_test = self._dataset
        results = evaluate_model(self.ml_object, x_test, y_test, self.model_dirname, fold)
        self.accuracy, self.score, self.scores = results
        return self

    def predict(self, input_data):
        pred, conf = get_prediction(input_data, self.ml_object)
        return pred, conf


if __name__ == "__main__":
    MLObject("dual_ml/ml_1", "../csv/suyash10gpieday.csv").train().evaluate().save()



