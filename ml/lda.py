
from ml_class import train_model, save_model

def lda_model():
    # Import and create a ML model
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    # Disgraceful 14%
    return LinearDiscriminantAnalysis()


def svc_model():
    from sklearn.svm import NuSVC, SVC
    from sklearn.ensemble import BaggingClassifier

    # Okay 43%
    return NuSVC(gamma="auto", random_state=1)

    # Terrible 20%
    # return SVC(kernel='linear', C=1)

    # Okay (slow to train) 43%
    # return BaggingClassifier(NuSVC(gamma="auto", random_state=1))


def tree_model():
    from sklearn.ensemble import BaggingClassifier
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier

    # Best (very slow to train) 57%
    # return BaggingClassifier(HistGradientBoostingClassifier())

    # Best 56%
    return HistGradientBoostingClassifier()

    # Very Good 54%
    # return GradientBoostingClassifier()

    # Very Good 51%
    # return RandomForestClassifier()

    # Good 34%
    # return DecisionTreeClassifier()

def knn_model():
    from sklearn.ensemble import BaggingClassifier
    from sklearn.neighbors import KNeighborsClassifier

    # Okay 40%
    return KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)

    # Okay 40%
    # return BaggingClassifier(KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2))


# NOTE: BaggingClassifier improve the results by about 1%, but it is extremely slow to train
# And may not be worth implementing on a Raspberry pi

if __name__ == "__main__":

    # Select a ML model
    ml_model = tree_model()

    # Train the ML model
    dataset_name = "suyash10gpieday.csv"
    ml_model, ml_scaler = train_model(ml_model, dataset_name)

    # Save the ML model
    model_name = "ml_test"
    save_model(ml_model, ml_scaler, model_name)

    # Use the ML model with the Myobands
    #myo_predict(model, scaler)



