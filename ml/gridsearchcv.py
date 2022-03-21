from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
import pandas as pd
import multiprocessing
sc=StandardScaler()
file_path = "csv/" + input("Enter filename for dataset: ") + ".csv"

dataset = pd.read_csv(file_path)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# to read
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

# scaling
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

lda = LinearDiscriminantAnalysis()

params = [{'solver': ['svd', 'lsqr', 'eigen']
           }]


# defining parameter range
grid = GridSearchCV(lda, params, cv=10, scoring='accuracy', return_train_score=False, verbose=1)

# fitting the model for grid search
grid_search = grid.fit(X_train, y_train)

print('Mean Accuracy: %.3f' % grid_search.best_score_)
print('Config: %s' % grid_search.best_params_)
