import pickle as pk
import os
from tensorflow import keras as tfk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from time import sleep

def training(input_data,output_data):

    input_data = pd.DataFrame(input_data)
    output_data = pd.DataFrame(output_data)

    input_data = np.asarray(input_data).astype(np.int_)
    output_data = np.asarray(output_data).astype(np.int_)

def training(input_data, output_data):
  
    from sklearn.model_selection import train_test_split
    input_training, input_testing, output_training, output_testing = train_test_split(input_data, output_data, test_size=1/10, random_state=0)
    
    # API Documentation:
    # https://keras.io/api/

    ## Artificial Neural Network (Models API)
    model = tfk.models.Sequential()
    
    # Nucleotide input layer (Documentation: Layers API)
    model.add(tfk.layers.Dense(units=1, activation='relu'))

    # Hidden layers
    model.add(tfk.layers.Dense(units=64, activation='relu'))
    model.add(tfk.layers.Dense(units=32, activation='relu'))
    model.add(tfk.layers.Dense(units=64, activation='relu'))

    # Output layer
    model.add(tfk.layers.Dense(units=1, activation='sigmoid'))

    # Generate the ANN (Optimizer, Metrics, and Losses API)
    model.compile(optimizer = 'Adam', loss = 'poisson', metrics = ['accuracy'])

    # Feed data to Neural Network
    model.fit(input_training, output_training, batch_size = 10, epochs = 10)
    # The batch size can be changed for the whole data set.

    return model, input_training, input_testing, output_training, output_testing

    pass

# Save a machine learning model
def save(model,filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    pk.dump(model, open(filepath,'wb'))

# Load a machine learning model
def load(filepath):
    model = pk.load(open(filepath,'rb'))
    return model

def test():
    print("Testing Machine Learning...")

    input = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    output = [3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48]

    model, input_training, input_testing, output_training, output_testing = training(input,output)

    save(model,'pkl/MachineLearningTest_v01.pkl')
    sleep(2)
    model = load('pkl/MachineLearningTest_v01.pkl')

    estimated = model.predict(input)
    actual = output

    print(estimated)
    print(actual)

    # If the output resemble the function y = x, then the predictions match the actual values.
    plt.scatter(estimated,actual)
    plt.show()

    pass

test()