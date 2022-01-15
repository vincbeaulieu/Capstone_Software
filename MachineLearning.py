
import pickle as pk
import os
from tensorflow import keras as tfk

def training():
    from sklearn.model_selection import train_test_split
    input_training, input_testing, output_training, output_testing = train_test_split(input_data, output_data, test_size=1/10, random_state=0)
    
    
    pass

# Saved ML model will be in the 'pkl' folder
def save(model,name=None):
    # Save the machine learning model
    filepath = 'pkl/'
    if name != None:
        filepath += name
    else:
        filepath += 'MachineLearning_model'

    # Check if file exist
    index = 0
    temp = filepath + "_v" + str(index).zfill(2)
    while(os.path.isfile(temp)):
        # If file does not exist, increment the name index
        index += 1
        temp = filepath + "_v" + str(index)
    # Found a unique name, use that name to save the model
    filepath = temp

    # Create the file
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Dump the model in the file
    pk.dump(model, open(filepath,'wb'))


# Load a machine learning model
def load(filename):
    filepath = "pkl/" + filename
    model = pk.load(open(filepath,'rb'))
    return model


def test():
    

    pass