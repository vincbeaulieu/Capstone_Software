
import pickle as pk
import os

# Saved ML model will be in the 'pkl' folder

# Save a machine learning model
def save(model,name=None):
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
def load(filepath):
    model = pk.load(open(filepath,'rb'))
    return model


def test():
    
    pass