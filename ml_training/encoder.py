
import numpy as np

# One Hot Encoder
def one_hot_encoder(sequence,categories,scale=None,remove_last=True):
    scale = (scale,len(categories))[scale==None]
    mapping = dict(zip(categories, range(scale)))
    results = [mapping[i] for i in sequence]
    output = np.rot90(np.eye(scale, dtype=int)[results])
    return (output,output[:-1])[remove_last]