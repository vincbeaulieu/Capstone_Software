from pathlib import Path

import pandas as pd
from sklearn.utils import shuffle

def data_remover(_data_values, _data_keys, excluded_gestures):
    values_temp, keys_temp = [], []
    for i, data_key in enumerate(_data_keys):
        if not (data_key in excluded_gestures):
            keys_temp.append(data_key)
            values_temp.append(_data_values[i])
    return values_temp, keys_temp


def data_extractor(dataset_path):
    # Extracting data from csv
    dataset = pd.read_csv(dataset_path)
    shuffle(dataset)
    in_data = dataset.iloc[:, :-1].values
    out_data = dataset.iloc[:, -1].values
    return in_data, out_data


# Save reformatted dataset to csv
def dataset_to_csv(datalist, path):
    Path(path).mkdir(parents=True, exist_ok=True)
    pathname = path + "/dataset.csv"
    df = pd.DataFrame(datalist)
    df.to_csv(pathname, index=False, header=False, mode='w')


