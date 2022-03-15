import pandas as pd
import csv
import os


# Return Pandas DataFrame and save output to CSV
def dataset_to_csv(filepath, dataset):
    dataframe = pd.DataFrame(dataset)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    dataframe.to_csv(filepath, index=False, header=False)
    return dataframe


# Read saved files and return 1D list
def reader(filepath):
    lines = []
    with open(filepath, 'r') as file:
        for line in file:
            lines.append(line[:-1])
    file.close()
    return lines


# Read saved files and return 2D list
def csv_reader(filepath, delimiter=','):
    table = []
    with open(filepath, newline='') as file:
        rows = csv.reader(file, delimiter=delimiter)
        for row in rows:
            table.append(row)
    file.close()
    return table


def test():
    print("Testing Toolbox...")

    my_dataset = [10, 20, 30, 40]
    filepath = '../csv/my_dataset.csv'

    print("\nToolbox test :: Export dataset to csv :: return: Pandas Dataframe")
    my_dataframe = dataset_to_csv(filepath, my_dataset)
    print(my_dataframe)

    print("\nToolbox test :: File Reader :: return: List")
    my_file = reader(filepath)
    print(my_file)

    print("\nToolbox test :: CSV Reader :: return: List of Lists")
    my_csvset = [[10, 20, 30, 40], [5, 15, 25, 35]]
    filepath = '../csv/my_csvset.csv'
    my_dataframe = dataset_to_csv(filepath, my_csvset)
    my_csv = csv_reader(filepath)
    print(my_csv)

    pass
