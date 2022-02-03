import multiprocessing
import time
import pandas as pd
import numpy as np

from MyoBandData import read_myoband_data, get_myoband_data

q_myo1 = multiprocessing.Queue()
q_myo2 = multiprocessing.Queue()

# -------- Main Program Loop -----------
if __name__ == '__main__':
    p = multiprocessing.Process(target=read_myoband_data, args=(q_myo1, q_myo2,))
    p.start()
    time.sleep(5)

    file_name = "HandClose.csv"
    secs = 1
    x = input("Press key for data collection")
    collect = True
    start_time = time.time()
    myo_data_1 = []
    myo_data_2 = []
    while collect:
        if time.time() - start_time < secs:
            m1, m2 = get_myoband_data(q_myo1, q_myo2)
            myo_data_1.append(m1)
            myo_data_2.append(m2)

        else:
            collect = False

    print('Writing to file:')
    myo_data_1 = np.array(myo_data_1)
    myo_data_2 =np.array(myo_data_2)
    myo_data = np.concatenate((myo_data_1, myo_data_2), axis=None)
    print(myo_data)
    print(myo_data_1)
    print(myo_data_2)
    cols = ["chan1", "chan2", "chan3", "chan4", "chan5", "chan6", "chan7", "chan8",
            "chan9", "chan10", "chan11", "chan12", "chan13", "chan14", "chan15", "chan16"]
    df = pd.DataFrame(cols, myo_data)
    df.to_csv(file_name, index=False, mode='a')
