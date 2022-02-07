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

    file_name = "HandRelax2.csv"
    secs = 5
    x = input("Press key for data collection")
    start_time = time.time()
    myo_data_1 = []
    myo_data_2 = []
    myo_data = []
    while True:
        if time.time() - start_time < secs:
            m1, m2 = get_myoband_data(q_myo1, q_myo2)
            emg = np.concatenate((m1,m2, "handRelax"), axis=None)
            myo_data_1.append(m1)
            myo_data_2.append(m2)
            myo_data.append(emg)
        else:
            break
    print('Writing to file')
    cols = ["chan1", "chan2", "chan3", "chan4", "chan5", "chan6", "chan7", "chan8",
            "chan9", "chan10", "chan11", "chan12", "chan13", "chan14", "chan15", "chan16", "gesture"]
    df = pd.DataFrame(myo_data, columns=cols)
    df.to_csv(file_name, index=False, header=False, mode='a')
    print("DONE")
