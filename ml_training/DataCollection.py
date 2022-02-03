import multiprocessing
import time
import pandas as pd

from MyoBandData import read_myoband_data, get_myoband_data

q_myo1 = multiprocessing.Queue()
q_myo2 = multiprocessing.Queue()

# -------- Main Program Loop -----------
if __name__ == '__main__':
    p = multiprocessing.Process(target=read_myoband_data, args=(q_myo1, q_myo2,))
    p.start()
    time.sleep(10)
    file_name = "HandClose.csv"
    secs = 20
    x = input("Press key for data collection")
    collect = True
    start_time = time.time()
    myo_data_1 = []
    myo_data_2 = []
    while collect:
        if time.time() - start_time < secs:
            myo_data_1, myo_data_2 = get_myoband_data(q_myo1, q_myo2)
        else:
            collect = False

    print('Writing to file:')
    myo_data = myo_data_1 + myo_data_2
    #print(myo_data)
    cols = ["chan1", "chan2", "chan3", "chan4", "chan5", "chan6", "chan7", "chan8",
            "chan9", "chan10", "chan11", "chan12", "chan13", "chan14", "chan15", "chan16"]
    df = pd.DataFrame(cols, myo_data)
    df.to_csv(file_name, index=False, mode='a')
