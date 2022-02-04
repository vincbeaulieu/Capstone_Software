import multiprocessing
from time import sleep
from pyomyo import Myo, emg_mode
import numpy as np
import csv

# Source of pyomyo library: https://github.com/PerlinWarp/pyomyo

q_myo1 = multiprocessing.Queue()
q_myo2 = multiprocessing.Queue()

BUFFER_SIZE = 5
ADDRESS_MYO1 = [211, 244, 61, 95, 129, 253]
ADDRESS_MYO2 = [74, 84, 45, 195, 122, 208]

# This method gets the 5th latest myo band data
# q is the multiprocessing.Queue which will hold the data
def get_myoband_data(q1, q2):
    emg1 = list(q1.get())
    emg2 = list(q2.get())
    return emg1, emg2

def read_myoband_data(q1, q2):
    # To change the mode of data, edit mode=emg_mode.<YourMode>
    myo_1 = Myo(mode=emg_mode.FILTERED)
    myo_2 = Myo(mode=emg_mode.FILTERED)

    myo_1.connect(ADDRESS_MYO1) # RED LED
    myo_2.connect(ADDRESS_MYO2) # Turquoise LED

    def add_to_queue_myo1(emg, movement):
        q1.put(emg)
        while q1.qsize() > BUFFER_SIZE:
            q1.get()

    def add_to_queue_myo2(emg, movement):
        q2.put(emg)
        while q2.qsize() > BUFFER_SIZE:
            q2.get()

    myo_1.add_emg_handler(add_to_queue_myo1)
    myo_2.add_emg_handler(add_to_queue_myo2)

    # Vibrate to know we connected okay
    # myo_1.vibrate(1)
    # myo_2.vibrate(1)

    while True:
        myo_1.run()
        myo_2.run()


if __name__ == "__main__":


    # To use the methods:
    # declare globally, q = multiprocessing.Queue()
    # In the main, add the following lines in a try block:
    # p = multiprocessing.Process(target=read_myoband_data, args=(q,))
    # p.start()
    # To get the latest value: call get_myoband_data(q) where q is the multiprocessing.Queue()
    try:
        p = multiprocessing.Process(target=read_myoband_data, args=(q_myo1,q_myo2,))
        p.start()
        
        # Erase content of the file:
        with open('myodata.csv','w') as file:
            file.close()

        while True:
            sleep(0.2)
            emg1, emg2 = get_myoband_data(q_myo1,q_myo2)
            emg_data = np.concatenate((emg1, emg2), axis=None)

            # Append content to the file:
            with open('myodata.csv','a',newline='') as file:
                writer = csv.writer(file)
                writer.writerow(emg_data)

            print(emg_data)
    except KeyboardInterrupt:
        p.terminate()
        p.join()
        pass