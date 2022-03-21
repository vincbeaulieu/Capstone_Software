import multiprocessing
import sys
from time import sleep
from pyomyo import Myo, emg_mode
import pandas as pd
import numpy as np
# Source of pyomyo library: https://github.com/PerlinWarp/pyomyo

ADDRESS_MYO1 = [211, 244, 61, 95, 129, 253]  # Red LED
ADDRESS_MYO2 = [74, 84, 45, 195, 122, 208]   # Turquoise LED

# This method gets the 5th latest myo band data
# q is the multiprocessing.Queue which will hold the data
def get_myoband_data(q1, q2):
    emg_1 = list(q1.get())
    emg_2 = list(q2.get())
    return emg_1, emg_2

def myoband_setup(q, addr):
    BUFFER_SIZE = 5

    # To change the mode of data, edit mode=emg_mode.<YourMode>
    myo = Myo(mode=emg_mode.RAW)
    myo.connect(addr)
    print(myo.bt.get_connections())
    print('device name: %s' % myo.read_attr(0x03).payload)

    # qsize raises a NotImplementedError on Mac OSX because of broken sem_getvalue()
    # This can be solved by returning the internal deque buffer length in the built-in library
    def add_to_queue_myo(emg, movement):
        q.put(emg)
        while q.qsize() > BUFFER_SIZE:
            q.get()

    myo.add_emg_handler(add_to_queue_myo)

    # Prevent the myo from disconnecting (Keep Alive)
    myo.sleep_mode(1)

    return myo


def read_myoband_data(myo_1, myo_2):
    try:
        while True:
            myo_1.run()
            myo_2.run()
    except KeyboardInterrupt:
        myo_1.disconnect()
        myo_2.disconnect()


if __name__ == "__main__":

    q_myo1 = multiprocessing.Queue()
    q_myo2 = multiprocessing.Queue()

    # Todo: while not connected:
    myo1 = myoband_setup(q_myo1, ADDRESS_MYO1)
    myo2 = myoband_setup(q_myo2, ADDRESS_MYO2)

    p1 = multiprocessing.Process(target=read_myoband_data, args=(myo1, myo2,))
    try:
        emg_toto = []

        p1.start()
        print("started")

        count = 15
        while count > 0:
            emg1, emg2 = get_myoband_data(q_myo1, q_myo2)
            emg_data = [emg1 + emg2]

            emg_toto.append(emg_data)

            # with open('myodata.csv','a',newline='\n') as file:
            #     file.write(str(emg_data))

            print(emg_data)

            count -= 1

    except KeyboardInterrupt:
        p1.terminate()
        p1.join()
