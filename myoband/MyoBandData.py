import multiprocessing
from time import sleep
from pyomyo import Myo, emg_mode
import pandas as pd
import numpy as np
# Source of pyomyo library: https://github.com/PerlinWarp/pyomyo

BUFFER_SIZE = 5
ADDRESS_MYO1 = [211, 244, 61, 95, 129, 253]
ADDRESS_MYO2 = [74, 84, 45, 195, 122, 208]

# This method gets the 5th latest myo band data
# q is the multiprocessing.Queue which will hold the data
def get_myoband_data(q1, q2):
    emg_1 = list(q1.get())
    emg_2 = list(q2.get())
    return emg_1, emg_2

# def read_myoband_data(q, addr):
#     # To change the mode of data, edit mode=emg_mode.<YourMode>
#     myo = Myo(mode=emg_mode.RAW)
#     myo.connect(addr)
#     print(myo.bt.get_connections())
#     print('device name: %s' % myo.read_attr(0x03).payload)
#
#     # qsize is broken on Mac and probably on Windows as well.
#     def add_to_queue_myo(emg, movement):
#         q.put(emg)
#         while q.qsize > BUFFER_SIZE:
#             q.get()
#
#     myo.add_emg_handler(add_to_queue_myo)


def read_myoband_data(q1, q2):

    # To change the mode of data, edit mode=emg_mode.<YourMode>
    myo_1 = Myo(mode=emg_mode.RAW)
    myo_2 = Myo(mode=emg_mode.RAW)

    myo_1.connect(ADDRESS_MYO1)  # RED LED
    myo_2.connect(ADDRESS_MYO2)  # Turquoise LED

    # Todo: check what the output print when connected, and check the output when disconnected
    print(myo_1.bt.get_connections())
    print(myo_2.bt.get_connections())

    # Todo: check what the output print when connected, and check the output when disconnected
    print('device name: %s' % myo_1.read_attr(0x03).payload)
    print('device name: %s' % myo_2.read_attr(0x03).payload)

    # qsize is broken on Mac and probably on Windows as well.
    def add_to_queue_myo1(emg, movement):
        q1.put(emg)
        while q1.qsize > BUFFER_SIZE:
            q1.get()

    def add_to_queue_myo2(emg, movement):
        q2.put(emg)
        while q2.qsize > BUFFER_SIZE:
            q2.get()

    myo_1.add_emg_handler(add_to_queue_myo1)
    myo_2.add_emg_handler(add_to_queue_myo2)

    # Vibrate to know we connected okay
    # myo_1.vibrate(1)
    # myo_2.vibrate(1)

    try:
        while True:
            myo_1.run()
            myo_2.run()

    except KeyboardInterrupt:
        myo_1.disconnect()
        myo_2.disconnect()


if __name__ == "__main__":
    # To use the methods:
    # declare globally, q = multiprocessing.Queue()
    # In the main, add the following lines in a try block:
    # p = multiprocessing.Process(target=read_myoband_data, args=(q,))
    # p.start()
    # To get the latest value: call get_myoband_data(q) where q is the multiprocessing.Queue()

    q_myo1 = multiprocessing.Queue()
    q_myo2 = multiprocessing.Queue()
    p = multiprocessing.Process(target=read_myoband_data, args=(q_myo1, q_myo2,))
    try:
        emg_toto = []

        p.start()

        count = 15
        while count > 0:
            emg1, emg2 = get_myoband_data(q_myo1, q_myo2)
            emg_data = [emg1 + emg2]
            # emg_data = np.concatenate((emg1, emg2), axis=None)

            emg_toto.append(emg_data)

            # with open('myodata.csv','a',newline='\n') as file:
            #     file.write(str(emg_data))

            print(emg_data)

            count -= 1

    except KeyboardInterrupt:
        p.terminate()
        p.join()
