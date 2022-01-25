import multiprocessing
import time

from pyomyo import Myo, emg_mode
# Source of pyomyo library: https://github.com/PerlinWarp/pyomyo

q = multiprocessing.Queue()
buffer_size = 5


# This method gets the 5th latest myo band data
# q is the multiprocessing.Queue which will hold the data
def get_myoband_data(q):
    print(q.qsize())
    emg = list(q.get())
    return emg


def read_myoband_data(q):
    # To change the mode of data, edit mode=emg_mode.<YourMode>
    m = Myo(mode=emg_mode.RAW)
    m.connect()

    def add_to_queue(emg, movement):
        q.put(emg)
        while q.qsize() > buffer_size:
            q.get()

    m.add_emg_handler(add_to_queue)

    # Vibrate to know we connected okay
    m.vibrate(1)

    while True:
        m.run()


if __name__ == "__main__":

    # To use the methods:
    # declare globally, q = multiprocessing.Queue()
    # In the main, add the following lines in a try block:
    # p = multiprocessing.Process(target=read_myoband_data, args=(q,))
    # p.start()
    # To get the latest value: call get_myoband_data(q) where q is the multiprocessing.Queue()
    try:
        p = multiprocessing.Process(target=read_myoband_data, args=(q,))
        p.start()
        while True:
            print(get_myoband_data(q))
    except KeyboardInterrupt:
        p.terminate()
        p.join()
