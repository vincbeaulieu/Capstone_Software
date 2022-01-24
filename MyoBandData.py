import multiprocessing
from pyomyo import Myo, emg_mode

# ------------ Myo Setup ---------------
q = multiprocessing.Queue()


def get_myoband_data(q):
    emg = list(q.get())
    return emg


def read_myoband_data(q):
    m = Myo(mode=emg_mode.RAW)
    m.connect()

    def add_to_queue(emg, movement):
        q.put(emg)

    m.add_emg_handler(add_to_queue)

    def print_battery(bat):
        print("Battery level:", bat)

    m.add_battery_handler(print_battery)

    # Orange logo and bar LEDs
    m.set_leds([128, 0, 0], [128, 0, 0])
    # Vibrate to know we connected okay
    m.vibrate(1)

    while True:
        m.run()


# -------- Main Program Loop -----------
if __name__ == "__main__":

    # to use the methods:
    # declare globally, q = multiprocessing.Queue()
    # then in the main, the following lines:
    # p = multiprocessing.Process(target=read_myoband_data, args=(q,))
    # p.start()
    # this is demonstrated in this script

    p = multiprocessing.Process(target=read_myoband_data, args=(q,))
    p.start()

    while True:
#        x = input("dd")
        print(get_myoband_data(q))
