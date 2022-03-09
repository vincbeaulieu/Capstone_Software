import serial
from threading import Thread
from time import sleep

class HapticFeedback(Thread):
    def __init__(self, port, baud_rate):
        Thread.__init__(self)
        self.is_enabled = False
        self.port = port
        self.baud_rate = baud_rate

    def run(self):
        with serial.Serial(self.port, self.baud_rate, timeout=1) as arduino:
            sleep(0.1)
            while(True):
                if arduino.isOpen():
                    if (self.is_enabled):
                        arduino.write(b'\x01')
                    else:
                        arduino.write(b'\x00')
                    sleep(1)
                else:
                    print('Arduino not connected, haptic feedback not operational')
            
    def enable(self):
        self.is_enabled = True

    def disable(self):
        self.is_enabled = False

def test():
    hf = HapticFeedback('/dev/ttyUSB0', 9600)
    hf.start()
    print('Haptic feedback test starting')
    print('Disabled')
    sleep(5)
    hf.enable()
    print('Enabled')
    sleep(5)
    hf.disable()
    print('Disabled')
    sleep(5)
    print('Exiting')
    exit(0)

if __name__ == '__main__':
    test()
