from serial import Serial, SerialException
from threading import Thread
from time import sleep

class HapticFeedback(Thread):
    def __init__(self, port, baud_rate):
        Thread.__init__(self)
        self.is_enabled = False
        self.port = port
        self.baud_rate = baud_rate
        self.running = True

    def run(self):
        try:
            with Serial(self.port, self.baud_rate, timeout=1) as arduino:
                sleep(0.1)
                while(self.running):
                    if arduino.isOpen():
                        if (self.is_enabled):
                            arduino.write(b'\x01')
                        else:
                            arduino.write(b'\x00')
                    sleep(0.1)   
                arduino.write(b'\x00')

        except SerialException as e:
            print('\nArduino not connected, haptic feedback not operational')

    def enable(self):
        self.is_enabled = True

    def disable(self):
        self.is_enabled = False

    def terminate(self):
        self.running = False

def test():
    hf = HapticFeedback('/dev/ttyUSB0', 9600)
    try:
        hf.start()
        print('Haptic feedback test starting')
        hf.enable()
        print('Enabled')
        sleep(5)
        hf.disable()
        print('Disabled')
        sleep(5)
        hf.enable()
        print('Enabled')
        sleep(5)
        print('Exiting')
        hf.terminate()
        hf.join()
    except KeyboardInterrupt:
        hf.terminate()
        hf.join()

if __name__ == '__main__':
    test()
