import serial
from time import sleep

if __name__ == '__main__':
    with serial.Serial("/dev/ttyUSB0", 9600, timeout=1) as arduino:
        sleep(0.1)
        if arduino.isOpen():
            print("{} connected!".format(arduino.port))
            count = 0
            while count < 10:
                arduino.write(b'\x01')
                count += 1
                print(count)
                sleep(1)
            arduino.write(b'\x00')