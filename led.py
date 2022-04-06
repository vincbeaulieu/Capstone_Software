import time

import RPi.GPIO as GPIO # Import Raspberry Pi GPIO library
from time import sleep # Import the sleep function from the time module

# GPIO.setwarnings(False) # Ignore warning for now
# GPIO.setmode(GPIO.BCM) # Use physical pin numbering
# GPIO.setup(10, GPIO.OUT, initial=GPIO.LOW) # Red
#GPIO.setup(8, GPIO.OUT, initial=GPIO.LOW) # Green
GPIO.setup('SPI1_CS0', GPIO.OUT, initial=GPIO.LOW)  # Red LED, pin 24
GPIO.setup('SPI1_CS1', GPIO.OUT, initial=GPIO.LOW)  # Green LED, pin 26

def set_light_on(led):
    if led == "r":
        GPIO.output('SPI1_CS0', GPIO.HIGH)
    elif led == "g":
        GPIO.output('SPI1_CS0', GPIO.HIGH)
    else:
        GPIO.output('SPI1_CS0', GPIO.HIGH)
        GPIO.output('SPI1_CS0', GPIO.HIGH)


def set_light_off(led):
    if led == "r":
        GPIO.output('SPI1_CS0', GPIO.LOW)
    elif led == "g":
        GPIO.output('SPI1_CS0', GPIO.LOW)
    else:
        GPIO.output('SPI1_CS0', GPIO.LOW)
        GPIO.output('SPI1_CS0', GPIO.LOW)


if __name__ == '__main__':
    GPIO.cleanup()
    GPIO.setmode(GPIO.BCM)  # Use physical pin numbering
    GPIO.setup(8, GPIO.OUT, initial=GPIO.LOW)  # Red
    GPIO.setup(7, GPIO.OUT, initial=GPIO.LOW) # Green

    while True:
        print("in while")
        set_light_off("r")
        time.sleep(1)
        set_light_off("g")
        time.sleep(1)
        set_light_on("r")
        time.sleep(1)
        set_light_on("g")
        time.sleep(1)
        set_light_off("ha")
        time.sleep(1)
        set_light_on("ha")