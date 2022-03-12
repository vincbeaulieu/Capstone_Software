# Sample code from https://raspberrypihq.com/use-a-push-button-with-raspberry-pi-gpio/
import RPi.GPIO as GPIO  # Import Raspberry Pi GPIO library
from time import time as time
from time import sleep as sleep


def myInterrupt(channel):
    global buttonStatus
    start_time = time.time()

    while GPIO.input(channel) == 0:  # Wait for the button up
        pass

    buttonTime = time.time() - start_time  # How long was the button down?

    if buttonTime >= 4:
        buttonStatus = 3
        print("Button was pushed for a very long time!")
    elif buttonTime >= 2:
        buttonStatus = 2
        print("Button was pushed for a long time!")
    elif buttonTime >= .1:
        buttonStatus = 1
        print("Button was pushed!")


GPIO.setwarnings(False)  # Ignore warning for now
GPIO.setmode(GPIO.BOARD)  # Use physical pin numbering
GPIO.setup(10, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)  # Set pin 10 to be an input pin
GPIO.add_event_detect(10, GPIO.FALLING, callback=myInterrupt, bouncetime=500)

while True:  # Run forever
    if GPIO.input(10) == GPIO.HIGH:
        print("Button was pushed!")
        sleep(0.1)
