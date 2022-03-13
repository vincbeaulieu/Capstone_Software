# Sample code from https://raspberrypihq.com/use-a-push-button-with-raspberry-pi-gpio/
from time import sleep, time
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BOARD)  # Use physical pin numbering
GPIO.setup(10, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)  # Set pin 10 to be an input pin


def my_callback(channel):
    global start
    global end
    global buttonStatus

    if GPIO.input(10) == 1:
        start = time()
    if GPIO.input(10) == 0:
        end = time()
        elapsed = end - start
        
        if elapsed >= 4:
            buttonStatus = 3
            print("Button was pushed for a while!")
        elif elapsed >= 2:
            buttonStatus = 2
            print("Button was pushed for a bit!")
        elif elapsed >= .1:
            buttonStatus = 1
            print("Button was pushed!")


GPIO.add_event_detect(10, GPIO.BOTH, callback=my_callback, bouncetime=200)

while True:
    print("-")
    sleep(0.25)
