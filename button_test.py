# Sample code from https://raspberrypihq.com/use-a-push-button-with-raspberry-pi-gpio/
from time import sleep, time
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BOARD)  # Use physical pin numbering
GPIO.setup(10, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # Set pin 10 to be an input pin


def myInterrupt(channel):
    global buttonStatus
    start_time = time.time()

    while GPIO.input(channel) == 0:  # Wait for the button up
        pass

    buttonTime = time.time() - start_time  # How long was the button down?

    if buttonTime >= 4:
        buttonStatus = 3
    elif buttonTime >= 2:
        buttonStatus = 2
    elif buttonTime >= .1:
        buttonStatus = 1


GPIO.add_event_detect(10, GPIO.FALLING, callback=myInterrupt, bouncetime=500)

# while True:  # Run forever
#     global buttonStatus
#     start_time = time.time()
#
#     buttonTime = time.time() - start_time  # How long was the button down?
#
#     if buttonTime >= 4:
#         buttonStatus = 3
#         print("Button was pushed for a very long time!")
#         time.sleep(0.5)
#
#     elif buttonTime >= 2:
#         buttonStatus = 2
#         print("Button was pushed for a long time!")
#         time.sleep(0.5)
#
#     elif buttonTime >= .1:
#         buttonStatus = 1
#         print("Button was pushed!")
#         time.sleep(0.5)
#
# GPIO.cleanup()
