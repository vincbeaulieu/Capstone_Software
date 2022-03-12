# Sample code from https://raspberrypihq.com/use-a-push-button-with-raspberry-pi-gpio/
import RPi.GPIO as GPIO  # Import Raspberry Pi GPIO library
from time import sleep, time


def myInterrupt(channel):
    global buttonStatus
    start_time = time.time()

    while GPIO.input(channel) == 0:  # Wait for the button up
        pass

    buttonTime = time.time() - start_time  # How long was the button down?

    if .1 <= buttonTime < 2:  # Ignore noise
        buttonStatus = 1  # 1= brief push
        return buttonStatus

    elif 2 <= buttonTime < 4:
        buttonStatus = 2  # 2= Long push
        return buttonStatus

    elif buttonTime >= 4:
        buttonStatus = 3  # 3= really long push
        return buttonStatus


GPIO.setwarnings(False)  # Ignore warning for now
GPIO.setmode(GPIO.BOARD)  # Use physical pin numbering
GPIO.setup(10, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)  # Set pin 10 to be an input pin
GPIO.add_event_detect(10, GPIO.FALLING, callback=myInterrupt, bouncetime=500)
#
# while True:  # Run forever
#     if GPIO.input(10) == GPIO.HIGH:
#         print("Button was pushed!")
#         sleep(0.1)
