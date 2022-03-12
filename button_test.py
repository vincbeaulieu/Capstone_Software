# Sample code from https://raspberrypihq.com/use-a-push-button-with-raspberry-pi-gpio/
from time import sleep, time
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BOARD) # Use physical pin numbering
GPIO.setwarnings(False)  # Ignore warning for now
GPIO.setup(10, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # Set pin 10 to be an input pin

while True:  # Run forever
    global buttonStatus
    start_time = time.time()

    buttonTime = time.time() - start_time  # How long was the button down?

    if buttonTime >= 4:
        buttonStatus = 3
        print("Button was pushed for a very long time!")
        time.sleep(0.5)

    elif buttonTime >= 2:
        buttonStatus = 2
        print("Button was pushed for a long time!")
        time.sleep(0.5)
        
    elif buttonTime >= .1:
        buttonStatus = 1
        print("Button was pushed!")
        time.sleep(0.5)

GPIO.cleanup()