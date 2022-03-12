# Sample code from https://raspberrypihq.com/use-a-push-button-with-raspberry-pi-gpio/
from time import sleep, time
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BOARD)  # Use physical pin numbering
GPIO.setup(10, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)  # Set pin 10 to be an input pin
#GPIO.setwarnings(False) # Ignore warning for now
#
# def myInterrupt(channel):
#     global buttonStatus
#     start_time = time()
#
#     while GPIO.input(channel) == 0:  # Wait for the button up
#         pass
#
#     buttonTime = time() - start_time  # How long was the button down?
#
#     if buttonTime >= 4:
#         buttonStatus = 3
#     elif buttonTime >= 2:
#         buttonStatus = 2
#     elif buttonTime >= .1:
#         buttonStatus = 1
#
#
# GPIO.add_event_detect(10, GPIO.FALLING, callback=myInterrupt, bouncetime=500)

#timer = time()

#while True: # Run forever
#   if GPIO.input(10) == GPIO.HIGH:
#       print(timer)
#       print("Button was pushed!")
#        sleep(0.5)

def my_callback(channel):
    global start
    global end
    if GPIO.input(10) == 1:
        start = time()
    if GPIO.input(10) == 0:
        end = time()
        elapsed = end - start
        print(elapsed)

    if elapsed >= 4:
        buttonStatus = 3
    elif elapsed >= 2:
        buttonStatus = 2  
   elif elapsed >= .1: 
        buttonStatus = 1

GPIO.add_event_detect(10, GPIO.BOTH, callback=my_callback, bouncetime=200)

while True:
    print("-")
    sleep(0.25)
