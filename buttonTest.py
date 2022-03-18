
from time import sleep, time

import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
GPIO.setup(15, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

global button_toggle
global button_state
global start


def button_status(setStatus=None):
    global button_state
    if setStatus is not None:
        button_state = setStatus
    return button_state


def my_callback(channel):
    global start
    global button_toggle

    if GPIO.input(15) == 1:
        if not button_toggle:
            start = time()
            button_toggle = True

        # print("Button pressed")

    else:
        # print("Button released")
        button_toggle = False

        elapsed = time() - start
        print(elapsed)

        if elapsed >= 3:
            button_status(2)
            print("Button was pushed for a while!")
        elif elapsed >= .1:
            button_status(1)
            print("Button was pushed")
        else:
            print("lol")
            button_status(0)


GPIO.add_event_detect(15, GPIO.BOTH, callback=my_callback, bouncetime=200)

if __name__ == '__main__':
    while True:
        sleep(0.5)
        print(button_status())
