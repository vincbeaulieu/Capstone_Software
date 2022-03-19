from time import sleep,time
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
GPIO.setup(15, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

global button_state
global button_toggle

buttontoggle = False
button_state = 0

def buttonStatus(setStatus=None):
    global button_state
    if setStatus != None:
        button_state = setStatus
    return button_state


def my_callback(channel):
    global start
    global buttontoggle

    if GPIO.input(15) == 1:
        if buttontoggle == False:
            start = time()
            buttontoggle = True

        #print("Button pressed")

    else:
        #print("Button released")
        buttontoggle = False

        elapsed = time() - start
        print(elapsed)

        if elapsed >= 5:
            buttonStatus(2)
            print("Button was pushed for a while!")
        elif elapsed >= 2:
            buttonStatus(1)
            print("Button was pushed")
        else:
            print("lol")
            buttonStatus(0)



GPIO.add_event_detect(15, GPIO.BOTH, callback=my_callback, bouncetime=200)
