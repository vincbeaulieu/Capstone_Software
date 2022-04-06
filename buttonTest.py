from time import sleep,time
import RPi.GPIO as GPIO

# GPIO.setmode(GPIO.BCM)
# GPIO.setup(15, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
# GPIO.setmode(GPIO.BOARD)
GPIO.setup('SPI1_SCK', GPIO.IN, pull_up_down=GPIO.PUD_DOWN)  # Button, pin 22

global button_state
global button_toggle 

buttontoggle = False
button_state = 0

def buttonStatus(setStatus=None):
    global button_state
    if setStatus != None:
        button_state = setStatus
    return button_state

start = 0
def my_callback(channel):
    global start
    global buttontoggle

    if GPIO.input('SPI1_SCK') == 1:
        if buttontoggle == False:
            start = time()
            buttontoggle = True

        #print("Button pressed")

    else:
        #print("Button released")
        buttontoggle = False

        elapsed = time() - start
        #print(elapsed)

        if elapsed >= 6:
            buttonStatus(2)
           # print("button status 2")
        elif elapsed >= 0.1:
            buttonStatus(1)
            #print("Button status 1")
        else:
            print("button status 0")
            buttonStatus(0)

# GPIO.add_event_detect(15, GPIO.BOTH, callback=my_callback, bouncetime=200)
GPIO.add_event_detect('SPI1_SCK', GPIO.BOTH, callback=my_callback, bouncetime=200)

if __name__ == '__main__':
    while True:
        print(buttonStatus())
        sleep(1)
