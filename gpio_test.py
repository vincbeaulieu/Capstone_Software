import Jetson.GPIO as GPIO
from time import sleep

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)  # Use physical pin numbering
# GPIO.setup(23, GPIO.IN)  # Red Button, pin 23
GPIO.setup(23, GPIO.OUT, initial=GPIO.LOW)  # Green LED, pin 26
GPIO.setup(24, GPIO.OUT, initial=GPIO.LOW)  # Red LED, pin 24
GPIO.setup(26, GPIO.OUT, initial=GPIO.LOW)  # Green LED, pin 26
while(True):
    # print("High")
    # GPIO.output(23, GPIO.LOW)
    GPIO.output(24, GPIO.HIGH)
    GPIO.output(26, GPIO.HIGH)
    # sleep(2)
    # print("Low")
    # GPIO.output(24, GPIO.LOW)
    # GPIO.output(26, GPIO.LOW)
    # sleep(2)