# SPDX-FileCopyrightText: 2021 ladyada for Adafruit Industries
# SPDX-License-Identifier: MIT
"""Simple test for a standard servo on channel 0 and a continuous rotation servo on
channel 1."""
import time
from adafruit_servokit import ServoKit
# Set channels to the number of servo channels on your kit.
# 16 for FeatherWing, 16 for Shield/HAT/Bonnet.
kit = ServoKit(channels=16)

gesture = {
    'close':[-1,-1,1,1,1], # close hand
    'open':[1,1,-1,-1,-1], # open hand
    'thumbs':[1,-1,1,1,1], # thumbs up
    'flip':[-1,-1,-1,1,1], # flip off
    'rock':[-1,1,1,1,-1], # rock and roll
    'peace':[-1,1,-1,1,1], # peace
    'ok':[-1,-1,-1,-1,-1], # ok
    'exit':[1,1,-1,-1,-1] # exit
}

def motion(input):
    for servo in range(0,5):
        kit.continuous_servo[servo].throttle = gesture[input][servo] 

def validGestures():
    print("\nValid Gestures:")
    for key in gesture.keys():
        print("    " + key)

validGestures()

while True:
    userInput = input("\nEnter Gesture: ")
    if userInput == 'exit':
        motion(userInput)
        break
    elif userInput in gesture:
        motion(userInput)
    else:
        print("Invalid Gesture")
        validGestures()
