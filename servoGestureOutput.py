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
    'handClose': [-1, -1, 1, 1, 1],  # close hand
    'handOpen': [1, 1, -1, -1, -1],  # open hand
    'handThumbsUp': [1, -1, 1, 1, 1],  # thumbs up
    'handFlip': [-1, -1, -1, 1, 1],  # flip off
    'handRock': [-1, 1, 1, 1, -1],  # rock and roll
    'handPeace': [-1, 1, -1, 1, 1],  # peace
    'handOk': [-1, -1, -1, -1, -1],  # ok
    'handExit': [1, 1, -1, -1, -1]  # exit
}

# TODO: test method :)
def motion(input):
    if input != "handRelax":
        for servo in range(0, 5):
            kit.continuous_servo[servo].throttle = gesture[input][servo]


def validGestures():
    print("\nValid Gestures:")
    for key in gesture.keys():
        print("    " + key)


#validGestures()

#while True:
#    userInput = input("\nEnter Gesture: ")
#    if userInput == 'exit':
#        motion(userInput)
#        break
#    elif userInput in gesture:
#        motion(userInput)
#    else:
#        print("Invalid Gesture")
#        validGestures()
