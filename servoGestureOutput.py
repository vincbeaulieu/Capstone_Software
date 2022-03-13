
# SPDX-FileCopyrightText: 2021 ladyada for Adafruit Industries
# SPDX-License-Identifier: MIT
"""Simple test for a standard servo on channel 0 and a continuous rotation servo on
channel 1."""

from adafruit_servokit import ServoKit

# Set channels to the number of servo channels on your kit.
# 16 for FeatherWing, 16 for Shield/HAT/Bonnet.
kit = ServoKit(channels=16)

gestures_positions = {
    'handClose':    [ 1,  1,  1, -1,  1],  # close hand
    'handOpen':     [-1, -1, -1,  1, -1],  # open hand
    'handThumbsUp': [-1,  1,  1, -1,  1],  # thumbs up
    'handFlip':     [ 1,  1, -1, -1,  1],  # flip off
    'handRock':     [ 1, -1,  1, -1, -1],  # rock and roll
    'handPeace':    [ 1, -1, -1, -1,  1],  # peace
    'handOk':       [ 1,  1, -1,  1, -1],  # ok
    'handIndex':    [ 1, -1,  1, -1,  1],
    'handRing':     [ 1,  1,  1,  1,  1],
    'handPinky':    [ 1,  1,  1, -1, -1],
    'handExit':     [-1, -1, -1,  1, -1]  # exit
}

# TODO: test method :)
def motion(gesture):
    if gesture != "handRelax":
        for servo in range(0, 5):
            kit.continuous_servo[servo].throttle = gestures_positions[gesture][servo]


def validGestures():
    print("\nValid Gestures:")
    for key in gestures_positions.keys():
        print("    " + key)


def GestureGui():
    validGestures()

    while True:
        userinput = input("\nEnter Gesture: ")
        if userinput == 'exit':
            motion(userinput)
            break
        elif userinput in gestures_positions:
            motion(userinput)
        else:
            print("Invalid Gesture")
            validGestures()


if __name__ == '__main__':
    try:
        GestureGui()
    except KeyboardInterrupt:
        print()
