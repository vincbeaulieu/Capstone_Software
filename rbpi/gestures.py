import random

import numpy as np

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

# hand Relax hold position
gestures_list = ['handRelax'] + list(gestures_positions.keys())

# Excluded gestures
handRemoved = ['handPeace', 'handRock', 'handOk', 'handFlip', 'handExit']

# Remove unwanted gesture
gestures = [g for g in gestures_list if g not in handRemoved]
random.shuffle(gestures)

class Gestures:
    def __init__(self, gestures=None, excluded_gestures=None):
        if gestures is None:
            gestures = gestures_list
        self.gestures = gestures

        if excluded_gestures is None:
            excluded_gestures = handRemoved
        self.gestures = [g for g in self.gestures if g not in excluded_gestures]
        self.excluded_gestures = excluded_gestures

    def unique(self, data):
        self.gestures = np.unique(np.array(data))
        return self

    def remove(self, excluded_gestures=None):
        self.gestures = [g for g in self.gestures if g not in excluded_gestures]
        self.excluded_gestures.extend(excluded_gestures)
        return self

    def update(self, data, excluded_gestures):
        self.unique(data)
        self.gestures = [g for g in self.gestures if g not in excluded_gestures]
        return self

    def list(self):
        return self.gestures

    def excluded(self):
        return self.excluded_gestures


gesture_object = Gestures()


if __name__ == "__main__":
    gestures_object = Gestures().update(gestures_list, ['handOpen', 'handClose'])
    list_of_gestures = gestures_object.excluded()
    print(list_of_gestures)