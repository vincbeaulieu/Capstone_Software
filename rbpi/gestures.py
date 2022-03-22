
gestures_positions = {
    'handClose':    [ 1,  1,  1, -1,  1],  # close hand
    'handOpen':     [-1, -1, -1,  1, -1],  # open hand
    'handThumbsUp': [-1,  1,  1, -1,  1],  # thumbs up
    'handFlip':     [ 1,  1, -1, -1,  1],  # major up
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

