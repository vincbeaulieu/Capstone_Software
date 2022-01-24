# https://stackoverflow.com/questions/59348014/how-can-i-generate-the-random-number-two-times-in-this-code

import numpy as np
from time import sleep

# Generate a random number
def generate(scale_factor):
    noise = np.random.normal(loc=0.0, scale=scale_factor, size=None)
    return noise

# Threading solution
def test(scale_factor, frequency_hz):
    period_sec = 1/frequency_hz
    while 1:
        sleep(period_sec)
        random_num = generate(scale_factor)