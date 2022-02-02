#import MachineLearning as ML
#import toolbox
#import noise_generator
import time
from servoController import ServoController

def main():
    test()
    return 0


def test():
    print("Calling Main Test Function...")
    # Comment the test when done
    # toolbox.test()
    # ML.test()
    # noise_generator.test() # placeholder

    servo1 = ServoController(0, 1)
    servo1.setContinuousServoFullReverseThrottle()
    servo1.resetContinuousServoThrottle()

    servo2 = ServoController(1, 1)
    servo2.setContinuousServoFullReverseThrottle()
    servo2.resetContinuousServoThrottle()

    pass


if __name__ == '__main__':
    main()
