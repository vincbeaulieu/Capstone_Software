import ml_training.MachineLearning as ML
import toolbox
from servoController import ServoController

def main():
    test()
    return 0

def test():
    print("Calling Main Test Function...")
    # Comment the test when done
    
    #toolbox.test()
    #ML.test()

    servo = ServoController(0, 0)
    servo.setContinuousServo(0, 1)
    servo.setContinuousServo(1, -1)
    pass

if __name__ == '__main__':
    main()


