
import MachineLearning as ML
import toolbox
import noise_generator
import ServoController from servoController

def main():
    test()
    return 0

def test():
    print("Calling Main Test Function...")
    # Comment the test when done
    
    #toolbox.test()
    #ML.test()
    #noise_generator.test() # placeholder

    servo = ServoController(0, 0);
    servo.setContinuousServo(0, 1);
    servo.setContinuousServo(1, -1);
    pass

if __name__ == '__main__':
    main()


