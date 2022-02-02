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
    #noise_generator.test() # placeholder

    servo = ServoController(0,1)
    pass

if __name__ == '__main__':
    main()


