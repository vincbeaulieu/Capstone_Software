from adafruit_servokit import ServoKit


class ServoController:

    def __init__(self, channel, degree):
        self.kit = ServoKit(channels=16)
        self.channel = channel
        self.age = degree

    def setStandardServoAngle(self, channel, degree):
        self.kit.servo[channel].angle = degree

    def changeStandardServoActuationRange(self, channel, degree):
        self.kit.servo[channel].actuation_range = degree

    def resetStandardServo(self, channel):
        self.kit.servo[channel].angle = 0

    def setContinuousServo(self, channel, throttle):
        self.kit.continuous_servo[channel].throttle = throttle

    def setContinuousServoFullThrottle(self, channel):
        self.kit.continuous_servo[channel].throttle = 1

    def setContinuousServoFullReverseThrottle(self, channel):
        self.kit.continuous_servo[channel].throttle = -1

    def setContinuousServoHalfThrottle(self, channel):
        self.kit.continuous_servo[channel].throttle = 0.5

    def setContinuousServoHalfReverseThrottle(self, channel):
        self.kit.continuous_servo[channel].throttle = -0.5

    def stopContinuousServoThrottle(self, channel):
        self.kit.continuous_servo[channel].throttle = 0
