from datetime import time
from adafruit_servokit import ServoKit
import board
import busio
import adafruit_pca9685


class ServoController:
    # Set channels to the number of servo channels on your Servo Kit.
    kit = ServoKit(channels=16)
    i2c = busio.I2C(board.SCL, board.SDA)
    hat = adafruit_pca9685.PCA9685(i2c)

    # For constructor we define the servo channel pin (0-15) and degree of initial angle usually between 0-180.
    def __init__(self, channel, degree):
        self.channel = channel
        self.degree = degree
        self.kit.servo[channel].angle = degree

    # Set Global frequency of all channel to a specific frequency.
    def setGlobalFrequency(self, frequency):
        self.hat.frequency = frequency

    # Set specific channel frequency.
    def setChannelFrequency(self, channel, frequency):
        led_channel = self.hat.channels[channel]
        led_channel.frequency = frequency

    # Get specific channel.
    def getChannel(self, channel):
        return self.hat.channels[channel]

    # Set Brightness of specific channel to either  0% , 25%, 50%, 75% or 100% using case values
    # [0, 0.25, 0.5, 0.75, 1].
    def setLedBrightness(self, channel, case):
        led_channel = self.hat.channels[channel]
        if case == 0:
            led_channel.duty_cycle = 0
        elif case == 0.25:
            led_channel.duty_cycle = 16384
        elif case == 0.5:
            led_channel.duty_cycle = 32768
        elif case == 0.75:
            led_channel.duty_cycle = 49151
        elif case == 1:
            led_channel.duty_cycle = 65535

    # Set the angle of a standard servo on a specific channel.
    def setStandardServoAngle(self, channel, degree):
        self.kit.servo[channel].angle = degree

    # Change the actuation range of a specific servo by default 180.
    def setStandardServoActuationRange(self, channel, degree):
        self.kit.servo[channel].actuation_range = degree

    # Set the pulse width range maximum and minimum.
    def setPulseWidthRange(self, channel, minimum, maximum):
        self.kit.servo[channel].set_pulse_width_range(minimum, maximum)

    # Reset the servo angle back to 0.
    def resetStandardServo(self, channel):
        self.kit.servo[channel].angle = 0

    # Set continuous servo to a throttle ranging from -1 to 1.
    def setContinuousServo(self, channel, throttle):
        self.kit.continuous_servo[channel].throttle = throttle

    # Set continuous servo to a full throttle.
    def setContinuousServoFullThrottle(self, channel):
        self.kit.continuous_servo[channel].throttle = 1

    # Set continuous servo to a full reverse throttle.
    def setContinuousServoFullReverseThrottle(self, channel):
        self.kit.continuous_servo[channel].throttle = -1

    # Set continuous servo to a half throttle.
    def setContinuousServoHalfThrottle(self, channel):
        self.kit.continuous_servo[channel].throttle = 0.5

    # Set continuous servo to a half reverse throttle.
    def setContinuousServoHalfReverseThrottle(self, channel):
        self.kit.continuous_servo[channel].throttle = -0.5

    # Stop continuous servo.
    def stopContinuousServoThrottle(self, channel):
        self.kit.continuous_servo[channel].throttle = 0


def test():
    servo = ServoController(0, 180)
    time.sleep(1)
    servo.setStandardServoAngle(1, 90)
    time.sleep(1)

    for x in range(1, 80):
        servo.setLedBrightness(2, 0.25)
        time.sleep(1)
        servo.setLedBrightness(2, 0.5)
        time.sleep(1)
        servo.setLedBrightness(2, 0.75)
        time.sleep(1)
        servo.setLedBrightness(2, 1)


if __name__ == "__main__":
    test()
